"""Cheap CPU localizer (BM25 over function/class definitions) + deployable triage features.

For each task: rank every definition at base_commit by BM25 against the issue text,
then describe the *predicted* fix location with the same structural features used
for the gold location. Nothing here reads the gold patch except the evaluation columns
(loc_* hit metrics), which are kept separate from the deployable features.
Usage: python -I src/localize.py data/verified_evals.json repos out/localize_verified.csv
"""
import csv
import json
import math
import re
import sys
import time
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, __import__("os").path.dirname(__file__))
from codegraph import CodeGraph, GitRepo, _BLOB_CACHE  # noqa: E402
from features import issue_tokens  # noqa: E402

SPLIT = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
STOPW = set("the a an and or of to in is it be for on with as that this by not are was if at from".split())


def toks(text):
    out = []
    for w in WORD.findall(text):
        lw = w.lower()
        if lw not in STOPW and len(lw) > 1:
            out.append(lw)
        for p in SPLIT.findall(w):
            p = p.lower()
            if p != lw and len(p) > 2 and p not in STOPW:
                out.append(p)
    return out


_TEXT_CACHE = {}
_TOK_CACHE = {}


def node_texts(repo, commit, g):
    """Source text of every node (path tokens + qualified name + body)."""
    files = dict(repo.ls_py(commit))
    texts = {}
    for p, nids in g.by_file.items():
        sha = files.get(p)
        if sha is None:
            continue
        lines = _TEXT_CACHE.get(sha)
        if lines is None:
            lines = repo.blob(sha).decode("utf-8", "ignore").splitlines()
            _TEXT_CACHE[sha] = lines
        for n in nids:
            nd = g.nodes[n]
            body = "\n".join(lines[nd["lineno"] - 1: nd["end"]])
            texts[n] = f"{p} {nd['qual']} {nd['qual']} {body}"
    return texts


def bm25_rank(query, texts, k1=1.2, b=0.75):
    q = Counter(toks(query))
    docs = {}
    for n, t in texts.items():
        h = hash(t)
        c = _TOK_CACHE.get(h)
        if c is None:
            c = _TOK_CACHE[h] = Counter(toks(t))
        docs[n] = c
    N = len(docs)
    avg = sum(sum(c.values()) for c in docs.values()) / max(N, 1)
    df = Counter()
    for c in docs.values():
        df.update(set(c) & set(q))
    idf = {w: math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5)) for w in q}
    scores = {}
    for n, c in docs.items():
        L = sum(c.values())
        s = 0.0
        for w in q:
            f = c.get(w)
            if f:
                s += idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * L / avg))
        if s:
            scores[n] = s
    return sorted(scores.items(), key=lambda kv: -kv[1])


def main(tasks_path, repos_dir, out_path):
    __import__("os").makedirs(__import__("os").path.dirname(out_path) or ".", exist_ok=True)
    tasks = json.load(open(tasks_path))
    by_repo = defaultdict(list)
    for t in tasks:
        by_repo[t["repo"]].append(t)
    rows, t0 = [], time.time()
    for repo, ts in sorted(by_repo.items(), key=lambda kv: len(kv[1])):
        gr = GitRepo(f"{repos_dir}/{repo.replace('/', '_')}.git")
        for i, t in enumerate(ts):
            g = CodeGraph(gr, t["base_commit"])
            texts = node_texts(gr, t["base_commit"], g)
            ranked = bm25_rank(t["problem_statement"], texts)
            top = [n for n, _ in ranked[:20]]
            sc = [s for _, s in ranked[:20]] + [0, 0]
            # files ranked by best node
            seen, files = set(), []
            for n in top:
                f = g.nodes[n]["file"]
                if f not in seen:
                    seen.add(f)
                    files.append(f)
            fc = Counter(g.nodes[n]["file"] for n in top[:10])
            ent = -sum(v / 10 * math.log(v / 10) for v in fc.values()) if top else 0
            itoks, _ = issue_tokens(t["problem_statement"])
            spec = set()
            for tk in itoks:
                c = g.by_short.get(tk, [])
                if 0 < len(c) <= 5:
                    spec.update(c)
            hop1 = 9
            if spec and top:
                d = g.bfs(spec, 8)
                hop1 = d.get(top[0], 9)
            p1 = top[0] if top else None
            amb1 = len(g.by_short[g.nodes[p1]["qual"].rsplit(".", 1)[-1]]) if p1 else 0
            fanin1 = sum(g.rcalls.get(p1, {}).values()) if p1 else 0
            lines1 = g.nodes[p1]["lines"] if p1 else 0
            top1_named = int(bool(p1) and g.nodes[p1]["qual"].rsplit(".", 1)[-1] in itoks)
            # evaluation only (uses gold):
            sp = t["expected_spans"]
            gold_files = set(sp)
            gold_short = {s.split(".")[-1] for ss in sp.values() for s in ss}
            row = dict(instance_id=t["instance_id"], repo=repo,
                       loc_top1_score=sc[0], loc_margin=sc[0] - sc[1],
                       loc_margin_rel=(sc[0] - sc[1]) / sc[0] if sc[0] else 0,
                       loc_files_top10=len(fc), loc_file_entropy=ent,
                       pred_hop=hop1, pred_ambiguity=amb1, pred_fanin=fanin1, pred_lines=lines1,
                       pred_named_in_issue=top1_named,
                       # evaluation columns
                       eval_file_top1=int(bool(files) and files[0] in gold_files),
                       eval_file_top5=int(bool(gold_files & set(files[:5]))),
                       eval_func_top10=int(any(g.nodes[n]["file"] in gold_files and
                                               g.nodes[n]["qual"].rsplit(".", 1)[-1] in gold_short
                                               for n in top[:10])))
            rows.append(row)
            if i % 20 == 0:
                print(f"{repo} {i+1}/{len(ts)} t={time.time()-t0:.0f}s", flush=True)
        gr.close()
        _TEXT_CACHE.clear()
        _TOK_CACHE.clear()
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("done", len(rows), f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(*sys.argv[1:])
