"""Compute static code-graph features for every SWE-bench Verified task.

Usage: python -I src/features.py data/verified_evals.json repos/ out/features_verified.csv
Each task's repository is analysed at its exact base_commit.
"""
import csv
import json
import ast
import re
import sys
import time
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict

sys.path.insert(0, __import__("os").path.dirname(__file__))
from codegraph import CodeGraph, GitRepo  # noqa: E402

STOP = set("""the and for with this that from have not but are was were when what which
then than into onto your you our can could should would will also only just like
none true false self cls args kwargs return value values type types data test tests
error errors issue bug fix code file files line lines python version print example
expected actual result results output input default object objects list dict str int
float bool set get call class function method module package import use used using""".split())

TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
TRACE = re.compile(r'File "([^"]+\.py)", line \d+, in ([A-Za-z_][A-Za-z0-9_<>]*)')


def issue_tokens(text):
    """Identifier-like tokens from an issue: dotted names, snake_case, CamelCase, traceback frames."""
    toks = set()
    trace_funcs, trace_files = set(), set()
    for f, fn in TRACE.findall(text):
        trace_files.add(f.replace("\\", "/"))
        trace_funcs.add(fn)
    for m in TOK.findall(text):
        parts = m.split(".")
        for i, p in enumerate(parts):
            code_like = ("_" in p or (p[:1].isupper() and any(c.isupper() for c in p[1:]))
                         or len(parts) > 1)
            if len(p) >= 3 and p.lower() not in STOP and code_like:
                toks.add(p)
    # words followed by "(" are calls even if plain lowercase: foo(
    for m in re.findall(r"([A-Za-z_][A-Za-z0-9_]{2,})\(", text):
        if m.lower() not in STOP:
            toks.add(m)
    toks |= {t for t in trace_funcs if t.lower() not in STOP and not t.startswith("<")}
    return toks, trace_files


def patch_stats(patch):
    files, hunks, add, rem = set(), 0, 0, 0
    for line in patch.splitlines():
        if line.startswith("diff --git"):
            files.add(line.split(" b/")[-1])
        elif line.startswith("@@"):
            hunks += 1
        elif line.startswith("+") and not line.startswith("+++"):
            add += 1
        elif line.startswith("-") and not line.startswith("---"):
            rem += 1
    return dict(patch_files=len(files), patch_hunks=hunks, patch_added=add,
                patch_removed=rem, patch_lines=add + rem)


def task_features(g: CodeGraph, task):
    sp = task["expected_spans"]
    spans = sp if isinstance(sp, dict) else ast.literal_eval(sp)
    gold, unresolved = [], 0
    for f, ss in spans.items():
        for s in ss:
            if s in ("imports", "module"):
                continue
            nid = g.resolve_span(f, s)
            if nid:
                gold.append(nid)
            else:
                unresolved += 1
    gold = sorted(set(gold))
    gold_files = sorted(spans)
    text = task["problem_statement"]
    toks, trace_files = issue_tokens(text)

    anchors_all, anchors_spec = set(), set()
    for t in toks:
        c = g.by_short.get(t, [])
        anchors_all.update(c)
        if 0 < len(c) <= 5:
            anchors_spec.update(c)

    shorts = {g.nodes[n]["qual"].rsplit(".", 1)[-1] for n in gold}
    gold_named = int(bool(shorts & toks))
    low = text.lower()
    gold_file_in_issue = 0
    for f in gold_files:
        mod = f[:-3].replace("/", ".")
        base = f.rsplit("/", 1)[-1]
        if f in text or mod in text or any(tf.endswith(f) for tf in trace_files) or \
           (len(base) > 6 and base.lower() in low and base not in ("models.py", "utils.py", "base.py", "__init__.py")):
            gold_file_in_issue = 1
    traceback_present = int(bool(trace_files))

    CAP = 8
    def hop(src):
        if not src or not gold:
            return CAP + 1
        dist = g.bfs(src, CAP)
        ds = [dist[n] for n in gold if n in dist]
        return min(ds) if ds else CAP + 1

    hop_spec = hop(anchors_spec)
    hop_all = hop(anchors_all)

    amb = [len(g.by_short[g.nodes[n]["qual"].rsplit(".", 1)[-1]]) for n in gold] or [0]
    fanin = [sum(g.rcalls.get(n, {}).values()) for n in gold] or [0]
    fanout = [sum(g.calls.get(n, {}).values()) for n in gold] or [0]
    lines = [g.nodes[n]["lines"] for n in gold] or [0]
    # pairwise spread among gold nodes
    spread = 0
    if len(gold) > 1:
        for i, a in enumerate(gold[:8]):
            dist = g.bfs([a], CAP)
            for b in gold[i + 1:8]:
                spread = max(spread, dist.get(b, CAP + 1))
    # 2-hop blast radius
    if gold:
        d2 = g.bfs(gold, 2)
        blast2 = len(d2) - len(gold)
    else:
        blast2 = 0
    file_defs = max((len(g.by_file.get(f, [])) for f in gold_files), default=0)

    feat = dict(
        instance_id=task["instance_id"], repo=task["repo"],
        repo_nodes=len(g.nodes), repo_files=g.n_files,
        n_gold_nodes=len(gold), n_gold_files=len(gold_files), unresolved_spans=unresolved,
        n_issue_tokens=len(toks), n_anchors_all=len(anchors_all), n_anchors_spec=len(anchors_spec),
        gold_named_in_issue=gold_named, gold_file_in_issue=gold_file_in_issue,
        traceback_present=traceback_present,
        hop_spec=hop_spec, hop_all=hop_all,
        gold_ambiguity_max=max(amb), gold_ambiguity_mean=sum(amb) / len(amb),
        gold_fanin_max=max(fanin), gold_fanout_max=max(fanout),
        gold_lines_sum=sum(lines), gold_spread=spread, blast2=blast2, gold_file_defs=file_defs,
        issue_chars=len(text),
    )
    feat.update(patch_stats(task["golden_patch"]))
    return feat


def main(tasks_path, repos_dir, out_path, only_repo=None):
    tasks = json.load(open(tasks_path))
    by_repo = defaultdict(list)
    for t in tasks:
        if only_repo and t["repo"] != only_repo:
            continue
        by_repo[t["repo"]].append(t)
    fields, rows = None, []
    t0 = time.time()
    for repo, ts in sorted(by_repo.items(), key=lambda kv: len(kv[1])):
        gr = GitRepo(f"{repos_dir}/{repo.replace('/', '_')}.git")
        for i, t in enumerate(ts):
            g = CodeGraph(gr, t["base_commit"])
            f = task_features(g, t)
            rows.append(f)
            fields = fields or list(f)
            if i % 10 == 0:
                print(f"{repo} {i+1}/{len(ts)} nodes={f['repo_nodes']} hop={f['hop_spec']} "
                      f"t={time.time()-t0:.0f}s", flush=True)
        gr.close()
        with open(out_path, "w", newline="") as fh:   # checkpoint after each repo
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print("done", len(rows), f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(*sys.argv[1:])
