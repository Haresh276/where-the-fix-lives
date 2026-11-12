"""Rebuild the competition's public tasks that have public Gemma 4 31B outcomes, from public git history.

Task ids are <repo>_<PR number>. For each: find the merged commit for that PR, take its parent as the base,
derive fix spans from the diff, compute fix-location features at the base commit, and flag tasks whose
tests need packages that are missing from the official sandbox (inline_snapshot, dirty_equals).
No competition data is used or needed.  Output: out/features_gemma_public55.csv, out/gemma_public_joined.csv
"""
import os
import re
import subprocess
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from codegraph import CodeGraph, GitRepo  # noqa: E402
from diffspans import git_diff, is_source, spans_from_patch  # noqa: E402

G = {"fastapi": "fastapi_fastapi", "requests": "psf_requests", "rich": "Textualize_rich", "httpx": "encode_httpx"}
O = pd.read_csv("data/gemma/pub/yosh_task_outcomes.csv")
OWN = "data/gemma/own/gemma31b_task_results_compact.csv"          # my own 129-task Gemma 4 31B run on Kaggle (4xL4)
ALL = "--all" in sys.argv
TASKS = sorted(pd.read_csv(OWN).instance_id) if ALL else sorted(O.task_id.unique())


def git(path, *a):
    return subprocess.run(["git", "-C", path, *a], capture_output=True, text=True, errors="ignore").stdout


def find_commit(path, n):
    log = git(path, "log", "--all", "--format=%H%x09%P%x09%s", "-E", "--grep", rf"#{n}\b")
    rows = [l.split("\t") for l in log.splitlines()]
    squash = [r for r in rows if re.search(rf"\(#{n}\)\s*$", r[2]) and len(r[1].split()) == 1]
    merge = [r for r in rows if f"Merge pull request #{n} " in r[2] + " "]
    pick = (squash or merge or [None])[0]
    return (pick[0], pick[1].split()[0], pick[2]) if pick else None


os.makedirs("out", exist_ok=True)
rows = []
for t in TASKS:
    r, n = t.rsplit("_", 1)
    path = f"repos/{G[r]}.git"
    found = find_commit(path, n)
    if not found:
        print("not found", t)
        continue
    sha, base, subject = found
    gr = GitRepo(path)
    g = CodeGraph(gr, base)
    spans = spans_from_patch(g, git_diff(path, base, sha))
    gold = [f"{f}::{q}" for f, qs in spans.items() for q in qs if f"{f}::{q}" in g.nodes]
    files = [l.split("\t")[-1] for l in git(path, "diff", "--numstat", base, sha).splitlines()]
    numstat = [l.split("\t") for l in git(path, "diff", "--numstat", base, sha).splitlines()]
    needs = any(("inline_snapshot" in git(path, "show", f"{sha}:{f}") or "dirty_equals" in git(path, "show", f"{sha}:{f}"))
                for f in files if f.endswith(".py") and not is_source(f))
    spread = 0
    for i, a in enumerate(gold[:8]):
        d = g.bfs([a], 8)
        for b in gold[i + 1:8]:
            spread = max(spread, d.get(b, 9))
    rows.append(dict(task_id=t, repo=r, repo_nodes=len(g.nodes), n_gold_files=len(spans), n_gold_nodes=len(gold),
                     gold_spread=spread,
                     gold_fanin_max=max([sum(g.rcalls.get(x, {}).values()) for x in gold] or [0]),
                     gold_ambiguity_max=max([len(g.by_short[g.nodes[x]["qual"].rsplit(".", 1)[-1]]) for x in gold] or [0]),
                     blast2=(len(g.bfs(gold, 2)) - len(gold)) if gold else 0,
                     src_patch_lines=sum(int(a) + int(b) for a, b, f in numstat if a != "-" and is_source(f)),
                     needs_missing_pkg=needs, title=subject,
                     gold_files=";".join(f for a_, b_, f in numstat if is_source(f))))
    gr.close()
F = pd.DataFrame(rows).set_index("task_id")
if ALL:
    F.to_csv("out/features_gemma_own129.csv")
    F.join(pd.read_csv(OWN).set_index("instance_id"), how="inner").to_csv("out/gemma_own_joined.csv")
    print(len(F), "of", len(TASKS), "tasks rebuilt;", int(F.needs_missing_pkg.sum()), "flagged")
    sys.exit()
F.to_csv("out/features_gemma_public55.csv")
Oa = O[O.arm.isin(["v8_noanz@ollama55", "v9_verify@ollama55"])]          # the two 55-task configurations
F.join(Oa.groupby("task_id").resolved.agg(["mean", "count"]), how="inner").to_csv("out/gemma_public_joined.csv")
print(len(F), "tasks rebuilt;", int(F.needs_missing_pkg.sum()), "flagged as unsolvable in the official sandbox")
