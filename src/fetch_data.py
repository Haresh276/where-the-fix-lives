"""Download the public inputs (not redistributed here) and clone the repositories.

python src/fetch_data.py          # ~2 GB of bare git clones, a few MB of JSON/CSV
Creates data/ and repos/ next to src/.
"""
import ast
import csv
import json
import os
import subprocess
import urllib.request

RAW = "https://raw.githubusercontent.com"
os.makedirs("data", exist_ok=True)
os.makedirs("repos", exist_ok=True)


def get(url, path):
    if not os.path.exists(path):
        print("fetch", url)
        urllib.request.urlretrieve(url, path)


# SWE-bench Verified tasks with curated fix spans (vendored by moatless-tools)
get(f"{RAW}/aorwall/moatless-tools/main/moatless/evaluation/swebench_verified_all_evaluations.json",
    "data/verified_evals.json")
# per-task outcomes of every leaderboard run
get(f"{RAW}/SWE-bench/swe-bench.github.io/master/data/leaderboards.json", "data/leaderboards.json")
# Agent Psychometrics (Ge et al., 2026): IRT difficulties and LLM-judged features
AP = f"{RAW}/dariakryvosheieva/agent-psychometrics/main"
get(f"{AP}/data/swebench_verified/irt/1d_1pl/items.csv", "data/ap_irt_items.csv")
get(f"{AP}/llm_judge_features/defaults/swebench_verified/llm_judge_features.csv", "data/ap_llm_judge.csv")
# public Gemma 4 31B per-task outcomes on competition tasks (yosh95, Apache-2.0)
os.makedirs("data/gemma/pub", exist_ok=True)
get(f"{RAW}/yosh95/g4-developer-agent-paper/main/paper/task_outcomes.csv", "data/gemma/pub/yosh_task_outcomes.csv")

# flatten leaderboard -> outcomes.csv + agents.json (Verified board, runs with per-instance details)
lb = [b for b in json.load(open("data/leaderboards.json"))["leaderboards"] if b["name"] == "Verified"][0]["results"]
rows, agents = [], []
for r in lb:
    pid = r.get("per_instance_details")
    if not pid or pid == "None":
        continue
    if isinstance(pid, str):
        pid = json.loads(pid) if pid.startswith('{"') else ast.literal_eval(pid)
    agents.append(dict(agent=r["folder"], name=r.get("name"), model=r.get("model_display"), org=r.get("model_org"),
                       os_model=r.get("os_model"), date=r.get("date"), resolved=r.get("resolved"), n=len(pid)))
    for iid, v in pid.items():
        rows.append((r["folder"], iid, int(bool(v.get("resolved"))), v.get("api_calls"), v.get("cost")))
json.dump(agents, open("data/agents.json", "w"), indent=1)
with open("data/outcomes.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["agent", "instance_id", "resolved", "api_calls", "cost"])
    w.writerows(rows)
print(len(agents), "runs with per-instance results")

REPOS = ["psf/requests", "pallets/flask", "mwaskom/seaborn", "pylint-dev/pylint", "pytest-dev/pytest", "pydata/xarray",
         "astropy/astropy", "scikit-learn/scikit-learn", "sphinx-doc/sphinx", "sympy/sympy", "django/django",
         "matplotlib/matplotlib", "fastapi/fastapi", "encode/httpx", "Textualize/rich"]
for r in REPOS:
    d = f"repos/{r.replace('/', '_')}.git"
    if not os.path.isdir(d):
        subprocess.run(["git", "clone", "--bare", f"https://github.com/{r}.git", d], check=True)
