"""First-pass analyses. Prints a report; writes out/merged.csv."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

F = pd.read_csv("out/features_verified.csv").set_index("instance_id")
irt = pd.read_csv("data/ap_irt_items.csv", index_col=0)["b"].rename("irt_b")
judge = pd.read_csv("data/ap_llm_judge.csv", index_col=0)
out = pd.read_csv("data/outcomes.csv")
agents = pd.DataFrame(json.load(open("data/agents.json"))).set_index("agent")
agents["rate"] = agents["resolved"].astype(float)

# fixed-scaffold (mini-SWE-agent) solve rates per task, overall and by cohort
M = out.pivot(index="instance_id", columns="agent", values="resolved")
weak = agents.index[agents.rate < 45]          # Qwen2.5-Coder-32B, Llama4 Mav, gpt-oss-120b, GPT5-nano, Kimi K2
strong = agents.index[agents.rate >= 70]
F["solve_all"] = M.mean(1)
F["solve_weak"] = M[weak].mean(1)
F["solve_strong"] = M[strong].mean(1)
F = F.join(irt)
F.to_csv("out/merged.csv")
print("tasks", len(F), "| weak agents", list(agents.loc[weak, "model"]), "| strong", len(strong))

GRAPH = ["hop_spec", "hop_all", "n_anchors_spec", "gold_named_in_issue", "gold_file_in_issue",
         "traceback_present", "gold_ambiguity_max", "gold_fanin_max", "gold_fanout_max",
         "gold_lines_sum", "gold_spread", "blast2", "gold_file_defs", "n_gold_nodes",
         "n_gold_files", "repo_nodes"]
PATCH = ["patch_files", "patch_hunks", "patch_lines"]

print("\n== Spearman with IRT difficulty (134 agents) and fixed-scaffold solve rates")
rows = []
for k in GRAPH + PATCH + ["issue_chars"]:
    rows.append((k, *(spearmanr(F[k], F[t], nan_policy="omit")[0] for t in ["irt_b", "solve_all", "solve_weak", "solve_strong"])))
print(pd.DataFrame(rows, columns=["feature", "rho_irt_b", "rho_all", "rho_weak", "rho_strong"]).round(3).to_string(index=False))

print("\n== Solve rate by hop distance bin")
F["hop_bin"] = pd.cut(F.hop_spec, [-1, 0, 1, 2, 8, 9], labels=["0 named", "1", "2", "3-8", "no anchor"])
print(F.groupby("hop_bin", observed=True)[["solve_weak", "solve_all", "solve_strong"]].agg(["mean", "count"]).round(3))

# ---- predicting IRT difficulty: CV R2 / Spearman
def cv_pred(X, y, groups=None):
    mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
    cv = GroupKFold(5) if groups is not None else KFold(5, shuffle=True, random_state=0)
    return cross_val_predict(mdl, X, y, cv=cv, groups=groups)

D = F.join(judge, how="inner").dropna(subset=["irt_b"])
y = D.irt_b.values
JUDGE = list(judge.columns)
LOG = lambda X: np.log1p(X.clip(lower=0))
sets = {"patch size": D[PATCH].pipe(LOG), "graph (ours)": D[GRAPH].pipe(LOG),
        "graph+patch": D[GRAPH + PATCH].pipe(LOG), "LLM-judge (Opus 4.6)": D[JUDGE],
        "LLM-judge+graph": pd.concat([D[JUDGE], D[GRAPH + PATCH].pipe(LOG)], axis=1)}
print(f"\n== Predict IRT difficulty, n={len(D)} (5-fold random | leave-repo-group-out)")
for name, X in sets.items():
    p = cv_pred(X.values, y)
    pg = cv_pred(X.values, y, groups=D.repo.values)
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"{name:24s} R2={r2:.3f} rho={spearmanr(p, y)[0]:.3f} | repo-out rho={spearmanr(pg, y)[0]:.3f}")

# ---- interaction: does structure hurt weak agents disproportionately? per-agent logistic slope
print("\n== Per-agent logistic coef of hop_spec (z), controlling patch_lines(log) and repo size(log)")
X = pd.DataFrame({"hop": F.hop_spec, "pl": np.log1p(F.patch_lines), "rn": np.log1p(F.repo_nodes),
                  "named": F.gold_named_in_issue})
Xz = (X - X.mean()) / X.std()
res = []
for a in M.columns:
    yy = M[a].reindex(Xz.index).values
    lr = LogisticRegression(C=10, max_iter=1000).fit(Xz.values, yy)
    res.append((a, agents.loc[a, "model"], agents.loc[a, "rate"], agents.loc[a, "os_model"], *lr.coef_[0]))
R = pd.DataFrame(res, columns=["agent", "model", "rate", "open", "b_hop", "b_patch", "b_repo", "b_named"]).sort_values("rate")
print(R.round(3).to_string(index=False))
print("spearman(rate, b_hop)=", round(spearmanr(R.rate, R.b_hop)[0], 3), "p=", round(spearmanr(R.rate, R.b_hop)[1], 4))
print("spearman(rate, b_named)=", round(spearmanr(R.rate, R.b_named)[0], 3), "p=", round(spearmanr(R.rate, R.b_named)[1], 4))
R.to_csv("out/per_agent_slopes.csv", index=False)
