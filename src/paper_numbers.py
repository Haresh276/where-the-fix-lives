"""Every number in the paper comes from this script.  Output: out/numbers.json (+ printed report).

Run from the repo root after features.py / localize.py:
    python src/paper_numbers.py [--boot 1000]
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, fisher_exact
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
B = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else 1000
rng = np.random.default_rng(2026)
N = {}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


# ------------------------------------------------------------------ data
F = pd.read_csv("out/features_verified.csv").set_index("instance_id")
L = pd.read_csv("out/localize_verified.csv").set_index("instance_id").drop(columns="repo")
F = F.join(L)
irt = pd.read_csv("data/ap_irt_items.csv", index_col=0)["b"].rename("irt_b")
judge = pd.read_csv("data/ap_llm_judge.csv", index_col=0)
out = pd.read_csv("data/outcomes.csv")
A = pd.DataFrame(json.load(open("data/agents.json"))).set_index("agent")
M = out.pivot(index="instance_id", columns="agent", values="resolved").loc[F.index]
A["rate"] = A.resolved.astype(float)
A["emp"] = M.mean() * 100
bad = A.index[(A.emp - A.rate).abs() > 5]          # per-instance files inconsistent with reported score
N["agents_dropped"] = list(A.loc[bad, "model"])
A, M = A.drop(bad), M.drop(columns=bad)
N["n_tasks"], N["n_agents"] = int(len(F)), int(M.shape[1])
N["n_open_weight"] = int((A.os_model.astype(str) == "True").sum())
N["rate_range"] = [float(A.rate.min()), float(A.rate.max())]
WEAK = A.index[A.rate < 45]
STRONG = A.index[A.rate >= 70]
N["weak_models"] = list(A.loc[WEAK].sort_values("rate").model)
N["n_strong"] = int(len(STRONG))
F["multi_file"] = (F.n_gold_files > 1).astype(int)
F["multi_node"] = (F.n_gold_nodes > 1).astype(int)
N["graph"] = dict(nodes_median=int(F.repo_nodes.median()), nodes_max=int(F.repo_nodes.max()),
                  unresolved_span_rate=round(F.unresolved_spans.sum() / (F.unresolved_spans.sum() + F.n_gold_nodes.sum()), 3),
                  frac_multi_file=round(F.multi_file.mean(), 3), frac_named=round(F.gold_named_in_issue.mean(), 3),
                  frac_file_in_issue=round(F.gold_file_in_issue.mean(), 3),
                  hop_dist={str(k): int(v) for k, v in F.hop_spec.value_counts().sort_index().items()})

# ------------------------------------------------------------------ R1/R2: pooled model, agent FE, feature x ability
FEAT = {"hop": F.hop_spec, "patch_lines(log)": np.log1p(F.patch_lines), "files(log)": np.log1p(F.n_gold_files),
        "func_named": F.gold_named_in_issue, "file_named": F.gold_file_in_issue, "spread": F.gold_spread,
        "issue_len(log)": np.log1p(F.issue_chars)}
X = pd.DataFrame(FEAT)
Xz = (X - X.mean()) / X.std()
lg = np.log(A.rate / (100 - A.rate))
ab = (lg - lg.mean()) / lg.std()
Ylong = M.stack().rename("y").reset_index()
Ylong.columns = ["iid", "agent", "y"]
Ylong = Ylong.join(Xz, on="iid")
Ylong["ab"] = Ylong.agent.map(ab)
cols = list(X.columns)
D_agent = pd.get_dummies(Ylong.agent, dtype=float).values


def pooled(df, dummies):
    Z = np.hstack([dummies, df[cols].values, df[cols].values * df[["ab"]].values])
    m = LogisticRegression(C=1e4, max_iter=5000).fit(Z, df.y.values)
    k = dummies.shape[1]
    return m.coef_[0][k:k + len(cols)], m.coef_[0][k + len(cols):]


main, inter = pooled(Ylong, D_agent)
iids = F.index.values
pos = {i: np.where(Ylong.iid.values == i)[0] for i in iids}
BM, BI = [], []
for b in range(B):
    s = rng.choice(iids, len(iids))
    idx = np.concatenate([pos[i] for i in s])
    m_, i_ = pooled(Ylong.iloc[idx], D_agent[idx])
    BM.append(m_)
    BI.append(i_)
BM, BI = np.array(BM), np.array(BI)
# leave-one-repo-out (Django is 46% of tasks)
loro = {}
for repo in F.repo.unique():
    keep = Ylong.iid.map(F.repo) != repo
    loro[repo] = pooled(Ylong[keep], D_agent[keep.values])
N["pooled"] = {c: dict(main=round(main[j], 3), main_ci=[round(np.percentile(BM[:, j], 2.5), 3), round(np.percentile(BM[:, j], 97.5), 3)],
                       inter=round(inter[j], 3), inter_ci=[round(np.percentile(BI[:, j], 2.5), 3), round(np.percentile(BI[:, j], 97.5), 3)],
                       inter_sign_stable_loro=int(sum(np.sign(v[1][j]) == np.sign(inter[j]) for v in loro.values())),
                       main_sign_stable_loro=int(sum(np.sign(v[0][j]) == np.sign(main[j]) for v in loro.values())))
               for j, c in enumerate(cols)}
N["n_repos"] = int(F.repo.nunique())
N["no_django"] = {c: dict(main=round(loro["django/django"][0][j], 3), inter=round(loro["django/django"][1][j], 3))
                  for j, c in enumerate(cols)}
N["boot_reps"] = B

N["dropped_runs"] = list(bad)


def pooled_ci(Xraw, keep_agents=None, reps=None):
    """Pooled agent-FE logit on an arbitrary feature frame; task bootstrap CI.  Returns {col: [est, lo, hi]}."""
    reps = reps or B
    Mx = M if keep_agents is None else M[keep_agents]
    Z = (Xraw - Xraw.mean()) / Xraw.std()
    cs = list(Z.columns)
    Y = Mx.stack().rename("y").reset_index()
    Y.columns = ["iid", "agent", "y"]
    Y = Y.join(Z, on="iid")
    D = pd.get_dummies(Y.agent, dtype=float).values

    def fit(df, Dm):
        m = LogisticRegression(C=1e4, max_iter=5000, fit_intercept=False).fit(np.hstack([Dm, df[cs].values]), df.y.values)
        return m.coef_[0][Dm.shape[1]:]
    est = fit(Y, D)
    ps = {i: np.where(Y.iid.values == i)[0] for i in iids}
    bs = []
    for _ in range(reps):
        ix = np.concatenate([ps[i] for i in rng.choice(iids, len(iids))])
        bs.append(fit(Y.iloc[ix], D[ix]))
    bs = np.array(bs)
    return {c: [round(est[j], 3), round(np.percentile(bs[:, j], 2.5), 3), round(np.percentile(bs[:, j], 97.5), 3)] for j, c in enumerate(cs)}


# hop recoded: "no usable anchor" (hop = 9) as its own indicator, distance capped at 4 among anchored tasks
BASEF = {k: v for k, v in FEAT.items() if k != "hop"}
anch = F.hop_spec < 9
X_re = pd.DataFrame(dict(BASEF, hop_anchored=F.hop_spec.where(anch, 0).clip(upper=4), no_anchor=(~anch).astype(int)))
N["hop_recoded"] = pooled_ci(X_re)
sd_na = float(X_re.no_anchor.std())
N["hop_recoded"]["no_anchor_per_task"] = [round(v / sd_na, 3) for v in N["hop_recoded"]["no_anchor"]]   # no anchor vs anchored
N["hop_recoded"]["no_anchor_odds_ratio"] = round(float(np.exp(N["hop_recoded"]["no_anchor_per_task"][0])), 2)
N["hop_recoded"]["n_no_anchor"] = int((~anch).sum())
N["hop_recoded"]["n_no_anchor_true"] = int((F.n_anchors_spec == 0).sum())
# anchored tasks only, original hop
keep_t = F.index[anch]
Xa = pd.DataFrame(FEAT).loc[keep_t]
Ma = M.loc[keep_t]
_M, _iids = M, iids
M, iids = Ma, keep_t.values
N["hop_anchored_only"] = pooled_ci(Xa)
M, iids = _M, _iids
if "hop_spec_call" in F:
    N["hop_call_only"] = pooled_ci(pd.DataFrame(dict(BASEF, hop=F.hop_spec_call)))
    N["hop_call_only"]["spearman_with_mixed"] = round(spearmanr(F.hop_spec, F.hop_spec_call)[0], 3)
    N["hop_call_only"]["n_no_path"] = int((F.hop_spec_call == 9).sum())

# scaffold versions: runs span several mini-SWE-agent releases -> refit within each release with >= 5 models
A["version"] = A.index.str.extract(r"mini-(v[0-9.]+)", expand=False).values
N["versions"] = {v: int(c) for v, c in A.version.value_counts().items()}
N["within_version"] = {}
for v, c in A.version.value_counts().items():
    if c >= 5:
        r = pooled_ci(X, keep_agents=list(A.index[A.version == v]), reps=max(B // 5, 100))
        N["within_version"][v] = dict(n_models=int(c), rate_range=[float(A[A.version == v].rate.min()), float(A[A.version == v].rate.max())],
                                      hop=r["hop"], files=r["files(log)"])
N["version_weak"] = {A.loc[a, "model"]: A.loc[a, "version"] for a in WEAK}

# per-agent models (same covariates) -> sign counts and slope vs ability
per = []
for a in M.columns:
    lr = LogisticRegression(C=10, max_iter=3000).fit(Xz.values, M[a].values)
    per.append(dict(zip(cols, lr.coef_[0]), agent=a, model=A.loc[a, "model"], rate=A.loc[a, "rate"]))
P = pd.DataFrame(per)
P.to_csv("out/per_agent_coefs.csv", index=False)
N["per_agent"] = {c: dict(n_negative=int((P[c] < 0).sum()), rho_with_rate=round(spearmanr(P.rate, P[c])[0], 3),
                          p=round(spearmanr(P.rate, P[c])[1], 4)) for c in cols}
Xc = sm.add_constant(Xz)
nsig = {c: 0 for c in cols}
for a in M.columns:
    r = sm.Logit(M[a].values, Xc.values).fit(disp=0)
    for j, c in enumerate(cols):
        nsig[c] += int((r.params[j + 1] < 0) and (r.pvalues[j + 1] < 0.05))
for c in cols:
    N["per_agent"][c]["n_negative_p05"] = nsig[c]
# null: shuffle hop across tasks (same shuffle for every model) -> how often do >= 34 / all 38 models share a sign?
jh = cols.index("hop")
cnt = []
for _ in range(200):
    Xp = Xz.values.copy()
    Xp[:, jh] = Xp[rng.permutation(len(Xp)), jh]
    cnt.append(sum(LogisticRegression(C=10, max_iter=2000).fit(Xp, M[a].values).coef_[0][jh] < 0 for a in M.columns))
cnt = np.array(cnt)
k = M.shape[1]
N["per_agent_null"] = dict(share_all_same_sign=round(float(np.mean((cnt == 0) | (cnt == k))), 3),
                           share_ge34_same_sign=round(float(np.mean((cnt <= k - 34) | (cnt >= 34))), 3))

# ------------------------------------------------------------------ cohort tables (descriptive)
def cohort(mask):
    r = {}
    for name, ag in [("weak", WEAK), ("strong", STRONG)]:
        sub = M.loc[mask, ag]
        k, n = int(sub.values.sum()), int(sub.size)
        t = sub.mean(axis=1).values                     # per-task mean over the cohort's models
        bs = [t[rng.integers(0, len(t), len(t))].mean() for _ in range(2000)]
        r[name] = dict(rate=round(k / n, 3), ci=[round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3)],
                       ci_wilson_naive=wilson(k, n), n_tasks=int(mask.sum()))
    r["ratio_weak_over_strong"] = round(r["weak"]["rate"] / r["strong"]["rate"], 3)
    return r


N["cohort"] = {
    "single_file": cohort(F.multi_file == 0), "multi_file": cohort(F.multi_file == 1),
    "single_func": cohort(F.multi_node == 0), "multi_func": cohort(F.multi_node == 1),
    "file_named": cohort(F.gold_file_in_issue == 1), "file_not_named": cohort(F.gold_file_in_issue == 0),
    "hop0": cohort(F.hop_spec == 0), "hop1": cohort(F.hop_spec == 1), "hop2plus": cohort((F.hop_spec >= 2) & (F.hop_spec <= 8)),
    "no_anchor": cohort(F.hop_spec == 9),
    "single_func_small_patch": cohort((F.multi_node == 0) & (F.patch_lines <= 10)),
    "multi_func_small_patch": cohort((F.multi_node == 1) & (F.patch_lines <= 10)),
}
lo_ = lambda p: np.log(p / (1 - p))
c_ = N["cohort"]
N["multi_file_gap"] = {g: dict(abs_pts=round(100 * (c_["single_file"][g]["rate"] - c_["multi_file"][g]["rate"]), 1),
                               log_odds=round(lo_(c_["single_file"][g]["rate"]) - lo_(c_["multi_file"][g]["rate"]), 2)) for g in ("weak", "strong")}
sm10 = F[F.patch_lines <= 10]
N["small_patch_mean_lines"] = dict(single_func=round(sm10[sm10.multi_node == 0].patch_lines.mean(), 2),
                                   multi_func=round(sm10[sm10.multi_node == 1].patch_lines.mean(), 2))
# per-model single vs multi function (for the Gemma comparison table)
N["by_model_single_vs_multi_func"] = {A.loc[a, "model"] + f" ({A.loc[a,'rate']:.0f}%)": [round(M.loc[F.multi_node == 0, a].mean(), 3),
                                                                                       round(M.loc[F.multi_node == 1, a].mean(), 3)]
                                      for a in A.sort_values("rate").index}

# ------------------------------------------------------------------ R3a: predicting generic difficulty (IRT b)
GRAPH = ["hop_spec", "hop_all", "n_anchors_spec", "gold_named_in_issue", "gold_file_in_issue", "traceback_present",
         "gold_ambiguity_max", "gold_fanin_max", "gold_fanout_max", "gold_lines_sum", "gold_spread", "blast2",
         "gold_file_defs", "n_gold_nodes", "n_gold_files", "repo_nodes"]
PATCH = ["patch_files", "patch_hunks", "patch_lines"]
DD = F.join(irt, how="inner").join(judge, how="inner")
y = DD.irt_b.values
LOG = lambda Z: np.log1p(Z.clip(lower=0))
sets = {"patch_size": LOG(DD[PATCH]), "graph": LOG(DD[GRAPH]), "graph+patch": LOG(DD[GRAPH + PATCH]),
        "llm_judge": DD[list(judge.columns)], "llm_judge+graph+patch": pd.concat([DD[list(judge.columns)], LOG(DD[GRAPH + PATCH])], axis=1)}
N["difficulty"] = {"n": int(len(DD))}
for name, Z in sets.items():
    mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
    p = cross_val_predict(mdl, Z.values, y, cv=KFold(5, shuffle=True, random_state=0))
    pg = cross_val_predict(mdl, Z.values, y, cv=GroupKFold(5), groups=DD.repo.values)
    bs = [spearmanr(p[s], y[s])[0] for s in (rng.integers(0, len(y), len(y)) for _ in range(B))]
    N["difficulty"][name] = dict(rho=round(spearmanr(p, y)[0], 3), ci=[round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3)],
                                 rho_repo_out=round(spearmanr(pg, y)[0], 3))

# ------------------------------------------------------------------ R3b: triage gap (before vs after localization)
# PRE must not use the true fix site: hop_spec / hop_all are distances TO the gold location, so they belong in POST.
PRE = ["n_anchors_spec", "n_anchors_all", "n_issue_tokens", "traceback_present", "issue_chars", "repo_nodes"]
LOCF = ["loc_top1_score", "loc_margin", "loc_margin_rel", "loc_files_top10", "loc_file_entropy", "pred_hop",
        "pred_ambiguity", "pred_fanin", "pred_lines", "pred_named_in_issue"]
SITE = ["hop_spec", "hop_all", "gold_named_in_issue", "gold_file_in_issue"]          # where the true fix site is
EXTENT = ["n_gold_files", "n_gold_nodes", "gold_spread"]                               # how many files / functions it spans
POST = PRE + SITE + EXTENT + ["patch_lines"]
TRI = [("before_search", PRE), ("before+bm25_localizer", PRE + LOCF), ("oracle_site", PRE + SITE),
       ("oracle_site+extent", PRE + SITE + EXTENT), ("oracle_full", POST), ("patch_lines_only", ["patch_lines"])]
BT = min(B, 200)
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))


def oof(Z, yy, groups):
    return cross_val_predict(mk(), Z, yy, cv=GroupKFold(5), groups=groups, method="predict_proba")[:, 1]


N["triage"] = {}
PRED = {}
for a in WEAK:
    yy = M[a].values
    r = {}
    for name, cs in TRI:
        Z = np.log1p(F[cs].clip(lower=0)).values
        p = oof(Z, yy, F.repo.values)
        PRED[(a, name)] = p
        bs = []
        for _ in range(BT):                      # refit inside the bootstrap; duplicates share a repo so never straddle folds
            s_ = rng.integers(0, len(yy), len(yy))
            if yy[s_].min() != yy[s_].max():
                bs.append(roc_auc_score(yy[s_], oof(Z[s_], yy[s_], F.repo.values[s_])))
        top = np.argsort(-p)[: len(yy) // 2]
        r[name] = dict(auc=round(roc_auc_score(yy, p), 3), ci=[round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3)],
                       solves_in_top_half=round(yy[top].sum() / yy.sum(), 3))
    r["base_rate"] = round(yy.mean(), 3)
    N["triage"][A.loc[a, "model"]] = r
# budget at stake: share of agent calls / solves in the bottom half of predicted solvability
Ccalls = out.pivot(index="instance_id", columns="agent", values="api_calls").loc[F.index]
N["budget"] = {}
N["checkpoint_sim"] = {}
for a in WEAK:
    yy, cc = M[a].values, Ccalls[a].values.astype(float)
    tot = np.nansum(cc)
    r = dict(calls_on_failures=round(np.nansum(cc[yy == 0]) / tot, 3), calls_on_failures_if_equal_cost=round(1 - yy.mean(), 3),
             median_calls_solved=float(np.nanmedian(cc[yy == 1])), median_calls_failed=float(np.nanmedian(cc[yy == 0])))
    for name in ("before_search", "before+bm25_localizer", "oracle_site+extent", "oracle_full"):
        bot = np.argsort(PRED[(a, name)])[: len(yy) // 2]
        r[name] = dict(bottom_half_calls=round(np.nansum(cc[bot]) / tot, 3), bottom_half_solves=round(yy[bot].sum() / yy.sum(), 3))
    N["budget"][A.loc[a, "model"]] = r
    # pre-declared checkpoint rule: tasks predicted to be in the bottom half get capped at K = median calls of this model's solved
    # tasks.  A solved task is lost if it needed more than K calls.  Compare against capping a random half.
    K = float(np.nanmedian(cc[yy == 1]))
    sim = dict(K=K)
    for name in ("before_search", "before+bm25_localizer", "oracle_site+extent", "oracle_full"):
        bot = np.argsort(PRED[(a, name)])[: len(yy) // 2]
        saved = np.nansum(np.clip(cc[bot] - K, 0, None)) / tot
        lost = ((yy[bot] == 1) & (cc[bot] > K)).sum() / yy.sum()
        sim[name] = dict(calls_saved=round(saved, 3), solves_lost=round(lost, 3))
    rs, rl = [], []
    for _ in range(500):
        bot = rng.permutation(len(yy))[: len(yy) // 2]
        rs.append(np.nansum(np.clip(cc[bot] - K, 0, None)) / tot)
        rl.append(((yy[bot] == 1) & (cc[bot] > K)).sum() / yy.sum())
    sim["random_half"] = dict(calls_saved=round(float(np.mean(rs)), 3), solves_lost=round(float(np.mean(rl)), 3))
    N["checkpoint_sim"][A.loc[a, "model"]] = sim

N["localizer"] = dict(file_top1=round(F.eval_file_top1.mean(), 3), file_top5=round(F.eval_file_top5.mean(), 3),
                      func_top10=round(F.eval_func_top10.mean(), 3))
N["bm25_hit_solve"] = {}
for a in list(WEAK) + list(A.sort_values("rate").index[-3:]):
    hit, miss = M.loc[F.eval_file_top1 == 1, a], M.loc[F.eval_file_top1 == 0, a]
    N["bm25_hit_solve"][A.loc[a, "model"] + f" ({A.loc[a,'rate']:.0f}%)"] = dict(hit=round(hit.mean(), 3), miss=round(miss.mean(), 3),
                                                                             ratio=round(hit.mean() / max(miss.mean(), 1e-9), 2))

# ------------------------------------------------------------------ R4: Gemma 4 31B on the competition's public tasks
try:
    G = pd.read_csv("out/gemma_public_joined.csv").set_index("task_id")
    Gs = G[~G.needs_missing_pkg]
    def split(df, k, cut):
        lo, hi = df[df[k] <= cut]["mean"], df[df[k] > cut]["mean"]
        return dict(low=round(lo.mean(), 3), n_low=int(len(lo)), high=round(hi.mean(), 3), n_high=int(len(hi)),
                    p=round(mannwhitneyu(lo, hi).pvalue, 4))
    N["gemma"] = dict(n_tasks=int(len(G)), n_solvable=int(len(Gs)), n_flagged=int(G.needs_missing_pkg.sum()),
                      solve_all=round(G["mean"].mean(), 3), solve_solvable=round(Gs["mean"].mean(), 3),
                      solve_flagged=round(G[G.needs_missing_pkg]["mean"].mean(), 3),
                      multi_func=split(Gs, "n_gold_nodes", 1), multi_file=split(Gs, "n_gold_files", 1),
                      patch=split(Gs, "src_patch_lines", 10),
                      multi_func_all=split(G, "n_gold_nodes", 1))
    # within small patches only: does multi-function still hurt?
    sm_ = Gs[Gs.src_patch_lines <= 30]
    N["gemma"]["multi_func_within_patch_le30"] = split(sm_, "n_gold_nodes", 1)
    Gf = Gs[Gs.n_gold_nodes >= 1]                    # drop tasks whose fix touches no Python definition
    N["gemma"]["n_no_source_def"] = int((Gs.n_gold_nodes == 0).sum())
    N["gemma"]["multi_func_def_only"] = split(Gf, "n_gold_nodes", 1)
    N["gemma"]["patch_def_only"] = split(Gf, "src_patch_lines", 10)
    N["gemma"]["repos"] = {k: int(v) for k, v in G.repo.value_counts().items()}
    N["gemma"]["flagged_solved"] = list(G[G.needs_missing_pkg & (G["mean"] > 0)].index)
    N["gemma"]["flagged_multi_func"] = int((G[G.needs_missing_pkg].n_gold_nodes > 1).sum())
    N["gemma"]["agree_multifunc_patch10"] = int(((Gs.n_gold_nodes > 1) == (Gs.src_patch_lines > 10)).sum())
    O = pd.read_csv("data/gemma/pub/yosh_task_outcomes.csv")
    N["gemma"]["per_config"] = {}
    for arm in ["v8_noanz@ollama55", "v9_verify@ollama55"]:
        rr = O[O.arm == arm].set_index("task_id").resolved.astype(int)
        for lab, df in [("n41", Gs), ("def_only", Gf)]:
            r_ = rr.reindex(df.index)
            hi = df.n_gold_nodes > 1
            t = [[int(r_[~hi].sum()), int((~hi).sum() - r_[~hi].sum())], [int(r_[hi].sum()), int(hi.sum() - r_[hi].sum())]]
            N["gemma"]["per_config"][f"{arm}|{lab}"] = dict(single=t[0], multi=t[1], fisher_p=round(fisher_exact(t)[1], 3))
    rows = []
    for arm in ["v8_noanz@ollama55", "v9_verify@ollama55"]:
        rr = O[O.arm == arm].set_index("task_id").resolved.astype(int)
        for t in Gs.index:
            rows.append(dict(y=int(rr[t]), multi=int(Gs.loc[t, "n_gold_nodes"] > 1), lp=np.log1p(Gs.loc[t, "src_patch_lines"]), task=t))
    Dg = pd.DataFrame(rows)
    mg = sm.GLM(Dg.y, sm.add_constant(Dg[["multi", "lp"]]), family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(Dg.task)[0]})
    N["gemma"]["joint_logit"] = dict(multi=round(mg.params["multi"], 2), multi_p=round(mg.pvalues["multi"], 3),
                                     log_patch=round(mg.params["lp"], 2), log_patch_p=round(mg.pvalues["lp"], 3))
except FileNotFoundError:
    pass


# ---------------------------------------------------------------- own Gemma 4 31B run (129 public tasks, Kaggle 4xL4)
if os.path.exists("out/gemma_own_joined.csv"):
    J = pd.read_csv("out/gemma_own_joined.csv", index_col=0)
    Sv = J[~J.needs_missing_pkg]
    R = dict(n_tasks=len(J), n_flagged=int(J.needs_missing_pkg.sum()), flagged_solved=int(J[J.needs_missing_pkg].resolved.sum()),
             n_solvable=len(Sv), solved=int(Sv.resolved.sum()), rate=round(Sv.resolved.mean(), 3),
             rate_all=round(J.resolved.mean(), 3), hours=round(J.duration_s.sum() / 3600, 1),
             n_timeout=int((J.err == "timeout").sum()), n_ctx=int((J.err == "ctx").sum()),
             time_on_failures=round(Sv[Sv.resolved == 0].duration_s.sum() / Sv.duration_s.sum(), 3),
             med_dur_solved=float(Sv[Sv.resolved == 1].duration_s.median()), med_dur_failed=float(Sv[Sv.resolved == 0].duration_s.median()))

    def fsplit(df, col, thr):
        hi = df[col] > thr
        a, b = df[~hi].resolved, df[hi].resolved
        t = [[int(a.sum()), int(len(a) - a.sum())], [int(b.sum()), int(len(b) - b.sum())]]
        return dict(low=round(a.mean(), 3), high=round(b.mean(), 3), n_low=int(len(a)), n_high=int(len(b)), p=round(fisher_exact(t)[1], 4))
    R["multi_func"] = fsplit(Sv, "n_gold_nodes", 1)
    R["multi_file"] = fsplit(Sv, "n_gold_files", 1)
    R["patch"] = fsplit(Sv, "src_patch_lines", 10)
    Sd = Sv[Sv.n_gold_nodes >= 1]
    R["multi_func_def_only"] = fsplit(Sd, "n_gold_nodes", 1)
    R["timeout_rate"] = {k: round(v, 2) for k, v in Sv.groupby(Sv.n_gold_nodes > 1).err.apply(lambda e: (e == "timeout").mean()).rename({False: "single", True: "multi"}).items()}
    Xg = sm.add_constant(pd.DataFrame({"multi": (Sv.n_gold_nodes > 1).astype(int), "lp": np.log1p(Sv.src_patch_lines)}))
    mg = sm.Logit(Sv.resolved, Xg).fit(disp=0)
    R["joint_logit"] = dict(multi=round(mg.params["multi"], 2), multi_p=round(mg.pvalues["multi"], 3),
                            log_patch=round(mg.params["lp"], 2), log_patch_p=round(mg.pvalues["lp"], 3))
    # same checkpoint rule as Section 3.3, in wall-clock time: cap flagged tasks at the median duration of solved tasks
    K = float(Sv[Sv.resolved == 1].duration_s.median())
    tot = Sv.duration_s.sum()
    R["checkpoint"] = dict(K_s=K)
    for nm, hi in [("multi_func", Sv.n_gold_nodes > 1), ("patch_gt10", Sv.src_patch_lines > 10)]:
        sv = np.clip(Sv[hi].duration_s - K, 0, None).sum() / tot
        ls = ((Sv.resolved == 1) & hi & (Sv.duration_s > K)).sum() / Sv.resolved.sum()
        rs, rl = [], []
        for _ in range(2000):
            h = pd.Series(rng.permutation(hi.values), index=Sv.index)
            rs.append(np.clip(Sv[h].duration_s - K, 0, None).sum() / tot)
            rl.append(((Sv.resolved == 1) & h & (Sv.duration_s > K)).sum() / Sv.resolved.sum())
        R["checkpoint"][nm] = dict(share_capped=round(hi.mean(), 2), time_saved=round(sv, 3), solves_lost=round(ls, 3),
                                   random_time_saved=round(float(np.mean(rs)), 3), random_solves_lost=round(float(np.mean(rl)), 3))
    G55 = pd.read_csv("out/gemma_public_joined.csv", index_col=0).join(J.resolved, how="inner")
    R["overlap_yosh"] = dict(n=len(G55), yosh_rate=round(G55["mean"].mean(), 3), own_rate=round(G55.resolved.mean(), 3))
    CPF = "data/gemma/own/gemma31b_checkpoint_results_compact.csv"      # second full run: same setup + edit-plan checkpoint prompt
    if os.path.exists(CPF):
        from scipy.stats import binomtest
        Cp = pd.read_csv(CPF).set_index("instance_id")
        cpv = Cp.resolved.reindex(Sv.index)
        lost = int(((Sv.resolved == 1) & (cpv == 0)).sum())
        gained = int(((Sv.resolved == 0) & (cpv == 1)).sum())
        R["checkpoint_arm"] = dict(n_tasks=int(len(Cp)), solved=int(cpv.sum()), rate=round(cpv.mean(), 3),
                                   base_solved=int(Sv.resolved.sum()), gained=gained, lost=lost,
                                   mcnemar_p=round(binomtest(min(lost, gained), lost + gained, 0.5).pvalue, 3) if lost + gained else 1.0,
                                   timeouts=int((Cp.err == "timeout").sum()), base_timeouts=int((J.err == "timeout").sum()),
                                   ctx=int((Cp.err == "ctx").sum()),
                                   time_change=round(Cp.duration_s.sum() / J.duration_s.sum() - 1, 3),
                                   flagged_solved=int(Cp.resolved.reindex(J[J.needs_missing_pkg].index).sum()))

    # ------------------------------------------------------------ agent traces of both runs (258 ATIF traces)
    TF = "data/gemma/own/trace_features.csv"
    if os.path.exists(TF) and os.path.exists(CPF):
        from sklearn.metrics import roc_auc_score as _auc
        Tr = pd.read_csv(TF)
        Tr["res"] = [(J.resolved if a == 0 else Cp.resolved)[t] for t, a in zip(Tr.task, Tr.arm)]
        for c_ in ["needs_missing_pkg", "n_gold_files", "n_gold_nodes", "src_patch_lines", "repo"]:
            Tr[c_] = Tr.task.map(J[c_])
        St = Tr[~Tr.needs_missing_pkg & (Tr.n_gold_files > 0)].copy()      # solvable tasks with a known source fix file
        St["te"] = np.where(St.t_edit >= 0, St.t_edit, 999)                # time of first source edit (999 = never)
        St["tg"] = np.where(St.t_gold >= 0, St.t_gold, 999)                # time the agent first opened a fixed file
        W = dict(n_tasks=int(St.task.nunique()), n_traces=int(len(Tr)))
        loc, ged = St.t_gold >= 0, St.n_gold_edited > 0
        W["found_fix_file"] = round(loc.mean(), 3)
        W["t_found_median"] = float(St[loc].t_gold.median())
        W["found_within_30s"] = round(St.t_gold.between(0, 30).mean(), 3)
        W["call_found_median"] = float(St[loc].k_gold.median())
        W["edited_fix_file"] = round(ged.mean(), 3)
        W["t_first_edit_median"] = float(St[St.te < 999].te.median())
        W["found_to_edit_median"] = float((St[(St.te < 999) & loc].te - St[(St.te < 999) & loc].tg).median())
        W["time_before_first_edit"] = round(np.where(St.te < 999, St.te, St.dur).sum() / St.dur.sum(), 3)
        W["ctx_near_limit"] = round((St.max_prompt_tok >= 14000).mean(), 3)
        W["graph_calls"] = int(Tr.n_graph.sum())
        W["all_calls"] = int(Tr.n_calls.sum())
        W["graph_tasks"] = int((St.n_graph > 0).sum())
        W["graph_tasks_solved"] = int(St[St.n_graph > 0].res.sum())
        W["funnel"] = {}
        for a in (0, 1):
            x = St[St.arm == a]
            l_, g_ = x.t_gold >= 0, x.n_gold_edited > 0
            W["funnel"][str(a)] = dict(n=int(len(x)), found=int(l_.sum()), edited=int(g_.sum()), solved=int(x.res.sum()),
                                       never_found=int((~l_ & (x.res == 0)).sum()), found_not_edited=int((l_ & ~g_ & (x.res == 0)).sum()),
                                       edited_failed=int((g_ & (x.res == 0)).sum()))
        W["funnel_total"] = {k: int(sum(W["funnel"][r][k] for r in ("0", "1"))) for k in W["funnel"]["0"]}
        # which signals separate solved from failed tasks?  task-clustered bootstrap over both runs
        gg = {t: g for t, g in St.groupby("task")}
        tk = St.task.unique()
        sig = dict(first_edit=lambda d: -d.te, found_file=lambda d: -d.tg, oracle_patch=lambda d: -d.src_patch_lines,
                   oracle_funcs=lambda d: -d.n_gold_nodes)
        bsv = {k: [] for k in sig}
        dif = []
        for _ in range(2000):
            bs_ = pd.concat([gg[t] for t in rng.choice(tk, len(tk))])
            for k, f in sig.items():
                bsv[k].append(_auc(bs_.res, f(bs_)))
            dif.append(bsv["first_edit"][-1] - bsv["oracle_patch"][-1])
        W["auc"] = {k: dict(auc=round(_auc(St.res, f(St)), 3), ci=[round(np.percentile(bsv[k], 2.5), 3), round(np.percentile(bsv[k], 97.5), 3)])
                    for k, f in sig.items()}
        W["auc_first_edit_minus_oracle_ci"] = [round(np.percentile(dif, 2.5), 3), round(np.percentile(dif, 97.5), 3)]
        Aa = St[St.arm == 0].set_index("task")
        Bb = St[St.arm == 1].set_index("task").reindex(Aa.index)
        W["cross_run"] = dict(auc_a_to_b=round(_auc(Bb.res, -Aa.te), 3), auc_b_to_a=round(_auc(Aa.res, -Bb.te), 3),
                              spearman_t_edit=round(spearmanr(Aa.te, Bb.te)[0], 2), outcome_agreement=round((Aa.res == Bb.res).mean(), 3),
                              both=int(((Aa.res == 1) & (Bb.res == 1)).sum()), a_only=int(((Aa.res == 1) & (Bb.res == 0)).sum()),
                              b_only=int(((Aa.res == 0) & (Bb.res == 1)).sum()))
        H = 180                                                             # half of the 6-minute task budget
        e_ = St.te <= H
        W["half_time"] = dict(threshold_s=H, early_n=int(e_.sum()), early_rate=round(St[e_].res.mean(), 3), late_n=int((~e_).sum()),
                              late_rate=round(St[~e_].res.mean(), 3),
                              cross_early_rate=round(pd.concat([Bb[Aa.te <= H].res, Aa[Bb.te <= H].res]).mean(), 3),
                              cross_late_rate=round(pd.concat([Bb[Aa.te > H].res, Aa[Bb.te > H].res]).mean(), 3))
        nev = St.te > 300
        W["no_edit_by_300"] = dict(n=int(nev.sum()), solved=int(St[nev].res.sum()), time_share=round(St[nev].dur.sum() / St.dur.sum(), 3))
        # stop rule: at H seconds, stop a task that has made no source edit.  A solve is lost if the task was stopped and finished later than H.
        W["stop_rule"] = {}
        for a in (0, 1):
            x = St[St.arm == a]
            tot_ = x.dur.sum()

            def sim(mask):
                return ((x[mask].dur - H).clip(lower=0).sum() / tot_, ((x.res == 1) & mask & (x.dur > H)).sum() / x.res.sum())
            late = x.te > H
            n_ = int(late.sum())
            sv, ls = sim(late)
            om = pd.Series(False, index=x.index)
            om[x.sort_values("src_patch_lines", ascending=False).index[:n_]] = True
            osv, ols = sim(om)
            rr = [sim(pd.Series(rng.permutation(late.values), index=x.index)) for _ in range(2000)]
            W["stop_rule"][str(a)] = dict(stopped=n_, time_saved=round(sv, 3), solves_lost=round(ls, 3),
                                          oracle_time_saved=round(osv, 3), oracle_solves_lost=round(ols, 3),
                                          random_time_saved=round(float(np.mean([r[0] for r in rr])), 3),
                                          random_solves_lost=round(float(np.mean([r[1] for r in rr])), 3))
        # decision-time (landmark) view: at second t, among task-runs that are still running, compare those that
        # have made a source edit with those that have not.  Only information the agent has at t is used.
        W["landmark"] = {}
        rem_all = St.dur.sum()
        for t in (60, 120, 180, 240, 300):
            Rn = St[St.dur > t]
            ed = Rn.te <= t
            W["landmark"][str(t)] = dict(running=int(len(Rn)), edited=int(ed.sum()), edited_solved=int(Rn[ed].res.sum()),
                                         edited_rate=round(Rn[ed].res.mean(), 3), not_edited=int((~ed).sum()),
                                         not_edited_solved=int(Rn[~ed].res.sum()), not_edited_rate=round(Rn[~ed].res.mean(), 3),
                                         not_edited_rest_share=round((Rn[~ed].dur - t).sum() / rem_all, 3))
        # does the edit clock add anything once the true patch size is known?  landmark at 180 s, task-clustered bootstrap
        L_ = St[St.dur > H].copy()
        L_["ed"] = (L_.te <= H).astype(float)
        L_["lp"] = np.log1p(L_.src_patch_lines)
        import statsmodels.api as _sm
        _m = _sm.Logit(L_.res, _sm.add_constant(L_[["ed", "lp"]])).fit(disp=0)
        gl = {t: g for t, g in L_.groupby("task")}
        tl = L_.task.unique()
        cb = []
        for _ in range(2000):
            b_ = pd.concat([gl[t] for t in rng.choice(tl, len(tl))])
            try:
                cb.append(_sm.Logit(b_.res, _sm.add_constant(b_[["ed", "lp"]])).fit(disp=0).params["ed"])
            except Exception:
                pass
        W["joint_180"] = dict(n=int(len(L_)), ed=round(_m.params["ed"], 2), ed_ci=[round(np.percentile(cb, 2.5), 2), round(np.percentile(cb, 97.5), 2)],
                              ed_p=float(_m.pvalues["ed"]), lp=round(_m.params["lp"], 2), lp_p=float(_m.pvalues["lp"]))
        W["by_repo"] = {r: dict(n=int(len(g)), solved=int(g.res.sum()), auc_first_edit=round(_auc(g.res, -g.te), 3),
                                auc_oracle_patch=round(_auc(g.res, -g.src_patch_lines), 3))
                        for r, g in St.groupby("repo") if g.res.nunique() > 1 and len(g) >= 20}
        # net effect of a stop rule when every stopped task (it has no patch, so the agent knows it failed) gets one retry,
        # with the retry outcome taken from the other run of the same task
        W["stop_retry"] = {}
        for T in (180, 240, 300):
            out = []
            for a, b in ((0, 1), (1, 0)):
                A_ = St[St.arm == a].set_index("task")
                B_ = St[St.arm == b].set_index("task").reindex(A_.index)
                stp = (A_.te > T) & (A_.dur > T)
                Bs = B_[stp]
                Bstp = Bs.te > T
                out.append(dict(base=int(A_.res.sum()), stopped=int(stp.sum()), lost=int((stp & (A_.res == 1)).sum()),
                                saved_h=round((A_[stp].dur - T).sum() / 3600, 2),
                                retry_h=round(float(np.where(Bstp, np.minimum(Bs.dur, T), Bs.dur).sum()) / 3600, 2),
                                retry_gain=int(Bs[~Bstp].res.sum()), total_h=round(A_.dur.sum() / 3600, 2)))
            W["stop_retry"][str(T)] = out
        # the agent's own stated plan (checkpoint run only)
        x = St[St.arm == 1]
        pp = x[x.plan_files >= 0]
        W["plan"] = dict(n_run=int(len(x)), n_stated=int(len(pp)), rate_stated=round(pp.res.mean(), 3),
                         n_not_stated=int((x.plan_files < 0).sum()), solved_not_stated=int(x[x.plan_files < 0].res.sum()),
                         not_stated_found_file=round((x[x.plan_files < 0].t_gold >= 0).mean(), 3),
                         files_match=round((pp.plan_files == pp.n_gold_files).mean(), 3),
                         one_func_agree=round(((pp.plan_funcs <= 1) == (pp.n_gold_nodes <= 1)).mean(), 3),
                         lines_spearman=round(spearmanr(pp.plan_lines.clip(0), pp.src_patch_lines)[0], 2),
                         auc_plan_lines=round(_auc(pp.res, -pp.plan_lines.where(pp.plan_lines >= 0, 10)), 3),
                         auc_oracle_patch_same_tasks=round(_auc(pp.res, -pp.src_patch_lines), 3),
                         big_true=int((pp.src_patch_lines > 10).sum()), big_true_planned_small=int(((pp.src_patch_lines > 10) & (pp.plan_lines <= 10)).sum()))
        R["traces"] = W
    N["gemma_own"] = R

json.dump(N, open("out/numbers.json", "w"), indent=1, default=float)
print(json.dumps(N, indent=1, default=float))

# third run (10 minutes per task) adds its own block
import os, subprocess, sys  # noqa: E402
if os.path.exists("data/gemma/own/gemma31b_tenmin_results_compact.csv"):
    subprocess.run([sys.executable, "src/tenmin_numbers.py"], check=True)
