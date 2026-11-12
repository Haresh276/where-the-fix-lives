"""Third Gemma 4 31B run: same agent, same 102 solvable tasks, 10 minutes per task instead of 6.

Adds N["gemma_own"]["tenmin"] to out/numbers.json.  Run after paper_numbers.py (which calls it at the end).
Inputs: data/gemma/own/gemma31b_tenmin_results_compact.csv (per-task outcomes)
        data/gemma/own/trace_features_tenmin.csv           (per-task first source edit, from the run's traces)
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest

N = json.load(open("out/numbers.json"))
P = pd.read_csv("data/gemma/own/gemma31b_tenmin_results_compact.csv").set_index("instance_id")
P = P[P.err.fillna("") != "deadline"]                       # tasks the session never started
A = pd.read_csv("data/gemma/own/gemma31b_task_results_compact.csv").set_index("instance_id").reindex(P.index)
B = pd.read_csv("data/gemma/own/gemma31b_checkpoint_results_compact.csv").set_index("instance_id").reindex(P.index)


def mcnemar(x, y):
    g, l_ = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    return g, l_, (binomtest(g, g + l_, 0.5).pvalue if g + l_ else 1.0)


g1, l1, p1 = mcnemar(P.resolved, A.resolved)
g2, l2, p2 = mcnemar(P.resolved, B.resolved)
T = dict(n=int(len(P)), solved=int(P.resolved.sum()), rate=round(P.resolved.mean(), 3),
         run1_solved=int(A.resolved.sum()), run2_solved=int(B.resolved.sum()),
         vs_run1=dict(gained=g1, lost=l1, p=round(p1, 3)), vs_run2=dict(gained=g2, lost=l2, p=round(p2, 3)),
         solved_after_360=int(((P.resolved == 1) & (P.duration_s > 360)).sum()),
         hours=round(P.duration_s.sum() / 3600, 2), run1_hours=round(A.duration_s.sum() / 3600, 2),
         time_ratio=round(P.duration_s.sum() / A.duration_s.sum(), 2),
         extra_time=round(P.duration_s.sum() / A.duration_s.sum() - 1, 3),
         at_wall=int((P.duration_s >= 590).sum()), run1_at_wall=int((A.duration_s >= 355).sum()),
         call_cap=int((P.tool_calls >= 40).sum()), run1_call_cap=int((A.tool_calls >= 40).sum()),
         wall_or_cap=int(((P.duration_s >= 590) | (P.tool_calls >= 40)).sum()),
         run1_wall_or_cap=int(((A.duration_s >= 355) | (A.tool_calls >= 40)).sum()))
try:
    F = pd.read_csv("data/gemma/own/trace_features_tenmin.csv").set_index("task").reindex(P.index)
    F["res"] = P.resolved
    F["te"] = np.where(F.t_edit >= 0, F.t_edit, 9999)
    lm = {}
    for t in (180, 300, 360):
        Rn = F[F.dur > t]
        ed = Rn.te <= t
        lm[str(t)] = dict(running=int(len(Rn)), not_edited=int((~ed).sum()), not_edited_solved=int(Rn[~ed].res.sum()),
                          edited=int(ed.sum()), edited_solved=int(Rn[ed].res.sum()))
    T["landmark"] = lm
    # stop rule inside this run: stop at S seconds if there is still no source edit
    T["stop"] = {}
    for S in (180, 240, 300, 360, 400):
        stp = (F.te > S) & (F.dur > S)
        T["stop"][str(S)] = dict(stopped=int(stp.sum()), lost=int((stp & (F.res == 1)).sum()),
                                 saved_h=round((F[stp].dur - S).sum() / 3600, 2),
                                 hours_after=round((F.dur.sum() - (F[stp].dur - S).sum()) / 3600, 2),
                                 saved_share=round((F[stp].dur - S).sum() / F.dur.sum(), 3),
                                 solved_after=int(F.res.sum() - (stp & (F.res == 1)).sum()))
    T["n_traces"] = int(F.dur.notna().sum())
    from sklearn.metrics import roc_auc_score
    T["auc_first_edit"] = round(roc_auc_score(F.res, -F.te), 3)
    T["after_360_share"] = round((F.dur - 360).clip(lower=0).sum() / F.dur.sum(), 3)
    late = F[(F.res == 1) & (F.te > 300) & (F.te < 9999)]
    T["late_edit_solves"] = dict(n=int(len(late)), edit_times=[int(x) for x in sorted(late.te)])
except FileNotFoundError:
    pass
N["gemma_own"]["tenmin"] = T
json.dump(N, open("out/numbers.json", "w"), indent=1, default=float)
print(json.dumps(T, indent=1))
