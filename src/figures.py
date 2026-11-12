"""Paper figures. Reads out/numbers.json, out/per_agent_coefs.csv. Writes paper/fig*.png."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

N = json.load(open("out/numbers.json"))
P = pd.read_csv("out/per_agent_coefs.csv")
WEAK, STRONG, GEMMA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False})


def errbar(ax, x, d, color, label=None):
    lo, hi = d["ci"]
    ax.errorbar([x], [d["rate"]], yerr=[[d["rate"] - lo], [hi - d["rate"]]], fmt="o", ms=6, color=color,
                ecolor=color, elinewidth=1.4, capsize=0, label=label)


# ---------------- Fig 1: solve rate by where the fix lives (weak vs strong cohorts)
C = N["cohort"]
groups = [("Fix site vs. issue\n(call-graph hops)", [("named\n(0)", "hop0"), ("1 hop", "hop1"), ("2+ hops", "hop2plus"), ("no\nanchor", "no_anchor")]),
          ("Files the fix\ntouches", [("one", "single_file"), ("several", "multi_file")]),
          ("Issue names\nthe fix file?", [("yes", "file_named"), ("no", "file_not_named")])]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True, gridspec_kw={"width_ratios": [4, 2, 2]})
for ax, (title, items) in zip(axes, groups):
    for i, (lab, key) in enumerate(items):
        errbar(ax, i - 0.12, C[key]["weak"], WEAK, "weak agents (5 models, 9–44%)" if (ax is axes[0] and i == 0) else None)
        errbar(ax, i + 0.12, C[key]["strong"], STRONG, "strong agents (13 models, ≥70%)" if (ax is axes[0] and i == 0) else None)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([f"{lab}\nn={C[k]['weak']['n_tasks']}" for lab, k in items], fontsize=7.5)
    ax.set_xlim(-0.5, len(items) - 0.5)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("tasks resolved")
axes[0].set_ylim(0, 0.9)
axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
fig.legend(loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("paper/fig1_where_fix_lives.png", dpi=220)
plt.close(fig)

# ---------------- Fig 2: per-agent coefficients vs agent strength (38 models, same scaffold)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
for ax, col, title in [(axes[0], "hop", "Call-graph hops, issue → fix"), (axes[1], "files(log)", "Files the fix touches")]:
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.scatter(P.rate, P[col], s=26, color=WEAK, edgecolor="white", linewidth=1.2, zorder=3)
    s = N["per_agent"][col]
    ax.set_title(f"{title}\nnegative for {s['n_negative']} of {len(P)} models ({s['n_negative_p05']} with p < .05)", fontsize=8.5, color=INK, loc="left")
    ax.set_xlabel("agent's SWE-bench Verified resolve rate (%)")
for _, r in P.iterrows():
    if r["rate"] < 15 or r["rate"] > 76.5:
        axes[1].annotate(r["model"].replace(" Instruct", ""), (r["rate"], r["files(log)"]), fontsize=6.5, color=MUTED,
                         xytext=(4, 3), textcoords="offset points")
axes[0].set_ylabel("effect on log-odds of success\n(per SD, controls: patch & issue size)")
fig.tight_layout()
fig.savefig("paper/fig2_per_agent.png", dpi=220)
plt.close(fig)

# ---------------- Fig 3: the triage gap (left) and a simulated edit-plan checkpoint (right)
T = N["triage"]
S = N["checkpoint_sim"]
order = sorted(T, key=lambda m: T[m]["base_rate"])
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={"width_ratios": [1.45, 1]})
stages = [("before_search", "issue only (before search)", "#a8a7a2"),
          ("before+bm25_localizer", "+ BM25 localizer hit", WEAK),
          ("oracle_site", "+ where the true fix is", "#8f6bd1"),
          ("oracle_site+extent", "+ how many files / functions it spans", STRONG),
          ("oracle_full", "+ true patch size", "#7a3410")]
off = [-0.32, -0.16, 0, 0.16, 0.32]
for j, (k, lab, col) in enumerate(stages):
    for i, m in enumerate(order):
        d = T[m][k]
        ax.errorbar([d["auc"]], [i + off[j]], xerr=[[d["auc"] - d["ci"][0]], [d["ci"][1] - d["auc"]]], fmt="o",
                    ms=4.2, color=col, elinewidth=1.1, label=lab if i == 0 else None)
ax.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (2, 2)))
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{m.replace(' Instruct','').replace('Qwen2.5-Coder','Qwen2.5-C.')} ({T[m]['base_rate']:.0%})" for m in order], fontsize=7.5)
ax.set_xlabel("AUC, which tasks the agent resolves\n(repository-grouped 5-fold CV, 95% CI)", fontsize=8)
ax.set_xlim(0.3, 0.9)
ax.grid(axis="y", visible=False)
ax.set_title("a  What predicts a weak model's solves", fontsize=8.5, color=INK, loc="left")
# right: share of solves lost when the predicted bottom half is capped at the model's median solve cost
for k, lab, col, mk_, dy in [("random_half", "random half", "#a8a7a2", "s", -0.16), ("before+bm25_localizer", "BM25-level ranking", WEAK, "o", 0),
                             ("oracle_site+extent", "planned-extent ranking (oracle)", STRONG, "o", 0.16)]:
    for i, m in enumerate(order):
        bx.scatter([S[m][k]["solves_lost"]], [i + dy], s=28, color=col, marker=mk_, edgecolor="white", linewidth=0.8,
                   label=lab if i == 0 else None, zorder=3)
for i, m in enumerate(order):
    bx.text(0.345, i, f"saves {S[m]['oracle_site+extent']['calls_saved']:.0%}", fontsize=6.5, color=MUTED, va="center")
bx.set_yticks(range(len(order)))
bx.set_yticklabels([])
bx.set_ylim(ax.get_ylim())
bx.set_xlabel("share of the model's solves lost", fontsize=8)
bx.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
bx.set_xlim(0.05, 0.44)
bx.grid(axis="y", visible=False)
bx.set_title("b  Cap the predicted-hard half", fontsize=8.5, color=INK, loc="left")

h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="lower left", ncol=2, fontsize=7, bbox_to_anchor=(0.0, 0.0), title="a: information level", title_fontsize=7)
h, l = bx.get_legend_handles_labels()
fig.legend(h, l, loc="lower right", ncol=1, fontsize=7, bbox_to_anchor=(1.0, 0.0), title="b: which half is capped", title_fontsize=7)
fig.tight_layout(rect=(0, 0.2, 1, 1))
fig.savefig("paper/fig3_triage_gap.png", dpi=220)
plt.close(fig)

# ---------------- Fig 4: Gemma 4 31B on the competition's public tasks
G = N["gemma_own"]
fp = lambda p: "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True, gridspec_kw={"width_ratios": [1.5, 1]})
xs = [0, 1]
ax.plot(xs, [C["single_func"]["strong"]["rate"], C["multi_func"]["strong"]["rate"]], "-o", color=STRONG, lw=2, ms=6)
ax.plot(xs, [C["single_func"]["weak"]["rate"], C["multi_func"]["weak"]["rate"]], "-o", color=WEAK, lw=2, ms=6)
ax.plot(xs, [G["multi_func"]["low"], G["multi_func"]["high"]], "-o", color=GEMMA, lw=2, ms=6)
lab = [("strong, SWE-bench (13 models)", C["multi_func"]["strong"]["rate"], STRONG),
       ("weak, SWE-bench (5 models)", C["multi_func"]["weak"]["rate"], WEAK),
       (f"Gemma 4 31B, my run\nn={G['multi_func']['n_low']} vs {G['multi_func']['n_high']} tasks", G["multi_func"]["high"], "#117a55")]
for (t, yv, col), dy in zip(lab, [0, -20, 14]):
    ax.annotate(t, (1, yv), xytext=(8, dy), textcoords="offset points", va="center", fontsize=7, color=col)
ax.set_xticks(xs)
ax.set_xticklabels(["one function", "several functions"], fontsize=7.5)
ax.set_xlim(-0.3, 2.4)
ax.set_ylim(0, 0.9)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_ylabel("tasks resolved")
ax.grid(axis="x", visible=False)
ax.set_title(f"a  By functions the fix touches ({fp(G['multi_func']['p'])})", fontsize=8.5, color=INK, loc="left")
bx.bar([0, 1], [G["patch"]["low"], G["patch"]["high"]], color=GEMMA, width=0.55)
for i, (v, n) in enumerate([(G["patch"]["low"], G["patch"]["n_low"]), (G["patch"]["high"], G["patch"]["n_high"])]):
    bx.text(i, v + 0.02, f"{v:.0%}\nn={n}", ha="center", fontsize=7.5, color="#117a55")
bx.set_xticks([0, 1])
bx.set_xticklabels(["≤ 10 lines", "> 10 lines"], fontsize=7.5)
bx.set_xlim(-0.6, 1.6)
bx.grid(axis="x", visible=False)
bx.set_title(f"b  By patch size ({fp(G['patch']['p'])})", fontsize=8.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig("paper/fig4_gemma.png", dpi=220)
plt.close(fig)

# ---------------- Fig 5: inside the Gemma 4 31B runs (258 traces)
import numpy as np  # noqa: E402
if "traces" in N.get("gemma_own", {}):
    TW = N["gemma_own"]["traces"]
    Tr = pd.read_csv("data/gemma/own/trace_features.csv")
    Jg = pd.read_csv("out/gemma_own_joined.csv", index_col=0)
    Cg = pd.read_csv("data/gemma/own/gemma31b_checkpoint_results_compact.csv").set_index("instance_id")
    Tr["res"] = [(Jg.resolved if a == 0 else Cg.resolved)[t] for t, a in zip(Tr.task, Tr.arm)]
    Tr = Tr[~Tr.task.map(Jg.needs_missing_pkg) & (Tr.task.map(Jg.n_gold_files) > 0)]
    te = np.where(Tr.t_edit >= 0, Tr.t_edit, np.nan)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.35, 1]})
    te = np.where(Tr.t_edit >= 0, Tr.t_edit, 999)
    ts_ = np.arange(30, 331, 15)

    def wil(k, n, z=1.96):
        if n == 0:
            return np.nan, np.nan, np.nan
        p_ = k / n
        d_ = 1 + z * z / n
        c_ = (p_ + z * z / (2 * n)) / d_
        h_ = z * np.sqrt(p_ * (1 - p_) / n + z * z / (4 * n * n)) / d_
        return p_, c_ - h_, c_ + h_
    for flag, col, lab in [(True, GEMMA, "has edited source by then"), (False, "#8f8e88", "has not edited yet")]:
        pts = []
        for t in ts_:
            run_ = Tr.dur.values > t
            grp = run_ & ((te <= t) == flag)
            pts.append(wil(int(Tr.res.values[grp].sum()), int(grp.sum())))
        pts = np.array(pts)
        ax.plot(ts_, pts[:, 0], "-o", color=col, ms=3, lw=1.6, label=lab)
        ax.fill_between(ts_, pts[:, 1], pts[:, 2], color=col, alpha=0.15, lw=0)
    L3 = TW["landmark"]["300"]
    ax.annotate(f"{L3['not_edited_solved']} of {L3['not_edited']}", (300, 0), xytext=(-4, 14), textcoords="offset points",
                fontsize=7, color="#5f5e59", ha="center", arrowprops=dict(arrowstyle="-", color="#8f8e88", lw=0.6))
    ax.axvline(180, color=INK, lw=0.9, ls=(0, (3, 2)))
    ax.text(184, 0.84, "half-time", fontsize=7, color=INK)
    ax.set_xlim(20, 340)
    ax.set_ylim(0, 0.9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("seconds into the task (task-runs still running)", fontsize=8)
    ax.set_ylabel("eventually solved", fontsize=8)
    ax.legend(fontsize=7, loc="lower left", frameon=True, framealpha=0.9)
    ax.grid(axis="x", visible=False)
    ax.set_title("a  What the agent's clock says, second by second", fontsize=8.5, color=INK, loc="left")
    A_ = TW["auc"]
    rows = [("first source edit (agent)", A_["first_edit"], GEMMA), ("true patch size (oracle)", A_["oracle_patch"], STRONG),
            ("true functions touched (oracle)", A_["oracle_funcs"], STRONG), ("opened a fixed file (agent)", A_["found_file"], "#a8a7a2")]
    for i, (lab, d, col) in enumerate(rows[::-1]):
        bx.errorbar([d["auc"]], [i], xerr=[[d["auc"] - d["ci"][0]], [d["ci"][1] - d["auc"]]], fmt="o", ms=5, color=col, elinewidth=1.3)
        bx.text(d["ci"][1] + 0.015, i, f"{d['auc']:.2f}", va="center", fontsize=7.5, color=INK)
    bx.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    bx.set_yticks(range(len(rows)))
    bx.set_yticklabels([r[0] for r in rows[::-1]], fontsize=7.5)
    bx.set_xlim(0.3, 1.0)
    bx.set_xlabel("AUC, solved vs failed", fontsize=8)
    bx.grid(axis="y", visible=False)
    bx.set_title("b  What separates the outcomes", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig("paper/fig5_traces.png", dpi=220)
    plt.close(fig)

print("ok")
