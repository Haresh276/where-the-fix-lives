# Where the Fix Lives: Code-Graph Structure and the Triage Gap for Local Coding Agents

**Haresh G.** (Kaggle: Haresh276) · Code and data: github.com/Haresh276/where-the-fix-lives

## Abstract

A local agent such as Gemma 4 31B fixes few bugs, and under a fixed 12-hour budget its failures also take the most time. I ask which properties of a bug's fix location predict failure, and at what point an agent could know them. I build static code graphs of 12 Python repositories at the base commit of all 500 SWE-bench Verified tasks, on CPU, and join them with per-task outcomes for 38 models run in one scaffold family (mini-SWE-agent; refits within each release agree). Two properties predict failure beyond patch size and issue length: an issue that names nothing connected to the fix in the graph ({{ci:hop_recoded.no_anchor_per_task}} log-odds) and a fix that touches more files ({{ci:hop_recoded.files(log)}} per SD). Multi-file fixes drop weak models from {{pct:cohort.single_file.weak.rate}} to {{pct1:cohort.multi_file.weak.rate}} resolved. None of this can be read from the issue: predicting a weak model's solves before search is at chance (AUC {{rng:triage|*|before_search.auc}}), and knowing where the true fix sits adds little over a cheap BM25 hit. What separates wins from losses is the *extent* of the edit, meaning how many files and functions it spans (AUC {{rng:triage|*|oracle_site+extent.auc}}). On public run logs, capping the predicted-hard half of tasks at a model's median solve cost loses {{rng:checkpoint_sim|*|oracle_site+extent.solves_lost|pct}} of its solves when ranked by that extent, against {{rng:checkpoint_sim|*|random_half.solves_lost|pct}} when capped at random. In my own run of Gemma 4 31B on all {{v:gemma_own.n_tasks}} public competition tasks on the competition's 4×L4 setup, it solves {{pct:gemma_own.multi_func.low}} of solvable one-function fixes and {{pct:gemma_own.multi_func.high}} of multi-function fixes, and {{pct:gemma_own.patch.low}} vs {{pct:gemma_own.patch.high}} split at a 10-line patch (p = {{p:gemma_own.patch.p}}). The usable rule is an edit-plan checkpoint, not a pre-screen.

## 1. Introduction

The Gemma 4 Developer Agent competition gives an agent 12 hours on four L4 GPUs to fix a hidden set of bugs. In my run of Gemma 4 31B on the competition's public tasks, failing tasks took twice as long as solved ones (median {{int:gemma_own.med_dur_failed}} s vs {{int:gemma_own.med_dur_solved}} s) and used {{pct:gemma_own.time_on_failures}} of total runtime. If an agent could tell early which bugs it will not fix, it could move that time to bugs it can.

Patch size is the obvious predictor, and it matters. But two three-line patches can differ completely. In one, the issue names the broken function. In the other, the broken line is three calls away from anything the issue mentions, in a file the issue never names. A repository's code graph makes that difference measurable, and the competition gives every agent graph tools.

I test three questions:

1. Does **where the fix sits in the code graph** predict failure beyond patch size and issue length, across models?
2. Can an agent read that signal **before** it searches, or only once it has found and planned the fix? I call the difference the *triage gap*.
3. Does the pattern hold for **Gemma 4 31B on the competition's own tasks**?

Prior work predicts task difficulty from LLM-judged task features [2], and shows on synthetic call-graph tasks that code understanding, not editing, limits coding agents [3]. Localization-first pipelines [7] and graph-guided localizers [8] treat finding the fix site as its own stage. I ask what the fix site, once found, says about whether the agent will succeed. I measure real repository structure instead of asking a model to judge it, and I compare against [2]'s released features.

## 2. Data and method

**Tasks and outcomes.** SWE-bench Verified [4] has 500 human-validated bug-fix tasks from 12 Python repositories; Django is 46% of them. The public leaderboard publishes per-task results for 40 runs of the bash-only *mini-SWE-agent* scaffold [5]. I drop two runs whose per-task files disagree with their reported score by more than 5 points (Claude 3.7 Sonnet, Gemini 3 Pro), leaving **38 models** (15 open-weight) resolving 9%–77%. The runs span 11 releases of the scaffold, so model and scaffold version are not fully separated. Each run gets its own fixed effect, and Section 3.1 refits within single releases. I call the five models below 45% *weak* (Qwen2.5-Coder 32B, Llama 4 Maverick, gpt-oss-120b, GPT-5 nano, Kimi K2) and the 13 at or above 70% *strong*.

**Code graphs at every commit.** For each task I parse every non-test Python file at its base commit, read from a bare git clone and cached by file hash, into definitions (functions, methods, classes). Definitions are linked by call edges and by containment edges (class to method). A call resolves by name to every definition with that name, which over-approximates, as most static Python call graphs do. Graphs have a median of {{int:graph.nodes_median}} nodes (maximum {{int:graph.nodes_max}}).

**Where the fix lives.** I map each task's curated fix spans to graph nodes (97.5% resolve) and compute, among others:

- *hops*: shortest graph distance from any definition the issue names (identifier tokens, traceback frames; names with more than five definitions are ignored) to any fixed definition. 0 means the issue names the fixed function. *No anchor* ({{v:hop_recoded.n_no_anchor}} tasks) means no usable name in the issue reaches the fix within 8 hops; for {{v:hop_recoded.n_no_anchor_true}} of these the issue names nothing in the graph at all;
- *files* and *functions* touched by the fix;
- whether the issue names the fix *file*;
- controls: patch lines and issue length.

*Example.* In `sphinx-doc__sphinx-9591`, the issue reports that a property's type annotation is not cross-referenced. The fix changes three lines in one method, `PyProperty.handle_signature`, three hops from the nearest definition the issue names. All 13 strong models fix it and none of the five weak models do. Patch size alone would call it easy.

**Analyses.** (a) A pooled logistic model of 19,000 task × model outcomes with a fixed effect per run and standardized features; 95% CIs from {{int:boot_reps}} task-level bootstrap resamples, plus leave-one-repository-out refits. (b) The same features fitted per model. (c) Ridge regression of item-response difficulty from [2], against [2]'s LLM-judged features. (d) Classifiers of each weak model's successes at increasing information levels, cross-validated with whole repositories held out (Section 3.3).

**Competition tasks.** The competition's public tasks are merged pull requests in fastapi, requests, rich and httpx, named `<repo>_<PR number>`. I ran Gemma 4 31B (QAT, the competition's default agent and tools, vLLM on 4×L4) on all {{v:gemma_own.n_tasks}} of them in one Kaggle session: 6 minutes, 40 tool calls and 70 turns per task, {{v:gemma_own.hours}} GPU-hours in total. I rebuilt each task from public git history (fix commit and its parent) and derived its fix spans from the diff. On 150 SWE-bench tasks this diff mapping matches the curated files in every case, with 91% precision and 83% recall at function level. {{v:gemma_own.n_flagged}} fastapi tasks have tests that import packages missing from the sandbox, as the forum reports [6]; Gemma solved {{v:gemma_own.flagged_solved}} of them, and I exclude them, leaving {{v:gemma_own.n_solvable}}. As a cross-check I use another entrant's public per-task outcomes on 55 of the same tasks [1].

## 3. Results

### 3.1 Where the fix sits predicts failure

![Figure 1](fig1_where_fix_lives.png)

*Figure 1. Resolve rate by where the fix lives, for weak and strong models (mean over models; 95% CIs from resampling tasks). SWE-bench Verified, 500 tasks.*

Holding patch size, issue length and the other features fixed, the strongest structural signal is the absence of an anchor. When no name in the issue reaches the fix, the log-odds of success change by {{ci:hop_recoded.no_anchor_per_task}} relative to an anchored task (odds ×{{f2:hop_recoded.no_anchor_odds_ratio}}). Among anchored tasks, distance itself points the same way but is not distinguishable from zero ({{ci:hop_recoded.hop_anchored}} per SD). (Coding "no anchor" as distance 9 gives {{cim:pooled.hop}} but mixes the two effects.) The result does not depend on edge choice: using call edges only, without containment, gives {{ci:hop_call_only.hop}} (rank correlation with the main distance {{f2:hop_call_only.spearman_with_mixed}}).

Files touched carries its own penalty ({{ci:hop_recoded.files(log)}} per SD, independent of patch lines). An issue that names the fix file helps ({{ci:hop_recoded.file_named}}). How the fix is spread across functions, once files and patch size are controlled, adds nothing ({{ci:hop_recoded.spread}}).

These estimates are stable. Hops keep their sign with any single repository removed ({{v:pooled.hop.main_sign_stable_loro}}/{{v:n_repos}}) and without Django ({{f2:no_django.hop.main}}); files weakens without Django ({{f2:no_django.files(log).main}}). Within single scaffold releases, both effects keep their sign: v2.0.0 (11 models) gives hops {{ci:within_version.v2.0.0.hop}} and files {{ci:within_version.v2.0.0.files}}, and v1.7.0, which holds three of the five weak models, gives {{ci:within_version.v1.7.0.hop}} and {{ci:within_version.v1.7.0.files}}.

![Figure 2](fig2_per_agent.png)

*Figure 2. Per-model effect of hops (left) and files touched (right) on log-odds of success, controlling patch and issue size. Each dot is one of 38 models.*

Fitted per model, hops is negative for {{v:per_agent.hop.n_negative}} of 38 models and significant for {{v:per_agent.hop.n_negative_p05}} (Figure 2). The per-model fits share the same 500 tasks, so this is a consistency check, not 38 replications. Shuffling hops across tasks still gives all 38 the same sign {{pct:per_agent_null.share_all_same_sign}} of the time.

In raw rates, weak models resolve {{pct:cohort.hop0.weak.rate}} of tasks whose issue names the fixed function and {{pct:cohort.hop2plus.weak.rate}} of those two or more hops away. For multi-file fixes they reach **{{pct1:cohort.multi_file.weak.rate}}** [{{pct1:cohort.multi_file.weak.ci.0}}, {{pct1:cohort.multi_file.weak.ci.1}}], against {{pct:cohort.single_file.weak.rate}} for single-file fixes.

**Do weak models suffer more?** Not on the log-odds scale. Multi-file fixes cost weak models {{v:multi_file_gap.weak.log_odds}} log-odds and strong models {{v:multi_file_gap.strong.log_odds}}, the same penalty. The weak cohort's larger relative drop (75% vs 48%) is what a shared penalty produces at a lower base rate: structure makes tasks hard for every model, and weak models sit lower on that axis.

### 3.2 Static structure vs. an LLM judge

*Table 1. Predicting item-response difficulty of the 500 tasks (Spearman ρ, 5-fold CV; 95% bootstrap CI).*

| Features | ρ | leave-repo-out ρ |
|---|---|---|
| Patch size only | {{f2:difficulty.patch_size.rho}} [{{f2:difficulty.patch_size.ci.0}}, {{f2:difficulty.patch_size.ci.1}}] | {{f2:difficulty.patch_size.rho_repo_out}} |
| Code-graph features (ours) | {{f2:difficulty.graph.rho}} [{{f2:difficulty.graph.ci.0}}, {{f2:difficulty.graph.ci.1}}] | {{f2:difficulty.graph.rho_repo_out}} |
| Graph + patch size | {{f2:difficulty.graph+patch.rho}} [{{f2:difficulty.graph+patch.ci.0}}, {{f2:difficulty.graph+patch.ci.1}}] | {{f2:difficulty.graph+patch.rho_repo_out}} |
| LLM-judged features [2] | {{f2:difficulty.llm_judge.rho}} [{{f2:difficulty.llm_judge.ci.0}}, {{f2:difficulty.llm_judge.ci.1}}] | {{f2:difficulty.llm_judge.rho_repo_out}} |
| LLM-judged + graph + patch | {{f2:difficulty.llm_judge+graph+patch.rho}} [{{f2:difficulty.llm_judge+graph+patch.ci.0}}, {{f2:difficulty.llm_judge+graph+patch.ci.1}}] | {{f2:difficulty.llm_judge+graph+patch.rho_repo_out}} |

This is a negative result. For *generic* difficulty, an LLM that reads the issue, patch and tests beats free static structure, and my features add almost nothing on top. The graph predicts something narrower: which known fixes weak models miss.

### 3.3 The triage gap

Could an agent skip its hopeless tasks early? For each weak model I predict which tasks it solves, holding out whole repositories, at five information levels:

1. the issue only (names, traceback, length, repository size);
2. plus a CPU BM25 localizer's top hit and the graph around it;
3. plus where the true fix sits (hops, whether its function or file is named);
4. plus its extent (files, functions);
5. plus the true patch size.

Levels 3–5 use the true fix and are oracles. Level 4 is the information an agent has once it has planned a correct edit.

Before search, prediction is at or below chance (AUC {{rng:triage|*|before_search.auc}}). The BM25 localizer, which ranks the true fix file first {{pct:localizer.file_top1}} of the time and in its top five {{pct:localizer.file_top5}}, lifts it to {{rng:triage|*|before+bm25_localizer.auc}}. Knowing exactly where the true fix sits barely helps beyond that ({{rng:triage|*|oracle_site.auc}}). The jump comes with the fix's extent: {{rng:triage|*|oracle_site+extent.auc}}, and {{rng:triage|*|oracle_full.auc}} with true patch size. Patch size alone reaches {{rng:triage|*|patch_lines_only.auc}}. **The signal is in how big the edit is, not in where it is.**

![Figure 3](fig3_triage_gap.png)

*Figure 3. (a) AUC for predicting which tasks each weak model resolves, by information level (repository-grouped 5-fold CV; 95% CI from refitting on task resamples). (b) Share of solves lost when the predicted bottom half of tasks is capped at the model's median calls per solved task.*

**Simulating a checkpoint.** The leaderboard logs record how many calls each run spent on each task. I declared one rule before looking: rank tasks by predicted solvability, cap the bottom half at the model's median calls per solved task, and count a solve as lost if it needed more calls than the cap. Ranked by planned extent (level 4), the cap saves {{rng:checkpoint_sim|*|oracle_site+extent.calls_saved|pct}} of all calls and loses {{rng:checkpoint_sim|*|oracle_site+extent.solves_lost|pct}} of solves. Capping a random half saves {{rng:checkpoint_sim|*|random_half.calls_saved|pct}} and loses {{rng:checkpoint_sim|*|random_half.solves_lost|pct}}. BM25-level information loses {{rng:checkpoint_sim|*|before+bm25_localizer.solves_lost|pct}}, which beats random for four of five models (Figure 3b). It prices the loss; it cannot show what the saved calls would buy.

The budget at stake is large. Weak models spend {{rng:budget|*|calls_on_failures|pct}} of their calls on tasks they fail; if every task cost the same, that share would be {{rng:budget|*|calls_on_failures_if_equal_cost|pct}}.

*Table 2. Share of a weak model's calls and solves in the bottom half of predicted solvability.*

| Model (resolve rate) | Calls on failed tasks | Before search: calls / solves | Planned extent: calls / solves |
|---|---|---|---|
| Qwen2.5-Coder 32B (9%) | {{pct:budget.Qwen2.5-Coder 32B Instruct.calls_on_failures}} | {{pct:budget.Qwen2.5-Coder 32B Instruct.before_search.bottom_half_calls}} / {{pct:budget.Qwen2.5-Coder 32B Instruct.before_search.bottom_half_solves}} | {{pct:budget.Qwen2.5-Coder 32B Instruct.oracle_site+extent.bottom_half_calls}} / **{{pct:budget.Qwen2.5-Coder 32B Instruct.oracle_site+extent.bottom_half_solves}}** |
| Llama 4 Maverick (21%) | {{pct:budget.Llama 4 Maverick Instruct.calls_on_failures}} | {{pct:budget.Llama 4 Maverick Instruct.before_search.bottom_half_calls}} / {{pct:budget.Llama 4 Maverick Instruct.before_search.bottom_half_solves}} | {{pct:budget.Llama 4 Maverick Instruct.oracle_site+extent.bottom_half_calls}} / **{{pct:budget.Llama 4 Maverick Instruct.oracle_site+extent.bottom_half_solves}}** |
| gpt-oss-120b (26%) | {{pct:budget.gpt-oss-120b.calls_on_failures}} | {{pct:budget.gpt-oss-120b.before_search.bottom_half_calls}} / {{pct:budget.gpt-oss-120b.before_search.bottom_half_solves}} | {{pct:budget.gpt-oss-120b.oracle_site+extent.bottom_half_calls}} / **{{pct:budget.gpt-oss-120b.oracle_site+extent.bottom_half_solves}}** |
| GPT-5 nano (35%) | {{pct:budget.GPT 5 nano.calls_on_failures}} | {{pct:budget.GPT 5 nano.before_search.bottom_half_calls}} / {{pct:budget.GPT 5 nano.before_search.bottom_half_solves}} | {{pct:budget.GPT 5 nano.oracle_site+extent.bottom_half_calls}} / **{{pct:budget.GPT 5 nano.oracle_site+extent.bottom_half_solves}}** |
| Kimi K2 (44%) | {{pct:budget.Kimi K2 Instruct.calls_on_failures}} | {{pct:budget.Kimi K2 Instruct.before_search.bottom_half_calls}} / {{pct:budget.Kimi K2 Instruct.before_search.bottom_half_solves}} | {{pct:budget.Kimi K2 Instruct.oracle_site+extent.bottom_half_calls}} / **{{pct:budget.Kimi K2 Instruct.oracle_site+extent.bottom_half_solves}}** |

### 3.4 Gemma 4 31B on the competition's tasks

![Figure 4](fig4_gemma.png)

*Figure 4. My Gemma 4 31B run on the {{v:gemma_own.n_solvable}} solvable public competition tasks. (a) By functions the fix touches, with the SWE-bench cohorts from Figure 1. (b) By patch size.*

Gemma 4 31B resolves {{v:gemma_own.solved}} of {{v:gemma_own.n_solvable}} solvable tasks ({{pct:gemma_own.rate}}). Fixes inside one function succeed **{{pct:gemma_own.multi_func.low}}** of the time (n = {{v:gemma_own.multi_func.n_low}}), and fixes spanning several functions **{{pct:gemma_own.multi_func.high}}** (n = {{v:gemma_own.multi_func.n_high}}; Fisher p = {{p:gemma_own.multi_func.p}}). That is the same drop as the SWE-bench weak cohort (Figure 4a). The split holds after dropping tasks whose fix touches no Python definition ({{pct:gemma_own.multi_func_def_only.low}} vs {{pct:gemma_own.multi_func_def_only.high}}, p = {{p:gemma_own.multi_func_def_only.p}}). Patch size separates the tasks more sharply: {{pct:gemma_own.patch.low}} for patches of 10 lines or fewer against {{pct:gemma_own.patch.high}} above (p = {{p:gemma_own.patch.p}}). Fitted together, patch size carries the effect ({{f2:gemma_own.joint_logit.log_patch}} per log-line, p = {{p:gemma_own.joint_logit.log_patch_p}}) and function count adds nothing (p = {{p:gemma_own.joint_logit.multi_p}}). Multi-file alone is weaker ({{pct:gemma_own.multi_file.low}} vs {{pct:gemma_own.multi_file.high}}, p = {{p:gemma_own.multi_file.p}}). This matches Section 3.3: what predicts Gemma's failures is the extent of the fix.

The failures are expensive. {{v:gemma_own.n_timeout}} of {{v:gemma_own.n_tasks}} tasks hit the 6-minute limit and {{v:gemma_own.n_ctx}} overflowed the 32k context. Multi-function tasks time out {{pct:gemma_own.timeout_rate.multi}} of the time against {{pct:gemma_own.timeout_rate.single}} for one-function tasks. Capping tasks with a patch over 10 lines at the median solve time ({{int:gemma_own.checkpoint.K_s}} s) would save {{pct:gemma_own.checkpoint.patch_gt10.time_saved}} of wall-clock time and lose {{pct:gemma_own.checkpoint.patch_gt10.solves_lost}} of solves. A random cap on the same number of tasks saves {{pct:gemma_own.checkpoint.patch_gt10.random_time_saved}} and loses {{pct:gemma_own.checkpoint.patch_gt10.random_solves_lost}}, so on one run the edge is modest. The other entrant's CPU runs give the same picture on their 55 tasks ({{pct:gemma.multi_func.low}} vs {{pct:gemma.multi_func.high}} on 41 solvable) [1].

## 4. What this means for a 12-hour budget

1. **Do not pre-screen from the issue.** Before search there is nothing to learn, and a skip rule at that point mostly skips solvable tasks.
2. **Checkpoint on the planned edit.** After locating the code and before editing, the agent should state how many files and functions it will change. A multi-file or multi-function plan is the low-yield half. For such a plan, make the smallest edit at the single most likely site, run one test, submit, and move on. A submitted partial patch costs nothing.
3. **Spend the time where it pays.** Under a binding 12-hour cap, less time on low-yield tasks lets `max_time_minutes` rise for all.

I tested point 2 directly. A second full run, identical except for the checkpoint text in the prompt, solved {{v:gemma_own.checkpoint_arm.solved}} of {{v:gemma_own.n_solvable}} solvable tasks against {{v:gemma_own.checkpoint_arm.base_solved}} ({{v:gemma_own.checkpoint_arm.gained}} gained, {{v:gemma_own.checkpoint_arm.lost}} lost; exact McNemar p = {{p:gemma_own.checkpoint_arm.mcnemar_p}}) and hit the 6-minute limit on {{v:gemma_own.checkpoint_arm.timeouts}} tasks instead of {{v:gemma_own.checkpoint_arm.base_timeouts}}. The direction matches the rule, but one run per arm cannot separate it from run-to-run noise. This complements a harness study finding that call quality beats call volume for Gemma 4 [9]: it says which tasks to stop spending calls on.

## 5. Limitations

The SWE-bench outcomes come from one scaffold family across 11 releases. Within-release refits agree, but run settings and dates still differ, and SWE-bench tasks may be memorized. Levels 3–5 of Figure 3 use the true fix. An agent must estimate its own edit's extent, and the gap between that estimate and the truth is untested. Name-based call resolution over-approximates edges, and the competition's own graph tools may differ from mine. My Gemma run is one sample per task on {{v:gemma_own.n_solvable}} tasks from only four repositories. Another entrant reports 34.5% resolved locally against about 6.9% on the hidden board [1], so local rates do not transfer directly to private repositories.

## 6. Release

Apache-2.0 at github.com/Haresh276/where-the-fix-lives:

- the commit-exact graph builder;
- the diff-to-span mapper;
- per-task features for all 500 SWE-bench Verified tasks and all 129 public competition tasks;
- my per-task Gemma 4 31B outcomes on the 129 tasks and the notebook that produced them;
- one script (`paper_numbers.py`) that regenerates every number in this paper from public data on CPU.

*AI-use statement: an AI coding assistant (Claude) was used for the code, analysis and drafting. The author directed the study, reviewed the results and edited the final text.*

## References

- [1] yosh95. *g4-developer-agent-paper*: per-task Gemma 4 31B outcomes on public competition tasks. GitHub, 2026. Apache-2.0.
- [2] C. Ge, D. Kryvosheieva, D. Fried, U. Girit, K. Hariharan. Agent Psychometrics: Task-Level Performance Prediction in Agentic Coding Benchmarks. arXiv:2604.00594, 2026.
- [3] N. Balepur, K. Tomlinson, T. Schnabel. Code Understanding is a Bottleneck for Coding Agents. arXiv:2610.10610, 2026.
- [4] C. E. Jimenez et al. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR 2024; OpenAI, SWE-bench Verified, 2024.
- [5] J. Yang et al. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. NeurIPS 2024; mini-SWE-agent, 2025.
- [6] Kaggle competition forum, Gemma 4 Developer Agent Paper Track: "27 fastapi tasks cannot pass in the provided sandbox", Nov 2026.
- [7] C. S. Xia, Y. Deng, S. Dunn, L. Zhang. Agentless: Demystifying LLM-based Software Engineering Agents. arXiv:2407.01489, 2024.
- [8] Z. Chen et al. LocAgent: Graph-Guided LLM Agents for Code Localization. ACL 2025.
- [9] veljjanoski. *Better calls, not more calls: what limits a small coding agent on a fixed budget.* Gemma 4 Developer Agent Paper Track, Nov 2026; github.com/veljjanoski/gemma4-developer-agent.
