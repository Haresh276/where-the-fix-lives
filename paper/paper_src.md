# Where the Fix Lives: Code-Graph Structure and the Triage Gap for Local Agents

**Haresh G.** (Kaggle: Haresh276) · Code, data and all three notebooks: github.com/Haresh276/where-the-fix-lives

## Abstract

Gemma 4 31B fails most of the bugs it is given, and the failures eat most of its 12 hours. I asked what predicts a failure, and how early the agent itself could tell. For 38 models on all 500 SWE-bench Verified tasks, I built code graphs at each base commit. Beyond patch size, failure follows an issue that names nothing connected to the fix ({{ci:hop_recoded.no_anchor_per_task}} log-odds) and a fix spanning more files ({{ci:hop_recoded.files(log)}} per SD). Neither is visible from the issue (AUC {{rng:triage|*|before_search.auc}}), and knowing where the fix sits adds little. How big the edit is does separate wins from losses (AUC {{rng:triage|*|oracle_site+extent.auc}}). Then I ran Gemma 4 31B three times on the competition's own tasks and setup, and read every trace. Gemma opens a file the fix changes in {{pct:gemma_own.traces.found_fix_file}} of task-runs, after a median of {{int:gemma_own.traces.t_found_median}} seconds, and that predicts nothing (AUC {{f2:gemma_own.traces.auc.found_file.auc}}). What predicts success is when it first edits source code (AUC {{f2:gemma_own.traces.auc.first_edit.auc}}, against {{f2:gemma_own.traces.auc.oracle_patch.auc}} for the true patch size). At 180 seconds, runs that have edited go on to solve {{pct:gemma_own.traces.landmark.180.edited_rate}}, the rest {{pct:gemma_own.traces.landmark.180.not_edited_rate}}. A third run with 10 minutes per task instead of 6 solved {{v:gemma_own.tenmin.solved}} of {{v:gemma_own.tenmin.n}}, between the two 6-minute runs ({{v:gemma_own.tenmin.run1_solved}} and {{v:gemma_own.tenmin.run2_solved}}), for {{pct:gemma_own.tenmin.extra_time}} more time. Finding the fix is easy for Gemma. Starting to write it is the bottleneck, and more minutes do not fix that.

## 1. Introduction

The competition gives an agent 12 hours on four L4 GPUs and a hidden pile of bugs. In my first full run of Gemma 4 31B on the public tasks, failed tasks took twice as long as solved ones (median {{int:gemma_own.med_dur_failed}} s against {{int:gemma_own.med_dur_solved}} s) and used {{pct:gemma_own.time_on_failures}} of the runtime. Most of the budget goes to bugs the agent was never going to fix.

Patch size is the obvious guess. But two three-line patches can be different jobs: in one the issue names the broken function, in the other the broken line is three calls away from anything the issue mentions. A code graph can measure that, and every agent here already gets graph tools. So I asked three questions.

1. Does **where the fix sits in the code graph** predict failure, beyond patch size and issue length, across many models?
2. Can an agent see that **before** it searches, or only after it has found and planned the fix? I call the difference the *triage gap*.
3. For **Gemma 4 31B on the competition's own tasks**, what can the agent itself notice early enough to act on?

**Related work.** Agent Psychometrics [2] predicts task difficulty from LLM-judged features. Balepur et al. [3] show on synthetic call graphs that understanding code limits agents more than editing it. Agentless [7] and LocAgent [8] treat localization as its own stage. I measure real repository structure instead, compare it with [2], and then look inside the agent's own trajectory.

## 2. Data and method

**Tasks and outcomes.** SWE-bench Verified [4] has 500 human-checked bug fixes from 12 Python repositories (Django alone is 46%). The leaderboard publishes per-task results for 40 runs of the bash-only *mini-SWE-agent* scaffold [5]. I dropped two whose per-task files disagree with their reported score by over 5 points, leaving **38 models** (15 open-weight, 9% to 77% resolved). The runs span 11 scaffold releases, so each gets its own fixed effect. *Weak* means the five models below 45% (Qwen2.5-Coder 32B, Llama 4 Maverick, gpt-oss-120b, GPT-5 nano, Kimi K2). *Strong* means the 13 at 70% or above.

**Code graphs at every commit.** For each task I parse every non-test Python file at its base commit into functions, methods and classes, linked by call edges (resolved by name, which over-approximates) and class-to-method containment. The median graph has {{int:graph.nodes_median}} nodes and the biggest has {{int:graph.nodes_max}}.

**Where the fix lives.** I map each task's curated fix spans onto graph nodes (97.5% resolve) and measure the following.

- *Hops*: graph distance from anything the issue names (identifiers, traceback frames) to the fixed code. *No anchor* ({{v:hop_recoded.n_no_anchor}} tasks) means nothing the issue names reaches the fix within 8 hops.
- *Files* and *functions* the fix touches, and whether the issue names the fix *file*.
- Patch lines and issue length, as controls.

*Example.* In `sphinx-doc__sphinx-9591` the fix is three lines in one method, three hops from anything the issue names. All 13 strong models fix it and none of the five weak ones do.

**Analyses.** A pooled logistic model of 19,000 task × model outcomes with run fixed effects and 95% CIs from {{int:boot_reps}} task-level bootstrap resamples; per-model fits; and classifiers of each weak model's solves at rising information levels, with whole repositories held out.

**My Gemma runs.** The public competition tasks are merged pull requests in fastapi, requests, rich and httpx. I ran Gemma 4 31B (QAT, the default agent and tools, vLLM on 4×L4) on all {{v:gemma_own.n_tasks}} in one Kaggle session, with 6 minutes, 40 tool calls and 70 turns per task and two tasks at a time ({{v:gemma_own.hours}} hours of agent time). Then I ran it twice more, each time changing one thing: an edit-plan checkpoint in the prompt (Section 4), and 10 minutes per task instead of 6 (Section 3.6). I rebuilt every task from public git history and took fix spans from the diff (on 150 SWE-bench tasks this matches the curated fix files every time). {{v:gemma_own.n_flagged}} fastapi tasks import packages the sandbox does not have [6]. Gemma solved {{v:gemma_own.flagged_solved}} of them, so I drop them and keep {{v:gemma_own.n_solvable}}. Every run saves a timestamped trace per task. Section 3.5 uses the {{v:gemma_own.traces.n_traces}} traces of the first two runs.

## 3. Results

### 3.1 Where the fix sits predicts failure

![Figure 1](fig1_where_fix_lives.png)

*Figure 1. Resolve rate by where the fix lives, weak vs strong models (mean over models; 95% CIs from resampling tasks). SWE-bench Verified, 500 tasks.*

With patch size, issue length and everything else held fixed, the strongest signal is a missing anchor. When nothing in the issue reaches the fix, the log-odds of success drop by {{ci:hop_recoded.no_anchor_per_task}} (odds ×{{f2:hop_recoded.no_anchor_odds_ratio}}). Among anchored tasks, distance points the same way, but I cannot separate it from zero ({{ci:hop_recoded.hop_anchored}} per SD).

Files touched has its own penalty ({{ci:hop_recoded.files(log)}} per SD, on top of patch lines). An issue that names the fix file helps a little ({{ci:hop_recoded.file_named}}). How spread out the fix is adds nothing once files and size are known ({{ci:hop_recoded.spread}}).

I tried hard to break these. Hops keep their sign with any one repository removed ({{v:pooled.hop.main_sign_stable_loro}}/{{v:n_repos}}), without Django ({{f2:no_django.hop.main}}), and with call edges only ({{ci:hop_call_only.hop}}). Both effects hold within single scaffold releases: v2.0.0 (11 models) gives hops {{ci:within_version.v2.0.0.hop}} and files {{ci:within_version.v2.0.0.files}}.

![Figure 2](fig2_per_agent.png)

*Figure 2. Per-model effect of hops (left) and files touched (right) on log-odds of success. Each dot is one of 38 models.*

Per model, hops is negative for {{v:per_agent.hop.n_negative}} of 38 and significant for {{v:per_agent.hop.n_negative_p05}} (Figure 2). The fits share the same 500 tasks, so this is a consistency check, not 38 replications. In raw rates, weak models resolve {{pct:cohort.single_file.weak.rate}} of single-file fixes and only **{{pct1:cohort.multi_file.weak.rate}}** of multi-file fixes.

Do weak models suffer more? No. Multi-file fixes cost weak models {{v:multi_file_gap.weak.log_odds}} log-odds and strong models {{v:multi_file_gap.strong.log_odds}}. Weak models just start lower.

### 3.2 Static structure vs. an LLM judge

I lost this one, and I am reporting it anyway. For general task difficulty, [2]'s LLM-judged features reach Spearman ρ = {{f2:difficulty.llm_judge.rho}}, against {{f2:difficulty.graph+patch.rho}} for my graph features plus patch size, and {{f2:difficulty.llm_judge+graph+patch.rho}} for both together. The graph answers a narrower question: which known fixes the weak models miss.

### 3.3 The triage gap

Could an agent skip its hopeless tasks early? For each weak model I predict which tasks it solves at five levels of information: (1) the issue only; (2) plus a CPU BM25 localizer's top hit and its graph neighbourhood; (3) plus where the true fix sits; (4) plus how many files and functions it spans; (5) plus the true patch size. Levels 3 to 5 are oracles. Level 4 is roughly what an agent knows once it has planned a correct edit.

![Figure 3](fig3_triage_gap.png)

*Figure 3. (a) AUC for predicting which tasks each weak model solves, by information level (repository-grouped CV; 95% CI from refitting on task resamples). (b) Solves lost when the predicted bottom half is capped at the model's median calls per solved task.*

Before search, prediction is at chance or worse (AUC {{rng:triage|*|before_search.auc}}). The BM25 localizer finds the fix file first {{pct:localizer.file_top1}} of the time and lifts this to {{rng:triage|*|before+bm25_localizer.auc}}. Knowing exactly where the true fix sits barely helps ({{rng:triage|*|oracle_site.auc}}), which honestly surprised me, since this paper started from the idea that location matters. The jump comes with the fix's extent, to {{rng:triage|*|oracle_site+extent.auc}}, and {{rng:triage|*|oracle_full.auc}} with the true patch size. **How big the edit is matters much more than where it is.**

I fixed one rule before looking at the results. Cap the predicted bottom half at the model's median calls per solved task, and count a solve as lost if it needed more. Ranked by extent, the cap saves {{rng:checkpoint_sim|*|oracle_site+extent.calls_saved|pct}} of calls and loses {{rng:checkpoint_sim|*|oracle_site+extent.solves_lost|pct}} of solves. A random half loses {{rng:checkpoint_sim|*|random_half.solves_lost|pct}}. Weak models spend {{rng:budget|*|calls_on_failures|pct}} of their calls on tasks they fail.

### 3.4 Gemma 4 31B on the competition's tasks

![Figure 4](fig4_gemma.png)

*Figure 4. My first Gemma 4 31B run, {{v:gemma_own.n_solvable}} solvable competition tasks. (a) By functions the fix touches, with the SWE-bench cohorts. (b) By patch size.*

Gemma solves {{v:gemma_own.solved}} of {{v:gemma_own.n_solvable}} ({{pct:gemma_own.rate}}). One-function fixes succeed **{{pct:gemma_own.multi_func.low}}** of the time (n = {{v:gemma_own.multi_func.n_low}}) and multi-function fixes **{{pct:gemma_own.multi_func.high}}** (n = {{v:gemma_own.multi_func.n_high}}, Fisher p = {{p:gemma_own.multi_func.p}}), the same drop the weak SWE-bench models show. Patch size splits harder, with {{pct:gemma_own.patch.low}} at 10 lines or fewer and {{pct:gemma_own.patch.high}} above (p = {{p:gemma_own.patch.p}}). Fitted together, patch size takes all of it (function count p = {{p:gemma_own.joint_logit.multi_p}}). Another entrant's runs give the same split ({{pct:gemma.multi_func.low}} vs {{pct:gemma.multi_func.high}}) [1].

### 3.5 Inside the runs: what the agent can see

Everything so far uses the true fix, which no agent has. The traces show what Gemma did instead.

![Figure 5](fig5_traces.png)

*Figure 5. First two Gemma 4 31B runs, {{v:gemma_own.traces.n_tasks}} tasks each. (a) Among task-runs still going at each second, the share that is eventually solved, split by whether a source file has been edited yet (95% Wilson bands). (b) AUC for solved vs failed task-runs (95% CI from resampling tasks).*

**Finding the file is the easy part.** Gemma opens a file the fix changes in {{pct:gemma_own.traces.found_fix_file}} of task-runs, after a median of {{int:gemma_own.traces.t_found_median}} seconds, usually with its very first tool call. How fast it gets there says nothing about success (AUC {{f2:gemma_own.traces.auc.found_file.auc}}), which is Section 3.3 again, from inside the agent.

**The time goes between finding and editing.** The median first source edit comes {{int:gemma_own.traces.found_to_edit_median}} s after the file was found, and {{pct:gemma_own.traces.time_before_first_edit}} of all runtime passes before any source edit. Of {{v:gemma_own.traces.funnel_total.n}} task-runs, {{v:gemma_own.traces.funnel_total.found}} found the file, {{v:gemma_own.traces.funnel_total.edited}} edited it and {{v:gemma_own.traces.funnel_total.solved}} solved the task. Gemma gets to the right file in seconds and then spends the clock reading around it. I know this feeling from exam prep.

**The first edit is the signal.** The time of the first source edit separates solved from failed task-runs with AUC {{auc:gemma_own.traces.auc.first_edit}}, better than the true patch size ({{auc:gemma_own.traces.auc.oracle_patch}}). But solved runs end early, which flatters any timing feature, so Figure 5a asks only what the agent knows at each second, among runs still going. At 180 s, {{v:gemma_own.traces.landmark.180.running}} task-runs are still going. The {{v:gemma_own.traces.landmark.180.edited}} that have edited source go on to solve {{pct:gemma_own.traces.landmark.180.edited_rate}}, and the {{v:gemma_own.traces.landmark.180.not_edited}} that have not solve {{pct:gemma_own.traces.landmark.180.not_edited_rate}}. At 240 s it is {{pct:gemma_own.traces.landmark.240.edited_rate}} against {{pct:gemma_own.traces.landmark.240.not_edited_rate}}. At 300 s, {{v:gemma_own.traces.landmark.300.not_edited_solved}} of the {{v:gemma_own.traces.landmark.300.not_edited}} unedited runs were ever solved. Having edited by 180 s still adds {{v:gemma_own.traces.joint_180.ed}} [{{v:gemma_own.traces.joint_180.ed_ci.0}}, {{v:gemma_own.traces.joint_180.ed_ci.1}}] log-odds once the true patch size is in the model (p = {{p:gemma_own.traces.joint_180.ed_p}}). First-edit time in one run predicts the *other* run's outcomes (AUC {{f2:gemma_own.traces.cross_run.auc_b_to_a}} and {{f2:gemma_own.traces.cross_run.auc_a_to_b}}), and it holds within fastapi ({{f2:gemma_own.traces.by_repo.fastapi.auc_first_edit}}) and rich ({{f2:gemma_own.traces.by_repo.rich.auc_first_edit}}).

**Its plans are honest but not very useful.** In the checkpoint run Gemma wrote a plan in {{v:gemma_own.traces.plan.n_stated}} of {{v:gemma_own.traces.plan.n_run}} tasks, naming the right number of files {{pct:gemma_own.traces.plan.files_match}} of the time. But plan size barely predicts the outcome (AUC {{f2:gemma_own.traces.plan.auc_plan_lines}}). {{v:gemma_own.traces.plan.big_true_planned_small}} of {{v:gemma_own.traces.plan.big_true}} fixes over 10 lines were planned as 10 lines or fewer. Gemma, like me, is an optimist about line counts. What matters is whether a plan shows up at all. Tasks with no plan solved {{v:gemma_own.traces.plan.solved_not_stated}} of {{v:gemma_own.traces.plan.n_not_stated}}, even though {{pct:gemma_own.traces.plan.not_stated_found_file}} of them had found the file.

**An alarm, not an axe.** Stopping every task with no source edit at 180 s saves {{rng:gemma_own.traces.stop_rule|*|time_saved|pct}} of runtime and loses {{rng:gemma_own.traces.stop_rule|*|solves_lost|pct}} of solves. Stopping as many by true patch size loses {{rng:gemma_own.traces.stop_rule|*|oracle_solves_lost|pct}}, and at random {{rng:gemma_own.traces.stop_rule|*|random_solves_lost|pct}}. The clock ranks as well as the oracle, but ranking is not paying off. A stopped task has no patch, so the agent knows it failed and can retry it. One retry per stopped task, with the outcome taken from the other run, wins back only {{v:gemma_own.traces.stop_retry.180.0.retry_gain}} and {{v:gemma_own.traces.stop_retry.180.1.retry_gain}} of the {{v:gemma_own.traces.stop_retry.180.0.lost}} and {{v:gemma_own.traces.stop_retry.180.1.lost}} solves lost. At 300 s the rule is free (no solve lost in either run), but it saves only {{pct1:gemma_own.traces.landmark.300.not_edited_rest_share}} of runtime, because the 6-minute wall already cuts those runs short. One more thing from the traces. The competition's code-graph tools were almost never used, {{v:gemma_own.traces.graph_calls}} of {{int:gemma_own.traces.all_calls}} tool calls in {{v:gemma_own.traces.graph_tasks}} task-runs, and none of those task-runs were solved. Slightly awkward for a paper about code graphs.

### 3.6 Does more time help? A third run

The other half of the idea was to give time to tasks that are still working. {{v:gemma_own.n_timeout}} of {{v:gemma_own.n_tasks}} tasks hit the 6-minute wall in the first run, and no simulation can say what more time would do, so I measured it. A third full run on all {{v:gemma_own.tenmin.n}} solvable tasks was identical to the first except for 10 minutes per task instead of 6.

It solved {{v:gemma_own.tenmin.solved}} of {{v:gemma_own.tenmin.n}}. The two 6-minute runs solved {{v:gemma_own.tenmin.run1_solved}} and {{v:gemma_own.tenmin.run2_solved}} of the same tasks ({{v:gemma_own.tenmin.vs_run1.gained}} gained and {{v:gemma_own.tenmin.vs_run1.lost}} lost against the first, McNemar p = {{p:gemma_own.tenmin.vs_run1.p}}). {{pct:gemma_own.tenmin.after_360_share}} of the run's time came after the old 6-minute mark, and it produced {{v:gemma_own.tenmin.solved_after_360}} solves, about what one run differs from the next by luck. With more time the 40-call cap starts to bind instead ({{v:gemma_own.tenmin.call_cap}} task-runs used all 40 calls, against {{v:gemma_own.tenmin.run1_call_cap}}).

The clock still predicts (first-edit AUC {{f2:gemma_own.tenmin.auc_first_edit}}), but the late cut-off moves with the budget. {{v:gemma_own.tenmin.landmark.300.not_edited_solved}} of the {{v:gemma_own.tenmin.landmark.300.not_edited}} runs with no edit at 300 s were solved this time, after first edits at {{v:gemma_own.tenmin.late_edit_solves.edit_times.0}} and {{v:gemma_own.tenmin.late_edit_solves.edit_times.1}} s. No run without a source edit by 400 s was solved, and stopping those would have saved {{pct:gemma_own.tenmin.stop.400.saved_share}} of the time.

## 4. What I would do with 12 hours

1. **Don't pre-screen from the issue, and don't reward finding the file.** Neither says anything.
2. **Use the first edit as an alarm.** No source edit by 180 s puts a task at {{pct:gemma_own.traces.landmark.180.not_edited_rate}}. That is the moment to push for the smallest edit at the best site, not to kill the task. No edit by 300 s of 360, or 400 s of 600, means stop. Neither cost a solve.
3. **Ask for a plan before the edit.** A missing plan is the earliest warning sign.
4. **Don't buy minutes.** Going from 6 to 10 minutes bought {{v:gemma_own.tenmin.solved_after_360}} late solves and no net gain. What helps has to change what Gemma does before its first edit, not how long it may keep reading.

I tested point 3 on the real setup. The second run, identical except for the plan text in the prompt, solved {{v:gemma_own.checkpoint_arm.solved}} of {{v:gemma_own.n_solvable}} against {{v:gemma_own.checkpoint_arm.base_solved}} ({{v:gemma_own.checkpoint_arm.gained}} gained, {{v:gemma_own.checkpoint_arm.lost}} lost, exact McNemar p = {{p:gemma_own.checkpoint_arm.mcnemar_p}}) and hit the 6-minute wall {{v:gemma_own.checkpoint_arm.timeouts}} times instead of {{v:gemma_own.checkpoint_arm.base_timeouts}}. Right direction, not significant. One run per arm cannot tell a 3-task gain from luck, and I would rather say that than round it up. This fits a harness study which found that call quality matters more than call volume for Gemma 4 [9]. The first edit tells you which tasks to stop spending calls on.

## 5. Limitations

The SWE-bench outcomes come from one scaffold family, and the tasks may be memorized. Figure 3's levels 3 to 5 use the true fix. Section 3.5 measures the agent's own view, but only for Gemma, and it decides which tool calls are source edits using filename rules. The third run changed the time limit but not the 40-call cap, so it tests more time, not more calls. My Gemma data are three runs on at most {{v:gemma_own.n_solvable}} tasks from four repositories. Another entrant reports 34.5% resolved locally against about 6.9% on the hidden board [1], so local rates do not carry over to private repositories.

## 6. Release

Apache-2.0 at github.com/Haresh276/where-the-fix-lives. It has the commit-exact graph builder, the diff-to-span mapper, per-task features for all 500 SWE-bench Verified and 129 competition tasks, outcomes and trace features for all three Gemma runs, the three notebooks, and one CPU script (`paper_numbers.py`) whose output fills in every number in this paper.

*AI-use statement: an AI assistant (Claude) wrote the code, ran the analysis, ran the Kaggle experiments on my account, and drafted this text in my style. I set the goal, directed the work, pushed back on the findings, and approved the final version.*

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
