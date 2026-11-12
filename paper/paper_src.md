# Where the Fix Lives: Code-Graph Structure and the Triage Gap for Local Agents

**Haresh G.** (Kaggle: Haresh276) · Code, data and both notebooks: github.com/Haresh276/where-the-fix-lives

## Abstract

A local agent like Gemma 4 31B fails most bugs, and under a 12-hour budget its failures also eat most of the clock. I wanted to know what predicts a failure, and when the agent itself could know. First, for 38 models on all 500 SWE-bench Verified tasks, I built static code graphs at each task's base commit and joined them with per-task outcomes. Two fix-location properties predict failure beyond patch size: an issue that names nothing connected to the fix ({{ci:hop_recoded.no_anchor_per_task}} log-odds) and a fix that spans more files ({{ci:hop_recoded.files(log)}} per SD). Neither is visible from the issue (AUC {{rng:triage|*|before_search.auc}} before search), and knowing where the true fix sits adds little. What separates wins from losses is how big the edit is (AUC {{rng:triage|*|oracle_site+extent.auc}}). Second, I ran Gemma 4 31B twice on all {{v:gemma_own.n_tasks}} public competition tasks, on the competition's own 4×L4 setup, and read all {{v:gemma_own.traces.n_traces}} agent traces. Gemma opens a file the fix changes in {{pct:gemma_own.traces.found_fix_file}} of task-runs, after a median of {{int:gemma_own.traces.t_found_median}} seconds, and that predicts nothing (AUC {{f2:gemma_own.traces.auc.found_file.auc}}). What predicts success is when it first edits source code: AUC {{f2:gemma_own.traces.auc.first_edit.auc}}, above the true patch size ({{f2:gemma_own.traces.auc.oracle_patch.auc}}), and it still works when one run's timing predicts the other run's outcomes. Task-runs with no source edit by 300 of 360 seconds solved {{v:gemma_own.traces.no_edit_by_300.solved}} of {{v:gemma_own.traces.no_edit_by_300.n}} and used {{pct:gemma_own.traces.no_edit_by_300.time_share}} of all runtime. The useful signal is not where the fix lives. It is whether the agent has started writing it.

## 1. Introduction

The competition gives an agent 12 hours on four L4 GPUs and a hidden pile of bugs. In my first full run of Gemma 4 31B on the public tasks, failed tasks took twice as long as solved ones (median {{int:gemma_own.med_dur_failed}} s against {{int:gemma_own.med_dur_solved}} s) and ate {{pct:gemma_own.time_on_failures}} of the runtime. So most of the budget goes to bugs the agent will never fix. If it could tell which ones early, it could spend that time on bugs it can.

Patch size is the obvious guess, and it does matter. But two three-line patches can be completely different jobs. In one, the issue names the broken function. In the other, the broken line sits three calls away from anything the issue mentions, in a file nobody named. A code graph can measure that difference, and every agent in this competition already gets graph tools. So I asked three questions:

1. Does **where the fix sits in the code graph** predict failure, beyond patch size and issue length, across many models?
2. Can an agent see that **before** it searches, or only after it has found and planned the fix? I call the difference the *triage gap*.
3. For **Gemma 4 31B on the competition's own tasks**, what can the agent itself observe early enough to act on?

**Related work.** Agent Psychometrics [2] predicts task difficulty from features an LLM judges from the issue, patch and tests. Balepur et al. [3] show on synthetic call-graph tasks that code understanding, not editing, is what limits coding agents. Agentless [7] and LocAgent [8] treat localization as its own stage. SWE-bench [4] and SWE-agent [5] supply the tasks and scaffold. I measure real repository structure instead of asking a model to judge it, compare against [2]'s released features, and then look inside the agent's own trajectory.

## 2. Data and method

**Tasks and outcomes.** SWE-bench Verified [4] has 500 human-checked bug fixes from 12 Python repositories (Django is 46%). The leaderboard publishes per-task results for 40 runs of the bash-only *mini-SWE-agent* scaffold [5]. I dropped two runs whose per-task files disagree with their own reported score by more than 5 points. That leaves **38 models** (15 open-weight), resolving 9% to 77%. The runs span 11 scaffold releases, so every run gets its own fixed effect and Section 3.1 refits within single releases. *Weak* means the five models below 45% (Qwen2.5-Coder 32B, Llama 4 Maverick, gpt-oss-120b, GPT-5 nano, Kimi K2); *strong* means the 13 at 70% or above.

**Code graphs at every commit.** For each task I parse every non-test Python file at its base commit, straight from a bare git clone and cached by file hash, into functions, methods and classes. They are linked by call edges and by class-to-method containment. Calls resolve by name, which over-approximates, as static Python call graphs usually do. The median graph has {{int:graph.nodes_median}} nodes; the biggest has {{int:graph.nodes_max}}.

**Where the fix lives.** I map each task's curated fix spans onto graph nodes (97.5% resolve) and measure:

- *hops*: graph distance from anything the issue names (identifiers, traceback frames) to the fixed code. 0 means the issue names the fixed function. *No anchor* ({{v:hop_recoded.n_no_anchor}} tasks) means nothing the issue names reaches the fix within 8 hops;
- *files* and *functions* the fix touches, and whether the issue names the fix *file*;
- controls: patch lines and issue length.

*Example.* In `sphinx-doc__sphinx-9591` the fix is three lines in one method, three hops from the nearest thing the issue names. All 13 strong models fix it. None of the five weak models do. Patch size alone would call it easy.

**Analyses.** A pooled logistic model of 19,000 task × model outcomes with a fixed effect per run; 95% CIs from {{int:boot_reps}} task-level bootstrap resamples, plus leave-one-repository-out refits. The same model per model. Ridge regression of [2]'s item-response difficulty. And classifiers of each weak model's solves at rising information levels, with whole repositories held out.

**My Gemma runs.** The public competition tasks are merged pull requests in fastapi, requests, rich and httpx. I ran Gemma 4 31B (QAT, the default agent and tools, vLLM on 4×L4) on all {{v:gemma_own.n_tasks}} in one Kaggle session: 6 minutes, 40 tool calls and 70 turns per task, {{v:gemma_own.hours}} GPU-hours. Then I ran it again with one change, an edit-plan checkpoint in the prompt (Section 4). I rebuilt every task from public git history and took its fix spans from the diff; on 150 SWE-bench tasks this matches the curated fix files every time (91% precision and 83% recall at function level). {{v:gemma_own.n_flagged}} fastapi tasks import packages the sandbox does not have [6]. Gemma solved {{v:gemma_own.flagged_solved}} of them, so I drop them, leaving {{v:gemma_own.n_solvable}}. Every run also saves a full trace per task: each model turn and tool call, with timestamps. Section 3.5 uses all {{v:gemma_own.traces.n_traces}} of them.

## 3. Results

### 3.1 Where the fix sits predicts failure

![Figure 1](fig1_where_fix_lives.png)

*Figure 1. Resolve rate by where the fix lives, weak vs strong models (mean over models; 95% CIs from resampling tasks). SWE-bench Verified, 500 tasks.*

Holding patch size, issue length and everything else fixed, the strongest signal is a missing anchor. When nothing in the issue reaches the fix, the log-odds of success drop by {{ci:hop_recoded.no_anchor_per_task}} (odds ×{{f2:hop_recoded.no_anchor_odds_ratio}}). Among anchored tasks, distance points the same way but I cannot tell it from zero ({{ci:hop_recoded.hop_anchored}} per SD). Using call edges only, without containment, changes nothing ({{ci:hop_call_only.hop}}).

Files touched carries its own penalty ({{ci:hop_recoded.files(log)}} per SD, on top of patch lines). An issue that names the fix file helps ({{ci:hop_recoded.file_named}}). How spread out the fix is adds nothing once files and size are known ({{ci:hop_recoded.spread}}).

I tried to break these. Hops keep their sign with any one repository removed ({{v:pooled.hop.main_sign_stable_loro}}/{{v:n_repos}}) and without Django ({{f2:no_django.hop.main}}). Within single scaffold releases both effects hold: v2.0.0 (11 models) gives hops {{ci:within_version.v2.0.0.hop}} and files {{ci:within_version.v2.0.0.files}}; v1.7.0, home to three of the five weak models, gives {{ci:within_version.v1.7.0.hop}} and {{ci:within_version.v1.7.0.files}}.

![Figure 2](fig2_per_agent.png)

*Figure 2. Per-model effect of hops (left) and files touched (right) on log-odds of success. Each dot is one of 38 models.*

Per model, hops is negative for {{v:per_agent.hop.n_negative}} of 38 and significant for {{v:per_agent.hop.n_negative_p05}} (Figure 2). The fits share the same 500 tasks, so this is a consistency check, not 38 replications. In raw rates, weak models resolve {{pct:cohort.single_file.weak.rate}} of single-file fixes and **{{pct1:cohort.multi_file.weak.rate}}** of multi-file fixes.

**Do weak models suffer more?** Not on the log-odds scale. Multi-file fixes cost weak models {{v:multi_file_gap.weak.log_odds}} log-odds and strong models {{v:multi_file_gap.strong.log_odds}}. Same penalty. Weak models just start lower.

### 3.2 Static structure vs. an LLM judge

This one is a loss, and I am reporting it anyway. For general task difficulty, [2]'s LLM-judged features reach Spearman ρ = {{f2:difficulty.llm_judge.rho}}. Patch size alone gets {{f2:difficulty.patch_size.rho}}, my graph features plus patch size {{f2:difficulty.graph+patch.rho}}, and adding mine to theirs {{f2:difficulty.llm_judge+graph+patch.rho}}. The graph does not beat an LLM at "how hard is this task". It answers a narrower question: which known fixes weak models miss.

### 3.3 The triage gap

Could an agent skip its hopeless tasks early? For each weak model I predict which tasks it solves, holding out whole repositories, with five levels of information:

1. the issue only;
2. plus a CPU BM25 localizer's top hit and the graph around it;
3. plus where the true fix sits;
4. plus how many files and functions it spans;
5. plus the true patch size.

Levels 3 to 5 use the true fix, so they are oracles. Level 4 is what an agent knows once it has planned a correct edit.

![Figure 3](fig3_triage_gap.png)

*Figure 3. (a) AUC for predicting which tasks each weak model solves, by information level (repository-grouped CV; 95% CI from refitting on task resamples). (b) Solves lost when the predicted bottom half is capped at the model's median calls per solved task.*

Before search, prediction is at or below chance (AUC {{rng:triage|*|before_search.auc}}). The BM25 localizer finds the fix file first {{pct:localizer.file_top1}} of the time and lifts it to {{rng:triage|*|before+bm25_localizer.auc}}. Knowing exactly where the true fix sits barely helps ({{rng:triage|*|oracle_site.auc}}). The jump comes with the fix's extent: {{rng:triage|*|oracle_site+extent.auc}}, and {{rng:triage|*|oracle_full.auc}} with true patch size. **The signal is in how big the edit is, not where it is.**

I fixed one rule before looking: cap the predicted bottom half at the model's median calls per solved task, and count a solve as lost if it needed more. Ranked by extent, the cap saves {{rng:checkpoint_sim|*|oracle_site+extent.calls_saved|pct}} of calls and loses {{rng:checkpoint_sim|*|oracle_site+extent.solves_lost|pct}} of solves. A random half loses {{rng:checkpoint_sim|*|random_half.solves_lost|pct}}. Weak models spend {{rng:budget|*|calls_on_failures|pct}} of their calls on tasks they fail, so the stakes are not small.

### 3.4 Gemma 4 31B on the competition's tasks

![Figure 4](fig4_gemma.png)

*Figure 4. My first Gemma 4 31B run, {{v:gemma_own.n_solvable}} solvable competition tasks. (a) By functions the fix touches, with the SWE-bench cohorts. (b) By patch size.*

Gemma solves {{v:gemma_own.solved}} of {{v:gemma_own.n_solvable}} ({{pct:gemma_own.rate}}). One-function fixes succeed **{{pct:gemma_own.multi_func.low}}** of the time (n = {{v:gemma_own.multi_func.n_low}}) and multi-function fixes **{{pct:gemma_own.multi_func.high}}** (n = {{v:gemma_own.multi_func.n_high}}; Fisher p = {{p:gemma_own.multi_func.p}}), the same drop as the weak SWE-bench models. Patch size splits harder: {{pct:gemma_own.patch.low}} at 10 lines or fewer, {{pct:gemma_own.patch.high}} above (p = {{p:gemma_own.patch.p}}), and fitted together, patch size carries all of it (function count p = {{p:gemma_own.joint_logit.multi_p}}). Another entrant's CPU runs give the same split on their 55 tasks ({{pct:gemma.multi_func.low}} vs {{pct:gemma.multi_func.high}}) [1]. {{v:gemma_own.n_timeout}} of {{v:gemma_own.n_tasks}} tasks hit the 6-minute wall.

### 3.5 Inside the runs: what the agent can see

Everything so far uses the true fix, which no agent has. The traces show what Gemma actually did.

![Figure 5](fig5_traces.png)

*Figure 5. Both Gemma 4 31B runs, {{v:gemma_own.traces.n_tasks}} tasks each. (a) When each task-run first edited a source file. (b) AUC for solved vs failed task-runs (95% CI from resampling tasks).*

**Finding the file is not the problem.** Gemma opens a file the fix changes in {{pct:gemma_own.traces.found_fix_file}} of task-runs, after a median of {{int:gemma_own.traces.t_found_median}} seconds, usually with its very first tool call. How fast it gets there says nothing about whether it will succeed (AUC {{f2:gemma_own.traces.auc.found_file.auc}}). This is Section 3.3 again, seen from inside the agent.

**The time goes between finding and editing.** The median first source edit arrives at {{int:gemma_own.traces.t_first_edit_median}} s, {{int:gemma_own.traces.found_to_edit_median}} s after the file was found. {{pct:gemma_own.traces.time_before_first_edit}} of all runtime passes before any source edit at all. The prompt passes 14k tokens in {{pct:gemma_own.traces.ctx_near_limit}} of task-runs, with only 16k left for input. Of {{v:gemma_own.traces.funnel_total.n}} task-runs, {{v:gemma_own.traces.funnel_total.found}} found the file, {{v:gemma_own.traces.funnel_total.edited}} edited it and {{v:gemma_own.traces.funnel_total.solved}} solved the task. Gemma finds the right file in seconds and then spends most of the clock not editing it.

**The first edit is the signal.** When the first source edit happens separates solved from failed task-runs with AUC {{auc:gemma_own.traces.auc.first_edit}}, above the true patch size ({{auc:gemma_own.traces.auc.oracle_patch}}). An agent can know this, and an oracle feature cannot beat it. I checked that it is not one run's clock fooling me: the two runs agree on {{pct:gemma_own.traces.cross_run.outcome_agreement}} of outcomes, and first-edit time in one run predicts the *other* run's outcomes with AUC {{f2:gemma_own.traces.cross_run.auc_b_to_a}} and {{f2:gemma_own.traces.cross_run.auc_a_to_b}}. Task-runs that edit before half-time (180 s) solve {{pct:gemma_own.traces.half_time.early_rate}}; the rest solve {{pct:gemma_own.traces.half_time.late_rate}}. No edit by 300 s: {{v:gemma_own.traces.no_edit_by_300.solved}} solved out of {{v:gemma_own.traces.no_edit_by_300.n}}, using {{pct:gemma_own.traces.no_edit_by_300.time_share}} of all runtime.

**Its plans are honest but not useful.** In the checkpoint run Gemma wrote a plan in {{v:gemma_own.traces.plan.n_stated}} of {{v:gemma_own.traces.plan.n_run}} tasks, and the plans name the right number of files {{pct:gemma_own.traces.plan.files_match}} of the time. But plan size barely predicts the outcome (AUC {{f2:gemma_own.traces.plan.auc_plan_lines}}, against {{f2:gemma_own.traces.plan.auc_oracle_patch_same_tasks}} for true patch size on the same tasks): {{v:gemma_own.traces.plan.big_true_planned_small}} of {{v:gemma_own.traces.plan.big_true}} fixes over 10 lines were planned as 10 lines or fewer. Gemma, like me, is an optimist about line counts. What does matter is whether a plan shows up at all. Tasks with no plan solved {{v:gemma_own.traces.plan.solved_not_stated}} of {{v:gemma_own.traces.plan.n_not_stated}}, although {{pct:gemma_own.traces.plan.not_stated_found_file}} of them had found the file.

**A stop rule.** Stopping every task with no source edit at 180 s would save {{rng:gemma_own.traces.stop_rule|*|time_saved|pct}} of runtime and lose {{rng:gemma_own.traces.stop_rule|*|solves_lost|pct}} of solves. Stopping the same number by true patch size loses {{rng:gemma_own.traces.stop_rule|*|oracle_solves_lost|pct}}; at random, {{rng:gemma_own.traces.stop_rule|*|random_solves_lost|pct}}. The agent's own clock matches or beats the oracle. One more thing the traces show: the competition's code-graph tools were almost never used, {{v:gemma_own.traces.graph_calls}} of {{int:gemma_own.traces.all_calls}} tool calls, in {{v:gemma_own.traces.graph_tasks}} task-runs, and none of those were solved.

## 4. What I would do with 12 hours

1. **Don't pre-screen from the issue.** Before search there is nothing to read.
2. **Don't reward finding the file.** Gemma finds it in seconds either way.
3. **Watch the first edit.** No source edit by half-time puts a task at {{pct:gemma_own.traces.half_time.late_rate}}. Make the smallest edit at the best site, test once, submit. No edit by 300 s: stop.
4. **Ask for a plan before the edit.** A plan that never shows up is the earliest warning. The plan's size is not.
5. **Give the saved time to tasks that are already editing,** by raising `max_time_minutes` for them.

I tested point 4 on the real setup. A second full run, identical except for the plan text in the prompt, solved {{v:gemma_own.checkpoint_arm.solved}} of {{v:gemma_own.n_solvable}} against {{v:gemma_own.checkpoint_arm.base_solved}} ({{v:gemma_own.checkpoint_arm.gained}} gained, {{v:gemma_own.checkpoint_arm.lost}} lost; exact McNemar p = {{p:gemma_own.checkpoint_arm.mcnemar_p}}) and hit the 6-minute wall {{v:gemma_own.checkpoint_arm.timeouts}} times instead of {{v:gemma_own.checkpoint_arm.base_timeouts}}. Right direction, not significant. One run per arm cannot tell a 3-task gain from luck, and I would rather say so than round it up. Points 3 and 5 I simulated and did not run. That is the next experiment, and it costs one more Kaggle session. This fits a harness study finding that call quality beats call volume for Gemma 4 [9]; the first edit says which tasks to stop spending calls on.

## 5. Limitations

The SWE-bench outcomes come from one scaffold family across 11 releases. Within-release refits agree, but settings and dates differ, and SWE-bench tasks may be memorized. Figure 3's levels 3 to 5 use the true fix; Section 3.5 measures the agent's own view, but only for Gemma, only under a 6-minute cap, and it identifies source edits from tool calls by filename rules. Name-based call resolution over-approximates edges. My Gemma data are two runs of {{v:gemma_own.n_solvable}} tasks from four repositories. Another entrant reports 34.5% resolved locally against about 6.9% on the hidden board [1], so local rates do not carry over to private repositories.

## 6. Release

Apache-2.0 at github.com/Haresh276/where-the-fix-lives: the commit-exact graph builder; the diff-to-span mapper; per-task features for all 500 SWE-bench Verified tasks and all 129 competition tasks; per-task outcomes and trace features for both Gemma runs, with both notebooks; and one script (`paper_numbers.py`) that regenerates every number here, on CPU. Every number in this paper is filled from that script's output, not typed by hand.

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
