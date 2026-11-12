# Where the Fix Lives

Code, data and paper for **"Where the Fix Lives: Code-Graph Structure and the Triage Gap for Local Agents"**
by Haresh G. (Kaggle: Haresh276), Gemma 4 Developer Agent Paper Track, 2026.

**Question.** Which properties of a bug's fix location (whether the issue names anything connected to it in the code
graph, how far it is, how many files and functions it touches) decide whether a coding agent fixes it, and when could
an agent know?

**Short answer.** On SWE-bench Verified (38 models), an issue that names nothing connected to the fix, and a fix
that spans several files, predict failure; neither is visible before search (AUC 0.45-0.54), and the size of the edit
is what separates wins from losses (AUC 0.63-0.72). Across three full Gemma 4 31B runs on the public competition
tasks, Gemma finds a file the fix changes in 92% of task-runs within a median 12 s, and that predicts nothing
(AUC 0.44). When it first edits source code predicts success (AUC 0.81, above the true patch size at 0.71). At 180 s,
runs that have edited go on to solve 49% and the rest 13%, and editing by 180 s still adds 1.74 log-odds once the
true patch size is known. An edit-plan checkpoint moved solves from 29 to 32 of 102 (p = 0.375). Ten minutes per task
instead of six solved 30 of 102, for 25% more time: more minutes do not help.

## Reproduce every number in the paper

```bash
pip install -r requirements.txt
python src/fetch_data.py                                   # public inputs + bare git clones (~2 GB)
python -I src/features.py data/verified_evals.json repos out/features_verified.csv   # ~5 min
python -I src/localize.py data/verified_evals.json repos out/localize_verified.csv   # ~6 min
python -I src/gemma_tasks.py                               # rebuild the 55 tasks with public outcomes from git
python -I src/gemma_tasks.py --all                         # rebuild all 129 tasks of my own Gemma 4 31B runs
python src/paper_numbers.py --boot 1000                    # -> out/numbers.json (~30 min on 4 CPU cores; also runs tenmin_numbers.py)
python src/figures.py                                      # -> paper/fig*.png
python src/fill_paper.py                                   # paper/paper_src.md + numbers.json -> paper/paper.md
```

Every number in the paper is a placeholder in `paper/paper_src.md` filled from `numbers.json` by `fill_paper.py`, so the
text cannot drift from the analysis. Seeds are fixed; a clean run reproduces `data/numbers.json`.

## Files

| Path | What it is |
|---|---|
| `src/codegraph.py` | Static call graph of a Python repo at any git commit, read from a bare clone, cached by blob SHA |
| `src/features.py` | Fix-location features for each SWE-bench Verified task |
| `src/localize.py` | CPU BM25 localizer + structure of its predicted fix site |
| `src/diffspans.py` | Maps a diff onto the definitions it touches (100% file agreement, 91% precision / 83% recall at function level vs. curated SWE-bench spans) |
| `src/gemma_tasks.py` | Rebuilds public competition tasks (`<repo>_<PR>`) from git history and joins public Gemma 4 31B outcomes |
| `src/paper_numbers.py` | Every number in the paper, with bootstrap CIs, within-release refits, the checkpoint simulation and the decision-time (landmark) trace analysis |
| `src/fill_paper.py` | Fills the paper's numeric placeholders from `numbers.json` |
| `data/features_verified.csv` | **Release:** fix-location features for all 500 SWE-bench Verified tasks |
| `data/localize_verified.csv` | **Release:** BM25 localizer output and hit metrics for all 500 tasks |
| `data/features_gemma_public55.csv` | **Release:** fix-location features for 55 public competition tasks |
| `data/features_gemma_own129.csv` | **Release:** fix-location features for all 129 public competition tasks |
| `data/gemma/own/gemma31b_task_results_compact.csv` | **Release:** my per-task Gemma 4 31B outcomes (resolved, tool calls, seconds, failure type) |
| `kaggle/wfl-gemma31b-129-tasks.ipynb` | The Kaggle notebook that produced those outcomes (4xL4, internet off) |
| `kaggle/wfl-gemma31b-checkpoint-arm.ipynb` | The same notebook plus one cell adding the edit-plan checkpoint to the prompt |
| `data/gemma/own/gemma31b_checkpoint_results_compact.csv` | **Release:** per-task outcomes of the checkpoint run |
| `kaggle/wfl-gemma31b-tenminute-arm.ipynb` | The same notebook with 10 minutes per task instead of 6, on the 102 solvable tasks |
| `data/gemma/own/gemma31b_tenmin_results_compact.csv` | **Release:** per-task outcomes of the ten-minute run |
| `data/gemma/own/trace_features_tenmin.csv` | **Release:** first source edit and duration per task from the ten-minute run's traces |
| `src/trace_rules.py` | The rule that decides which tool call is a source edit (reproduces `trace_features.csv` exactly) |
| `src/tenmin_numbers.py` | Numbers for the ten-minute run (called by `paper_numbers.py`) |
| `data/gemma/own/trace_features.csv` | **Release:** 30 behaviour features per task-run from all 258 agent traces (when the agent first opened a fixed file, first source edit, stated plan, tool mix, prompt size) |
| `data/numbers.json` | All paper numbers |
| `paper/` | Paper (Markdown + PDF) and figures |

## Data sources (fetched, not redistributed)

- SWE-bench Verified tasks and curated fix spans (via moatless-tools).
- Per-task outcomes of mini-SWE-agent runs from the public SWE-bench leaderboard.
- IRT difficulties and LLM-judged features from *Agent Psychometrics* (Ge et al., 2026), the comparison baseline.
- Per-task Gemma 4 31B outcomes on public competition tasks by yosh95 (Apache-2.0).

## License

Apache License 2.0.
