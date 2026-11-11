# Where the Fix Lives

Code, data and paper for **"Where the Fix Lives: Code-Graph Structure and the Triage Gap for Local Coding Agents"**
by Haresh G. (Kaggle: Haresh276), Gemma 4 Developer Agent Paper Track, 2026.

**Question.** Which properties of a bug's fix location (whether the issue names anything connected to it in the code
graph, how far it is, how many files and functions it touches) decide whether a coding agent fixes it, and when could
an agent know?

**Short answer.** An issue that names nothing connected to the fix, and a fix that spans several files, predict
failure for 38 models in the mini-SWE-agent scaffold family (refits within single releases agree). Multi-file fixes cut
weak models to 7.4% resolved. This cannot be read from the issue (AUC 0.45–0.54 before search); knowing where the true
fix sits adds little over a BM25 hit; knowing how many files/functions the edit spans gives 0.63–0.72. A simulated
edit-plan checkpoint on public logs loses about half as many solves as random capping. My own run of Gemma 4 31B on all
129 public competition tasks (Kaggle, 4xL4): 36% resolved for one-function fixes vs 15% for multi-function fixes, and 42%
vs 11% split at a 10-line patch (p < 0.001), the same drop as the weak SWE-bench models.
A second full run with the edit-plan checkpoint added to the prompt solved 32 vs 29 of 102 solvable tasks
(4 gained, 1 lost; McNemar p = 0.375) with 46 instead of 56 timeouts: the right direction, not yet significant.

The analysis runs on CPU with no API keys and no competition data. Only the Gemma run itself needs the Kaggle GPU notebook.

## Reproduce every number in the paper

```bash
pip install -r requirements.txt
python src/fetch_data.py                                   # public inputs + bare git clones (~2 GB)
python -I src/features.py data/verified_evals.json repos out/features_verified.csv   # ~5 min
python -I src/localize.py data/verified_evals.json repos out/localize_verified.csv   # ~6 min
python -I src/gemma_tasks.py                               # rebuild the 55 tasks with public outcomes from git
python -I src/gemma_tasks.py --all                         # rebuild all 129 tasks of my own Gemma 4 31B run
python src/paper_numbers.py --boot 1000                    # -> out/numbers.json (~30 min on 4 CPU cores)
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
| `src/paper_numbers.py` | Every number in the paper, with bootstrap CIs, within-release refits and the checkpoint simulation |
| `src/fill_paper.py` | Fills the paper's numeric placeholders from `numbers.json` |
| `data/features_verified.csv` | **Release:** fix-location features for all 500 SWE-bench Verified tasks |
| `data/localize_verified.csv` | **Release:** BM25 localizer output and hit metrics for all 500 tasks |
| `data/features_gemma_public55.csv` | **Release:** fix-location features for 55 public competition tasks |
| `data/features_gemma_own129.csv` | **Release:** fix-location features for all 129 public competition tasks |
| `data/gemma/own/gemma31b_task_results_compact.csv` | **Release:** my per-task Gemma 4 31B outcomes (resolved, tool calls, seconds, failure type) |
| `kaggle/wfl-gemma31b-129-tasks.ipynb` | The Kaggle notebook that produced those outcomes (4xL4, internet off) |
| `kaggle/wfl-gemma31b-checkpoint-arm.ipynb` | The same notebook plus one cell adding the edit-plan checkpoint to the prompt |
| `data/gemma/own/gemma31b_checkpoint_results_compact.csv` | **Release:** per-task outcomes of the checkpoint run |
| `data/numbers.json` | All paper numbers |
| `paper/` | Paper (Markdown + PDF) and figures |

## Data sources (fetched, not redistributed)

- SWE-bench Verified tasks and curated fix spans (via moatless-tools).
- Per-task outcomes of mini-SWE-agent runs from the public SWE-bench leaderboard.
- IRT difficulties and LLM-judged features from *Agent Psychometrics* (Ge et al., 2026), the comparison baseline.
- Per-task Gemma 4 31B outcomes on public competition tasks by yosh95 (Apache-2.0).

## License

Apache License 2.0.
