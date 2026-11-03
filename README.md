# Where the Fix Lives

Code and data pipeline for the Kaggle **Gemma 4 Developer Agent Paper Track** writeup
*"Where the Fix Lives: Code-Graph Structure Predicts Small-Agent Failure"* by Haresh G. (Kaggle: Haresh276).

**Question:** which structural properties of a bug's fix location — call-graph distance from what the
issue mentions, number of files touched, whether the fix is lexically findable — decide whether a
small local coding agent (32B class, like Gemma 4 31B) can fix it?

Everything runs on CPU. No GPU, no API keys.

## Status
Work in progress — full results, figures and reproduction steps will be added before the
paper deadline (12 Nov 2026).

## Layout
| File | What it does |
|---|---|
| `src/codegraph.py` | Builds a static call graph of a Python repo at any git commit (reads a bare clone, caches parses by blob SHA) |
| `src/features.py` | Structural features of each task's fix location (issue-to-fix hops, ambiguity, fan-in, spread, …) |
| `src/localize.py` | CPU BM25 localizer + features of the *predicted* fix location |
| `src/analyze.py` | Correlations, difficulty prediction, per-agent and pooled interaction models |

## Data sources (not redistributed)
- SWE-bench Verified task records (base commits, gold patches, gold fix spans)
- Per-task outcomes of 40 mini-SWE-agent runs from the public SWE-bench leaderboard
- IRT difficulties and LLM-judge features from *Agent Psychometrics* (Ge, Kryvosheieva et al., 2026), used as the comparison baseline

## License
Apache License 2.0.
