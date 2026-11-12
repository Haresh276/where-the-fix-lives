"""Which tool call in an ATIF trace counts as a source edit.

An edit_file or write_file call on a .py file of the repository, excluding test files and the scratch scripts Gemma
writes to reproduce or check a bug (repro*.py, reproduce*.py, verify*.py, check*.py, test*.py, ...).
This rule reproduces the first-edit times in data/gemma/own/trace_features.csv on all 129 traces of the first run.
"""
import re

SCRATCH = re.compile(r"^(repro|reproduce|reproduction|test|verify|check|debug|tmp|scratch)", re.I)


def is_source_edit(function_name, path):
    if function_name not in ("edit_file", "write_file") or not path:
        return False
    p = path.strip()
    if p.startswith("./"):
        p = p[2:]
    if not p.endswith(".py"):
        return False
    base = p.split("/")[-1]
    if SCRATCH.match(base) or re.search(r"repro", base, re.I):
        return False
    if re.search(r"(^|/)tests?/", p):
        return False
    return True


def trace_features(steps):
    """steps: the ATIF 'steps' list. Returns (duration_s, first_source_edit_s or -1)."""
    ts = [s.get("extra", {}).get("elapsed_s") for s in steps]
    ts = [t for t in ts if t is not None]
    edits = []
    for s in steps:
        for c in s.get("tool_calls") or []:
            a = c.get("arguments") or {}
            if is_source_edit(c.get("function_name"), a.get("filepath") or a.get("path") or a.get("file_path")):
                t = (c.get("extra") or {}).get("elapsed_s")
                if t is not None:
                    edits.append(round(t))
    return (round(max(ts)) if ts else -1), (min(edits) if edits else -1)
