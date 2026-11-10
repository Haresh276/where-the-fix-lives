"""Map a unified diff onto the definitions (functions/classes) it touches at the base commit.

Used for tasks where no curated fix-location labels exist (the competition's public PRs).
Validated against SWE-bench Verified's curated `expected_spans` (see paper, Sec. 2).
"""
import re
import subprocess

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
NON_SOURCE = re.compile(r"(^|/)(tests?|testing|docs?|docs_src|examples?|scripts|benchmarks?)/|"
                        r"(^|/)test_[^/]*\.py$|_test\.py$|conftest\.py$")


def is_source(path):
    return path.endswith(".py") and not NON_SOURCE.search(path)


def changed_old_lines(patch_text):
    """{file: set(old line numbers touched)} from a unified diff (any context size)."""
    out, cur, old = {}, None, 0
    for line in patch_text.splitlines():
        if line.startswith("--- "):
            continue
        if line.startswith("+++ "):
            p = line[4:].strip()
            cur = p[2:] if p.startswith("b/") else (None if p == "/dev/null" else p)
            if cur:
                out.setdefault(cur, set())
            continue
        m = HUNK.match(line)
        if m:
            old = int(m.group(1))
            continue
        if cur is None:
            continue
        if line.startswith("-"):
            out[cur].add(old)
            old += 1
        elif line.startswith("+"):
            out[cur].add(max(old - 1, 1))      # insertion point: line before
        elif line.startswith(" "):
            old += 1
    return out


def spans_from_patch(g, patch_text):
    """{file: [qualname, ...]} innermost definitions containing changed lines (source files only)."""
    res = {}
    for f, lines in changed_old_lines(patch_text).items():
        if not is_source(f):
            continue
        nodes = [g.nodes[n] for n in g.by_file.get(f, [])]
        quals = set()
        for ln in lines:
            inside = [nd for nd in nodes if nd["lineno"] <= ln <= nd["end"]]
            if inside:
                quals.add(min(inside, key=lambda nd: nd["end"] - nd["lineno"])["qual"])
        res[f] = sorted(quals)
    return res


def git_diff(repo_path, base, sha):
    return subprocess.run(["git", "-C", repo_path, "diff", "-U0", base, sha],
                          capture_output=True, text=True, errors="ignore").stdout
