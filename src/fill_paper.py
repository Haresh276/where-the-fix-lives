"""Fill {{...}} placeholders in paper/paper_src.md from out/numbers.json -> paper/paper.md.

{{v:a.b.c}}      raw value            {{pct:a.b}}   0.412 -> 41%      {{pct1:a.b}} 0.074 -> 7.4%
{{ci:a.b}}       [est, lo, hi]        -> −0.32 [−0.50, −0.21]
{{cim:a.b}}      {main, main_ci}      -> −0.28 [−0.46, −0.11]
{{auc:a.b}}      {auc, ci}            -> 0.64 [0.60, 0.71]
{{rng:path|*|leaf}}  min–max over the '*' level, as 2-dp numbers; add '|pct' for percentages
Every placeholder must resolve; a missing key stops the build.
"""
import json
import re
import sys

N = json.load(open("out/numbers.json"))


def get(path, root=N):
    """Dotted path; keys may themselves contain dots (e.g. 'v2.0.0', 'Qwen2.5-Coder'), so match greedily."""
    cur, parts = root, path.split(".")
    while parts:
        if isinstance(cur, list):
            cur = cur[int(parts.pop(0))]
            continue
        for j in range(len(parts), 0, -1):
            k = ".".join(parts[:j])
            if k in cur:
                cur, parts = cur[k], parts[j:]
                break
        else:
            raise KeyError(path)
    return cur


def num(x, d=2):
    s = f"{x:.{d}f}"
    return s.replace("-", "−")


def rng(spec):
    parts = spec.split("|")
    head, tail = parts[0], parts[2]
    pct = len(parts) > 3 and parts[3] == "pct"
    vals = [get(tail, v) for v in get(head).values() if isinstance(v, dict)]
    lo, hi = min(vals), max(vals)
    if pct:
        return f"{lo*100:.0f}–{hi*100:.0f}%"
    return f"{num(lo)}–{num(hi)}"


def sub(m):
    kind, path = m.group(1), m.group(2)
    if kind == "rng":
        return rng(path)
    x = get(path)
    if kind == "v":
        return str(x)
    if kind == "int":
        return f"{int(x):,}"
    if kind == "f2":
        return num(x)
    if kind == "pct":
        return f"{x*100:.0f}%"
    if kind == "pct1":
        return f"{x*100:.1f}%"
    if kind == "ci":
        return f"{num(x[0])} [{num(x[1])}, {num(x[2])}]"
    if kind == "cim":
        return f"{num(x['main'])} [{num(x['main_ci'][0])}, {num(x['main_ci'][1])}]"
    if kind == "auc":
        return f"{num(x['auc'])} [{num(x['ci'][0])}, {num(x['ci'][1])}]"
    if kind == "p":
        return f"{x:.3f}".rstrip("0") if x >= 0.001 else "<0.001"
    raise KeyError(kind)


src = open("paper/paper_src.md").read()
out = re.sub(r"\{\{(\w+):([^}]+)\}\}", sub, src).replace("p = <0.001", "p < 0.001")
left = re.findall(r"\{\{[^}]*\}\}", out)
if left:
    sys.exit(f"unresolved: {left}")
open("paper/paper.md", "w").write(out)
body = out.split("## References")[0]
print("words (body, excl. references):", len(re.sub(r"[#*|`\-]", " ", body).split()))
