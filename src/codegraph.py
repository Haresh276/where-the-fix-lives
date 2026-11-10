"""Static call-graph construction for a Python repository at a given git commit.

Design goals: zero GPU, zero network at analysis time, deterministic.
- Files are read straight from a bare git repo (`git cat-file --batch`), so any
  commit can be analysed without a checkout.
- Per-file parse results are cached by blob SHA: files unchanged between two
  commits are parsed once.
- Call resolution is name-based (an over-approximation, like most static
  Python call graphs): a call to `foo` or `x.foo` links to every definition
  whose short name is `foo`, with preference rules (same class for self.foo(),
  explicit `from m import foo`). Each candidate edge carries weight 1/k where
  k is the number of candidates, so ambiguity is visible to downstream code.

Node id format:  "<path/to/file.py>::<Qual.Name>"  (Qual.Name may be a function,
a method "Class.method", a nested "outer.inner", or a class "Class").
"""
from __future__ import annotations

import ast
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class DefInfo:
    qual: str              # Class.method / func / Class
    kind: str              # 'func' | 'class'
    lineno: int
    end_lineno: int
    cls: str | None        # enclosing class short name for methods
    calls: list = field(default_factory=list)   # list of (callee_short_name, via_self:bool)


@dataclass
class ModuleInfo:
    defs: list             # list[DefInfo]
    imports: dict          # local alias -> imported short name (for `from x import y as z`)
    ok: bool = True


def _call_name(node: ast.Call):
    f = node.func
    if isinstance(f, ast.Name):
        return f.id, False
    if isinstance(f, ast.Attribute):
        via_self = isinstance(f.value, ast.Name) and f.value.id in ("self", "cls")
        return f.attr, via_self
    return None, False


class _Visitor(ast.NodeVisitor):
    def __init__(self):
        self.defs: list[DefInfo] = []
        self.stack: list[tuple[str, str]] = []   # (name, kind)
        self.cur: list[DefInfo] = []
        self.imports: dict[str, str] = {}

    def _qual(self, name):
        return ".".join([n for n, _ in self.stack] + [name])

    def _cls(self):
        for n, k in reversed(self.stack):
            if k == "class":
                return n
        return None

    def visit_ImportFrom(self, node):
        for a in node.names:
            self.imports[a.asname or a.name] = a.name
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        d = DefInfo(self._qual(node.name), "class", node.lineno,
                    getattr(node, "end_lineno", node.lineno), None)
        self.defs.append(d)
        self.stack.append((node.name, "class"))
        self.cur.append(d)
        self.generic_visit(node)
        self.cur.pop()
        self.stack.pop()

    def _func(self, node):
        d = DefInfo(self._qual(node.name), "func", node.lineno,
                    getattr(node, "end_lineno", node.lineno), self._cls())
        self.defs.append(d)
        self.stack.append((node.name, "func"))
        self.cur.append(d)
        self.generic_visit(node)
        self.cur.pop()
        self.stack.pop()

    visit_FunctionDef = _func
    visit_AsyncFunctionDef = _func

    def visit_Call(self, node):
        name, via_self = _call_name(node)
        if name and self.cur:
            self.cur[-1].calls.append((name, via_self))
        self.generic_visit(node)


def parse_module(src: bytes) -> ModuleInfo:
    try:
        tree = ast.parse(src)
    except Exception:
        try:  # py2-era files: try dropping undecodable bytes
            tree = ast.parse(src.decode("utf-8", "ignore"))
        except Exception:
            return ModuleInfo([], {}, ok=False)
    v = _Visitor()
    v.visit(tree)
    return ModuleInfo(v.defs, v.imports)


_BLOB_CACHE: dict[str, ModuleInfo] = {}


class GitRepo:
    def __init__(self, path):
        self.path = path
        self._cat = subprocess.Popen(["git", "-C", path, "cat-file", "--batch"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def ls_py(self, commit):
        out = subprocess.run(["git", "-C", self.path, "ls-tree", "-r", commit],
                             capture_output=True, check=True).stdout.decode()
        res = []
        for line in out.splitlines():
            meta, p = line.split("\t", 1)
            _, typ, sha = meta.split()
            if typ == "blob" and p.endswith(".py"):
                res.append((p, sha))
        return res

    def blob(self, sha) -> bytes:
        self._cat.stdin.write((sha + "\n").encode())
        self._cat.stdin.flush()
        header = self._cat.stdout.readline().split()
        size = int(header[2])
        data = self._cat.stdout.read(size)
        self._cat.stdout.read(1)
        return data

    def close(self):
        self._cat.stdin.close()
        self._cat.wait()


def is_test_path(p: str) -> bool:
    parts = p.split("/")
    fn = parts[-1]
    return ("tests" in parts or "test" in parts or "testing" in parts
            or fn.startswith("test_") or fn.endswith("_test.py") or fn == "conftest.py")


class CodeGraph:
    """Nodes, containment and call edges for one commit."""

    def __init__(self, repo: GitRepo, commit: str, include_tests=False):
        self.nodes: dict[str, dict] = {}
        self.by_short: dict[str, list[str]] = defaultdict(list)
        self.by_file: dict[str, list[str]] = defaultdict(list)
        self.calls: dict[str, dict[str, float]] = defaultdict(dict)   # caller -> {callee: w}
        self.rcalls: dict[str, dict[str, float]] = defaultdict(dict)
        self.contain: dict[str, set] = defaultdict(set)               # undirected parent/child
        files = repo.ls_py(commit)
        self.n_files = 0
        raw = []
        for p, sha in files:
            if not include_tests and is_test_path(p):
                continue
            mi = _BLOB_CACHE.get(sha)
            if mi is None:
                mi = parse_module(repo.blob(sha))
                _BLOB_CACHE[sha] = mi
            self.n_files += 1
            for d in mi.defs:
                nid = f"{p}::{d.qual}"
                self.nodes[nid] = dict(file=p, qual=d.qual, kind=d.kind, cls=d.cls,
                                       lines=d.end_lineno - d.lineno + 1,
                                       lineno=d.lineno, end=d.end_lineno)
                short = d.qual.rsplit(".", 1)[-1]
                self.by_short[short].append(nid)
                self.by_file[p].append(nid)
                if "." in d.qual:
                    parent = f"{p}::{d.qual.rsplit('.', 1)[0]}"
                    self.contain[nid].add(parent)
                    self.contain[parent].add(nid)
                raw.append((nid, p, d, mi.imports))
        # resolve calls
        for nid, p, d, imports in raw:
            for name, via_self in d.calls:
                name = imports.get(name, name)
                cands = self.by_short.get(name)
                if not cands:
                    continue   # library / builtin call
                if via_self and d.cls:
                    same = [c for c in cands if self.nodes[c]["cls"] == d.cls]
                    if same:
                        cands = same
                same_file = [c for c in cands if self.nodes[c]["file"] == p]
                if not via_self and same_file and len(cands) > 1:
                    cands = same_file
                # a call to a class name means its constructor
                w = 1.0 / len(cands)
                for c in cands:
                    if c == nid:
                        continue
                    self.calls[nid][c] = self.calls[nid].get(c, 0) + w
                    self.rcalls[c][nid] = self.rcalls[c].get(nid, 0) + w

    # ------------------------------------------------------------------ queries
    def neighbors(self, nid, edges="all"):
        """Undirected neighbourhood: calls and called-by, plus containment unless edges == "call"."""
        out = set(self.calls.get(nid, ())) | set(self.rcalls.get(nid, ()))
        if edges != "call":
            out |= self.contain.get(nid, set())
        return out

    def bfs(self, sources, max_depth=8, edges="all"):
        dist = {s: 0 for s in sources if s in self.nodes}
        frontier = list(dist)
        d = 0
        while frontier and d < max_depth:
            d += 1
            nxt = []
            for u in frontier:
                for v in self.neighbors(u, edges):
                    if v not in dist:
                        dist[v] = d
                        nxt.append(v)
            frontier = nxt
        return dist

    def resolve_span(self, file, span):
        """Map an expected_spans entry ('Class.method' or 'func') to a node id."""
        nid = f"{file}::{span}"
        if nid in self.nodes:
            return nid
        # spans sometimes give only the method name or extra nesting
        tail = span.split(".")[-1]
        cands = [n for n in self.by_file.get(file, []) if self.nodes[n]["qual"].endswith(tail)]
        return cands[0] if len(cands) == 1 else None
