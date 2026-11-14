#!/usr/bin/env python3
"""Pick the tests a change can affect, for a Cargo workspace, Python roots, or both.

    affected.py [--base REF] [--tier 0|1|2] [--run] [--repo DIR]

Changed files = `git diff BASE...HEAD` + staged + unstaged + untracked.

Rust: each changed file maps to the workspace package whose directory holds it
(deepest match). The selection is those packages plus every workspace package that
depends on them, normal or dev, transitively. A change to a "global" file (root
Cargo.toml, Cargo.lock, toolchain, nextest config, ...) selects the whole workspace.

Python: per `[[python]]` root in the config. Changed files under a root select that
root; `[[python.map]]` rules turn non-Python changes (templates, fixtures) into test
paths. With pytest-testmon installed and `testmon = true`, tier 0 lets testmon pick.

Config: `.test-affected.toml` at the repo root (optional; defaults cover a plain
workspace). Tiers: 0 = affected, fast flags; 1 = affected, full flags; 2 = everything.
Prints one shell command per suite; `--run` runs them and stops at the first failure.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

RUST_GLOBAL = ["Cargo.toml", "Cargo.lock", "rust-toolchain*", ".cargo/*", ".config/nextest.toml",
               "clippy.toml", "rustfmt.toml", ".test-affected.toml"]


def sh(args: list[str], cwd: Path) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout


def changed_files(repo: Path, base: str) -> list[str]:
    out: set[str] = set()
    try:
        out |= set(sh(["git", "diff", "--name-only", f"{base}...HEAD"], repo).split())
    except subprocess.CalledProcessError:
        # shallow clone without the base: fall back to the working tree only
        print(f"# warning: cannot diff against {base}; using working-tree changes only", file=sys.stderr)
    out |= set(sh(["git", "diff", "--name-only", "HEAD"], repo).split())
    out |= set(sh(["git", "ls-files", "--others", "--exclude-standard"], repo).split())
    return sorted(out)


def matches(path: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, g) for g in globs)


# ---------------------------------------------------------------- Rust

@dataclass
class Workspace:
    root: Path
    dirs: dict[str, Path]               # package -> manifest dir, relative to repo
    rdeps: dict[str, set[str]]          # package -> workspace packages that depend on it

    @classmethod
    def load(cls, repo: Path) -> Workspace:
        meta = json.loads(sh(["cargo", "metadata", "--format-version", "1", "--no-deps"], repo))
        pkgs = meta["packages"]
        names = {p["name"] for p in pkgs}
        dirs = {p["name"]: Path(p["manifest_path"]).parent.relative_to(repo.resolve()) for p in pkgs}
        rdeps: dict[str, set[str]] = {n: set() for n in names}
        for p in pkgs:
            for d in p["dependencies"]:          # kind None (normal), "dev", "build": all matter to tests
                if d["name"] in names:
                    rdeps[d["name"]].add(p["name"])
        return cls(repo, dirs, rdeps)

    def owner(self, path: str) -> str | None:
        best, depth = None, -1
        for name, d in self.dirs.items():
            s = str(d)
            if s == "." or path == s or path.startswith(s + "/"):
                n = 0 if s == "." else s.count("/") + 1
                if n > depth:
                    best, depth = name, n
        return best

    def closure(self, seeds: set[str]) -> set[str]:
        out, todo = set(seeds), list(seeds)
        while todo:
            for r in self.rdeps[todo.pop()]:
                if r not in out:
                    out.add(r)
                    todo.append(r)
        return out


def rust_command(repo: Path, cfg: dict, files: list[str], tier: int) -> tuple[list[str], str] | None:
    if not (repo / "Cargo.toml").exists():
        return None
    ws = Workspace.load(repo)
    rcfg = dict(cfg.get("rust", {}))
    # [rust.platform.darwin] / [rust.platform.linux]: per-host overrides (GPU features differ)
    for k, v in rcfg.pop("platform", {}).get(sys.platform, {}).items():
        rcfg[k] = {**rcfg.get(k, {}), **v} if isinstance(v, dict) else v
    globals_ = RUST_GLOBAL + rcfg.get("global", [])
    ignore = rcfg.get("ignore", ["*.md", "docs/*", "packaging/*", ".github/*"])
    if tier >= 2 or any(matches(f, globals_) for f in files):
        pkgs, why = None, "everything" if tier >= 2 else "global file changed"
    else:
        seeds = set()
        for f in files:
            # [[rust.map]]: files outside any crate that some crate's tests read
            for rule in rcfg.get("map", []):
                if fnmatch.fnmatch(f, rule["glob"]):
                    seeds |= set(rule["packages"])
            if matches(f, ignore):
                continue
            if (o := ws.owner(f)) is not None:
                seeds.add(o)
        if not seeds:
            return None
        pkgs = sorted(ws.closure(seeds))
        why = f"changed {sorted(seeds)} -> {len(pkgs)}/{len(ws.dirs)} packages"
    tiercfg = rcfg.get(f"tier{min(tier, 2)}", {})
    cmd = shlex.split(rcfg.get("runner", "cargo nextest run"))
    cmd += shlex.split(tiercfg.get("args", rcfg.get("args", "")))
    if pkgs is None:
        cmd.append("--workspace")
    else:
        for p in pkgs:
            cmd += ["-p", p]
    # features as "pkg/feat": keep only those whose package is selected, else cargo refuses
    feats = [f for f in tiercfg.get("features", rcfg.get("features", []))
             if pkgs is None or "/" not in f or f.split("/", 1)[0] in pkgs]
    if feats:
        cmd += ["--features", ",".join(feats)]
    cmd += shlex.split(tiercfg.get("after", ""))
    env = {**rcfg.get("env", {}), **tiercfg.get("env", {})}
    # env values and `pre` are shell, unquoted: "$run/store" and "$(mktemp -d)" work
    prefix = " ".join(f"{k}={v}" for k, v in env.items())
    pre = rcfg.get("pre", "")
    return cmd, f"# rust: {why}\n" + (pre + "; " if pre else "") + (prefix + " " if prefix else "") + shlex.join(cmd)


# ---------------------------------------------------------------- Python

@dataclass
class PyRoot:
    root: str
    cmd: str = "uv run pytest"
    tests: str = "tests"
    src: list[str] = field(default_factory=lambda: ["*.py", "pyproject.toml", "uv.lock"])
    testmon: bool = False
    env: dict = field(default_factory=dict)
    map: list[dict] = field(default_factory=list)
    tier0: str = "-x -q -m 'not slow and not performance'"
    tier1: str = "-q -m 'not performance'"
    tier2: str = "-q"
    pre: str = ""                                  # shell run first, e.g. run=$(mktemp -d)
    tier_env: dict = field(default_factory=dict)   # {"0": {...}, "1": {...}}: per-tier env, unquoted (${VAR:-x} works)


def stem_tests(tests_dir: Path, stems: set[str]) -> set[str]:
    out = set()
    pats = [re.compile(rf"\b{re.escape(s)}\b") for s in stems]
    for p in tests_dir.rglob("test_*.py"):
        text = p.read_text(errors="replace")
        if any(x.search(text) for x in pats):
            out.add(str(p.relative_to(tests_dir)))
    return out


def python_commands(repo: Path, cfg: dict, files: list[str], tier: int) -> list[str]:
    out = []
    for raw in cfg.get("python", []):
        r = PyRoot(**raw)
        pre = "" if r.root == "." else r.root.rstrip("/") + "/"
        local = [f[len(pre):] for f in files if f.startswith(pre)]
        targets: set[str] = set()
        rule_env: dict[str, str] = {}
        for rule in r.map:                         # repo-relative inputs -> test paths (root-relative)
            hits = [f for f in files if fnmatch.fnmatch(f, rule["glob"])]
            if not hits:
                continue
            targets |= set(rule.get("tests", []))
            # env = {VAR = "{seg3}"}: a value from the matched path (e.g. the backend directory)
            for k, v in rule.get("env", {}).items():
                segs = hits[0].split("/")
                rule_env[k] = re.sub(r"\{seg(\d+)\}", lambda m, segs=segs: segs[int(m.group(1))], v)
            if rule.get("grep_stem"):
                # test files under `tests` that name the changed file's stem (an interface,
                # a kernel template, a library module); none found -> the rule's fallback
                found = stem_tests(repo / r.root / r.tests, {Path(h).name.split(".")[0] for h in hits})
                targets |= {str(Path(r.tests) / p) for p in found} or set(rule.get("fallback", [r.tests]))
        py_changed = [f for f in local if matches(f, r.src) and not any(fnmatch.fnmatch(pre + f, m["glob"]) for m in r.map)]
        test_changed = [f for f in py_changed if f.startswith(r.tests + "/") and Path(f).name.startswith("test_")]
        if tier < 2 and not py_changed and not targets:
            continue
        flags = {0: r.tier0, 1: r.tier1}.get(tier, r.tier2)
        if tier >= 2:
            sel, why = [r.tests], "everything"
        elif r.testmon and tier == 0:
            sel, why = ["--testmon", r.tests], "testmon"
        elif py_changed and len(test_changed) < len(py_changed):
            # library code changed: without a dependency map, the honest answer is the root's suite
            sel, why = [r.tests], f"{len(py_changed)} source files changed"
        else:
            sel = sorted(targets | set(test_changed))
            if r.tests in sel:
                sel = [r.tests]
            why = "changed tests / mapped inputs"
        envd = {**r.env, **r.tier_env.get(str(min(tier, 2)), {})}
        env = " ".join(f"{k}={v}" for k, v in envd.items())
        cd = f"cd {shlex.quote(r.root)} && " if r.root != "." else ""
        exp = "".join(f"export {k}={shlex.quote(v)}; " for k, v in rule_env.items())
        pre = r.pre + "; " if r.pre else ""
        out.append(f"# python[{r.root}]: {why}\n{cd}{pre}{exp}{env + ' ' if env else ''}{r.cmd} {flags} {shlex.join(sel)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--base", default=os.environ.get("TEST_AFFECTED_BASE", "origin/main"))
    ap.add_argument("--tier", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    repo = Path(sh(["git", "rev-parse", "--show-toplevel"], a.repo).strip())
    cfgp = repo / ".test-affected.toml"
    cfg = tomllib.loads(cfgp.read_text()) if cfgp.exists() else {}
    files = changed_files(repo, a.base)
    print(f"# {len(files)} changed files vs {a.base}, tier {a.tier}")
    cmds = []
    rc = rust_command(repo, cfg, files, a.tier)
    if rc:
        cmds.append(rc[1])
    cmds += python_commands(repo, cfg, files, a.tier)
    if not cmds:
        print("# nothing to test")
        return 0
    for c in cmds:
        print(c)
        if a.run:
            body = c.split("\n", 1)[1]
            r = subprocess.run(["bash", "-c", body], cwd=repo)
            if r.returncode:
                return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
