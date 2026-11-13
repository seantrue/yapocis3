---
name: test-affected
description: Run only the tests a change can affect, in tiers, across the Rust workspaces and Python packages (plumcraft, plumimpress, yapocis-rs, Yapocis3, yapocis-external, impressionist-rs, Virtual-Impressionist). Use after editing code and before saying a change works, before committing, or when asked to "run the tests". Do not run a repo's full suite unless asked for tier 2.
---

# test-affected

Full suites here take hours; most edits need seconds to minutes. `affected.py` maps
the diff to the tests that can see it and prints (or runs) one command per suite.

```sh
python3 .claude/skills/test-affected/affected.py --tier 0          # print
python3 .claude/skills/test-affected/affected.py --tier 0 --run    # run, stop at first failing suite
python3 .claude/skills/test-affected/affected.py --base HEAD~3 ... # diff base (default origin/main)
```

Per-repo rules are in `.test-affected.toml` at the repo root; nextest profiles (`quick`) in
`.config/nextest.toml`. `affected.py` and this file are the same in plumcraft, plumimpress,
yapocis-rs, Yapocis3, yapocis-external, impressionist-rs and Virtual-Impressionist: change them
in all seven.

## Tiers

| Tier | When | What |
|---|---|---|
| 0 | after each logical edit | affected crates/tests, one hash algorithm (md5), one backend/device, fail fast |
| 1 | before a commit | affected crates/tests, this host's backends, no fail-fast |
| 2 | only when asked, or a global file changed | the repo's documented gate (AGENTS.md / CLAUDE.md) |

Run tier 0 yourself without asking. Run tier 1 before proposing a commit. Start tier 2
in the background (`run_in_background`) and keep working; never block a turn on it.

## Reading results

- Report failures only: test id, the assertion, and the first relevant frame. Not the log.
- nextest: `--message-format libtest-json` or `--failure-output final` keeps output short.
  pytest: `-q -rfE`.
- A failure in a crate you did not touch is still yours if it is in the printed closure:
  that is why it was selected.
- "nothing to test" for a docs-only diff is correct; say so, don't run tier 2 to be safe.

## Rules that are not in the script

- **Never fix a test by changing backend, device, tolerance or hash algorithm.** (All three
  repos' CLAUDE.md.) Tier 0 runs one backend for speed; a failure there is a failure.
- **Hash algorithms:** tier 0/1 use md5 only. If the diff touches hashing, the store, or
  anything `commonkit::digest` / `commonkit.digest` reaches, run tier 1 again with
  `COMMONKIT_HASH=b3` (and a fresh `COMMONKIT_TEST_RUN` / `COMMONKIT_STORE_ROOT`, as CI does).
- **Yapocis3 → yapocis-rs:** a change under `src/yapocis/kernels/` or `src/yapocis/interfaces/`
  changes what yapocis-rs vendors. Say so; regenerating the bundle there is a separate step
  (`yapocis-tools bundle`, update `bundle/SOURCE`), and its tier 0 is then `-p yapocis` with
  `rustcpu`. Kernel changes also need the matching port in `yapocis-rt/src/cpu/kernels/`.
- **yapocis-external → impressionist-rs:** the same chain one level up. A kernel or interface
  change in yapocis-external is re-exported into impressionist-rs's
  `crates/yapocis-external/bundle` (`yapocis-tools bundle --project-only`, update `bundle/SOURCE`)
  and ported in `crates/yapocis-external/src/cpu/`; then run impressionist-rs's tier 0 there.
  Virtual-Impressionist reads yapocis-external through `YAPOCIS_PROJECT`, so run its tests that
  name the kernel too.
- **yapocis-rs `bindings/python`** is outside the workspace: for a change there, run
  `cargo clippy --manifest-path bindings/python/Cargo.toml -- -D warnings`, then Yapocis3's
  tier 0 with `YAPOCIS_T0_BACKEND=rustcpu` after `maturin develop`.
- **GPU tests on a shared host:** pin cards by name (`YAPOCIS_T0_DEVICE=3080`), never index.
- **plumcraft UI:** kittests are in `plumcraft-ui` and `plumcraft-editor`; rendering ones need
  a Vulkan adapter (lavapipe on Linux CI). On a machine without one, say they were not run.

## Host notes

Set once per machine (shell profile), not per command:

- Mac: `YAPOCIS_T0_BACKEND=metal`
- Linux GPU host: `YAPOCIS_T0_BACKEND=ocl YAPOCIS_T0_DEVICE=3080`
- CI / no GPU: unset (numbax for Yapocis3; rustcpu for yapocis-rs)
