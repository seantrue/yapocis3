# Features

Features requested in passing (`feature: …`), kept until built or declined. Numbers are never reused. Entries with an issue are also tracked on GitHub.

## Open

### F1. `simplex2d` tileable output that actually tiles seamlessly

- Reported: 2026-11-11
- Report: "2 A ado feature request for C" (the maintainer, choosing option 2C from the #65 plan as a feature request)
- Context: `tileable` wraps the lattice in skewed coordinates, so the output repeats along the diagonal and never on x or y. No period fixes that. Possible approaches: 4-D simplex on a torus, or periodic gradient noise. It would change recorded tileable outputs in impressionist (3), plumimpress (4), yapocis-rs (5 bundles) and PlumCraft (1, plus its port).
- Issue: #83

## Done

### F2. CI on the rustcpu backend only

- Reported: 2026-11-12
- Report: "Feature: Add ci for yapocis, on the rustcpu backend only"
- Context: yapocis3 has no `.github/workflows`; its only check on PRs is the Read the Docs build. `make test-rustcpu` runs the suite with yapocis-rs's CPU backend (674 passed, 45 skipped at the 1.2.0 gate) and needs no GPU, but it needs a `yapocis_rs` wheel, which today is built locally from yapocis-rs (`556ef5c` or later).
- Decided (2026-11-12): "For CI on yapocis, build a local wheel to test rustcpu" (the job builds `yapocis_rs` from a pinned yapocis-rs commit); "We never test CI on MacOS -- check your notes. Test on the most recent supported Python." (Linux, Python 3.13); "1: not yet, 2: yes" (no ruff/typecheck yet; confirm the Mako-redirect test in a clean checkout: it passes).
- Done: 2026-11-12, PR #106 (`0ce20ed`), `.github/workflows/ci.yml`; the first green run had 639 passed, 118 skipped.
- Issue: #102

## Declined
