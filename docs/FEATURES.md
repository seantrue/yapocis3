# Features

Features requested in passing (`feature: …`), kept until built or declined. Numbers are never reused. Entries with an issue are also tracked on GitHub.

## Open

### F1. `simplex2d` tileable output that actually tiles seamlessly

- Reported: 2026-11-11
- Report: "2 A ado feature request for C" (the maintainer, choosing option 2C from the #65 plan as a feature request)
- Context: `tileable` wraps the lattice in skewed coordinates, so the output repeats along the diagonal and never on x or y. No period fixes that. Possible approaches: 4-D simplex on a torus, or periodic gradient noise. It would change recorded tileable outputs in impressionist (3), plumimpress (4), yapocis-rs (5 bundles) and PlumCraft (1, plus its port).
- Issue: #83

### F2. CI on the rustcpu backend only

- Reported: 2026-11-12
- Report: "Feature: Add ci for yapocis, on the rustcpu backend only"
- Context: yapocis3 has no `.github/workflows`; its only check on PRs is the Read the Docs build. `make test-rustcpu` runs the suite with yapocis-rs's CPU backend (674 passed, 45 skipped at the 1.2.0 gate) and needs no GPU, but it needs a `yapocis_rs` wheel, which today is built locally from yapocis-rs (`556ef5c` or later).
- Open question: where CI gets the `yapocis_rs` wheel: build it in the job from a pinned yapocis-rs commit (Rust toolchain, maturin, cache), download a wheel that yapocis-rs's own CI publishes as an artifact or release, or install it from PyPI if yapocis-rs is published there.
- Open question: which Python versions and OS runners (Linux only, or macOS too), and whether CI also runs ruff/typecheck.
- Open question: the known Mako-redirect artefact (`test_old_compiled_modules_are_removed`) only fails under the local redirect; confirm it passes in a clean CI checkout.
- Issue: #102

## Done

## Declined
