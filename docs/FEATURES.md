# Features

Features requested in passing (`feature: …`), kept until built or declined. Numbers are never reused. Entries with an issue are also tracked on GitHub.

## Open

### F1. `simplex2d` tileable output that actually tiles seamlessly

- Reported: 2026-11-11
- Report: "2 A ado feature request for C" (the maintainer, choosing option 2C from the #65 plan as a feature request)
- Context: `tileable` wraps the lattice in skewed coordinates, so the output repeats along the diagonal and never on x or y. No period fixes that. Possible approaches: 4-D simplex on a torus, or periodic gradient noise. It would change recorded tileable outputs in impressionist (3), plumimpress (4), yapocis-rs (5 bundles) and PlumCraft (1, plus its port).
- Issue: #83

## Done

## Declined
