# OxiGAF Design Documents

This directory contains the original design plans and architecture documents for the OxiGAF project.

> **⚠️ Historical document — planning snapshot, not current status.** This
> README (and `IMPLEMENTATION_PLAN.md` alongside it) reflects design
> planning as it stood on **2026-02-09**, before the v0.1.0 release. Every
> item this file used to mark `0%` / `CRITICAL` — Latent Upsampler,
> IP-Adapter, Classifier-Free Guidance, gradient verification, safetensors
> support, and the weight-conversion script — shipped in v0.1.0; see
> `CHANGELOG.md` at the repository root for what is actually implemented.
> The percentage/status tracking that used to live in this file has been
> removed rather than refreshed with new numbers, since a second hand-typed
> snapshot would only go stale again the same way. For current, maintained
> status use `CHANGELOG.md` and each crate's own `crates/*/TODO.md`.

## Overview

OxiGAF is a Pure Rust implementation of GAF (Gaussian Avatar Reconstruction from Monocular Videos via Multi-View Diffusion), targeting 100% Pure Rust with zero C/Fortran dependencies (following COOLJAPAN Pure Rust Policy).

## Document Index

Only two of the originally-planned design documents in this directory were ever written:

- **[IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)** — original whole-workspace implementation roadmap (same 2026-02-09 snapshot caveat as above)
- **[oxigaf-flame-plan.md](./oxigaf-flame-plan.md)** — FLAME parametric head model design

`oxigaf-diffusion-plan.md`, `oxigaf-render-plan.md`, `oxigaf-trainer-plan.md`, `oxigaf-cli-plan.md`, `oxigaf.md`, and `oxigaf-workspace-plan.md` were referenced from an earlier version of this index but were never written and do not exist in this directory. For the design and status of those modules, use each crate's own documentation instead:

| Module | Documentation |
|---|---|
| `oxigaf-flame` | `crates/oxigaf-flame/README.md`, `crates/oxigaf-flame/TODO.md`, plus the design doc above |
| `oxigaf-diffusion` | `crates/oxigaf-diffusion/README.md`, `crates/oxigaf-diffusion/TODO.md` |
| `oxigaf-render` | `crates/oxigaf-render/README.md`, `crates/oxigaf-render/TODO.md` |
| `oxigaf-trainer` | `crates/oxigaf-trainer/README.md`, `crates/oxigaf-trainer/TODO.md` |
| `oxigaf-cli` | `crates/oxigaf-cli/README.md`, `crates/oxigaf-cli/TODO.md` |
| `oxigaf` (meta crate) | `crates/oxigaf/README.md`, `crates/oxigaf/TODO.md` |
| Workspace structure | root `README.md` and `Cargo.toml` |

## How to Use These Documents

### For New Contributors
1. Read `CHANGELOG.md` for what is actually implemented today.
2. Read `IMPLEMENTATION_PLAN.md` and `oxigaf-flame-plan.md` for the historical design rationale behind the architecture.
3. Check the corresponding `crates/*/TODO.md` for current, maintained status and remaining work.

### For Understanding Architecture
- These design documents explain **why** decisions were made, as of the 2026-02-09 planning snapshot.
- `crates/*/TODO.md` files track **what** currently remains to be done.
- `CHANGELOG.md` is the authoritative record of what has shipped, by version.

## Design Philosophy

### COOLJAPAN Pure Rust Policy
- No C/Fortran dependencies (default features)
- Replace openblas → oxiblas
- Replace bincode → oxicode
- Replace rustfft → oxifft
- Replace z3 → oxiz
- Replace zip → oxiarc-archive ✅ (completed 2026-02-09)

### Code Quality Standards
- No unwrap policy (use proper error handling)
- No expect in library code
- File size limit: < 2000 lines
- Workspace version management (`*.workspace = true`)
- Comprehensive automated test suite (see `CHANGELOG.md` / CI for current counts)

### Feature Flag Design
- Default: Pure Rust, CPU-only
- Optional: `simd`, `parallel` (FLAME performance)
- Optional: `flash_attention`, `mixed_precision` (diffusion performance/memory)
- Optional: `gpu_debug` (GPU validation and NaN/Inf debug hooks)

## Related Documentation

- **Root README.md** — Project overview and quick start
- **CHANGELOG.md** — Version history and release notes (authoritative current status)
- **Crate READMEs** — Individual `crates/*/README.md` for API docs
- **TODO Files** — Current status in `crates/*/TODO.md`

## Questions and Feedback

For questions about design decisions or historical implementation status:
1. Check `CHANGELOG.md` first for current status.
2. Review the corresponding `crates/*/TODO.md` for outstanding work.
3. Read the design documents above for the original architectural rationale.

## License

All design documents are part of the OxiGAF project.
Copyright © COOLJAPAN OU (Team Kitasan)

---

*Documents originally written and reorganized: 2026-02-09*
*Marked historical, status apparatus removed: 2026-08-24*
