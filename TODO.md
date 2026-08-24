# OxiGAF TODO — Production-Grade Release Plan (0.1.2)

> Regenerated 2026-08-24 from an exhaustive 21-agent production-readiness audit
> (every src file of all 7 crates read end-to-end, plus cross-cutting audits of
> dependencies/Pure-Rust compliance, release readiness, CLI wiring, GPU
> performance, and the no-unwrap policy). The audit produced **1,114 unique
> findings** (53 critical / 217 high / 604 medium / 240 low), all tracked to
> resolution through the workstreams below.

Baseline before this effort: 12,537 tests passing (46 skipped), zero warnings,
`cargo check --workspace --all-features` clean.

## W0 — Pure Rust & workspace policy (DONE)

- [x] Swap `candle-core`/`candle-nn` to the COOLJAPAN fork
      `oxicandle-core`/`oxicandle-nn` 0.11.0 (package rename only; dependency
      keys unchanged). Removes `onig`/`onig_sys` (C Oniguruma) via tokenizers →
      fancy-regex. Verified: zero `onig` entries in Cargo.lock.
- [x] Remove `candle-transformers` (declared but never referenced in source;
      keeping it would land two incompatible `candle_core` libs in one graph).
- [x] `hf-hub`: `default-features = false, features = ["ureq"]` — drops
      `openssl-sys`/`native-tls` (C OpenSSL). Verified gone from Cargo.lock.
- [x] Create workspace `deny.toml` (bans openblas/bincode/rustfft/z3/rusqlite/
      zip/flate2/… families) with wrapper-scoped, TODO-annotated phase-1
      exceptions. `cargo deny check bans` → **ok**.
- [x] Internal path deps: stale `version = "0.1.1"` pins in member crates →
      `{ workspace = true }`; workspace table carries `version = "0.1.2"` so
      crates stay publishable.
- [x] Track `Cargo.lock` (workspace ships a binary); removed from .gitignore.
- Residual non-Rust on the default feature set: **`ring` only**
  (rustls ← ureq ← hf-hub), documented as TODO in deny.toml. Dev-only:
  criterion → `alloca` (alloca.c), also documented.

## W1 — Audited defect fixes (IN PROGRESS — 74-agent implementation wave)

All 1,114 findings partitioned into file-exclusive buckets
(no two agents own the same file), executing now:

- [ ] **Wave-1 buckets (37 × Sonnet, effort=max)** — every easy/medium bug,
      stub, error-handling, performance, docs-drift finding per crate.
      Highlights: DDPM sigmoid beta schedule NaN, Horn–Schunck alpha no-op,
      SH band-3 constants mis-assigned, atan2 yaw swap (frontal face = 90°),
      Umeyama reflection case, KVCache deep-clone per hit, ICP brute force
      despite kiddo, lazy sequences re-parsing whole files per frame, ~58
      silent-fallback error-handling holes, 5 production `unreachable!` +
      2 `panic!` + 1 `.expect()` (the only no-unwrap violations left).
- [ ] **Hard buckets (20 × Opus)** — real implementations for: ControlNet
      (proper encoder copy + zero-convolutions), avatar/identity conditioning
      weight loading (no more random-weight inference), dynamic landmark
      embedding (real contour chains, stop corrupting iBUG jaw slots), weak-
      perspective rotation estimation, mesh symmetry map, DDPM strided-timestep
      posterior (critical: output stayed noise), batch generator (was returning
      uniform grey images), `oxigaf::pipeline::{render_from_file, export}`
      (were silent no-ops), CLI `convert` .pkl parser (was structurally unable
      to succeed), CLI `benchmark` real measurements (was timing `simulate_*`
      stubs), `setup` real asset URLs + sha256 verification.
- [ ] **GPU chains (2 × Opus, sequential)** — forward: RadixSorter scratch
      overflow >262,144 Gaussians, tile_assign hardcoded tile_size, unbounded
      sort_keys writes, barrier-before-return UB corrupting edge tiles,
      12 MB/frame host clear-upload → GPU-side clear, radix passes over full
      capacity; backward: **gradient attributed to the wrong Gaussian** when
      early termination diverges within a tile (root cause of the relaxed 25%
      position-gradient test threshold), plus barrier UB.
- [ ] **CLI wiring (3 × Opus, sequential)** — 41 modules (~57k lines,
      including never-compiled `scene_optimizer.rs`) unreachable from the
      binary; wire all into a coherent subcommand taxonomy with clap args,
      validation, JSON output, exit codes, shell completions.
- [ ] **Docs buckets (4 × Sonnet)** — rewrite the four per-crate READMEs whose
      examples reference APIs that no longer exist (verified symbol-by-symbol
      drift), fix advertised-but-undefined `cuda`/`metal`/`accelerate`
      features, fill CHANGELOG `[0.1.2]`, complete per-crate publication
      metadata.

## W1.5 — ring/TLS elimination via COOLJAPAN stack (PLANNED, after W1 wave)

Verified feasible (ureq 3.4 manifest + kizzasi precedent). Executes right after
the W1 wave releases file ownership of assets.rs / oxigaf-cli/Cargo.toml:

- [ ] Drop `hf-hub` entirely (sole consumer: oxigaf-cli/src/assets.rs).
      hf-hub 1.0 is a reqwest/aws-lc-rs rewrite (worse: C/asm AWS-LC, no
      provider hook), so replacement — not upgrade — is the pure-Rust path.
- [ ] Direct deps: `ureq = { version = "3.4", default-features = false,
      features = ["rustls-no-provider", "rustls-webpki-roots"] }` (no ring via
      `_ring`, no gzip/flate2, pure-Rust Mozilla roots) + `rustls 0.23
      default-features=false (std, tls12, logging)` +
      `oxitls-rustcrypto-provider = "0.3"` (COOLJAPAN rustls-rustcrypto fork,
      RUSTSEC-2026-0104 fixed).
- [ ] assets.rs: install the RustCrypto provider once via
      `rustls::crypto::CryptoProvider::install_default(
      oxitls_rustcrypto_provider::provider())` (std::sync::Once; tolerate
      AlreadyInstalled), then implement HF Hub download directly: GET
      `https://huggingface.co/{repo}/resolve/{rev}/{file}` with redirects,
      optional `Authorization: Bearer $HF_TOKEN`, indicatif progress, existing
      sha256 verification and cache layout.
- [ ] deny.toml: delete the ring wrappers (ring becomes fully banned), drop
      "ureq" from the flate2 wrappers, update the residual-C note to "none on
      the default feature set" (alloca stays dev/bench-only).
- [ ] Verify: `cargo tree -i ring` empty; `cargo deny check bans` ok; asset
      download tests pass.

## W2 — Convergence & verification (NEXT)

- [ ] `cargo check --workspace --all-features` → fix all compile errors from
      the wave (dedicated serial fix pass).
- [ ] `cargo nextest run --workspace --all-features --no-fail-fast` → converge
      to zero failures (baseline was 12,537 pass).
- [ ] `cargo clippy --workspace --all-features -- -D warnings` → zero.
- [ ] Doc tests + `cargo deny check bans` re-run.
- [ ] Orchestrator gatekeeper review of wave diffs (spot-check every hard/GPU
      bucket, sample wave-1 buckets).

## Deferred / known-tradeoff items

- [ ] `zip` via oxicandle-core (.pth loading) — upstream oxiarc swap.
- [ ] `zip 6.0` via ndarray-npy `npz` — oxiarc-archive-backed .npz reader.
- [ ] flate2/miniz_oxide inside image's png/tiff/exr decoders — needs an
      oxiarc-deflate backend upstream; EXR output is a live CLI feature.
- [ ] GPU backward-pass gradient re-verification against finite differences
      once the wrong-Gaussian attribution fix lands (expect the 25% position
      threshold to tighten).
- [ ] Python `scripts/convert_*.py` — assess retirement once `oxigaf-bridge`
      and CLI `convert` cover the same paths end-to-end.
