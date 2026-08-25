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
      **Superseded by W1.5 below**: `hf-hub` was later dropped entirely (not
      just reconfigured) in favour of a direct `ureq`/`rustls`/
      `oxitls-rustcrypto-provider` stack. This line is kept as a historical
      record of the first step; see W1.5 for the crate that is actually in
      the dependency graph today.
- [x] Create workspace `deny.toml` (bans openblas/bincode/rustfft/z3/rusqlite/
      zip/flate2/… families) with wrapper-scoped, TODO-annotated phase-1
      exceptions. `cargo deny check bans` → **ok**.
- [x] Internal path deps: stale `version = "0.1.1"` pins in member crates →
      `{ workspace = true }`; workspace table carries `version = "0.1.2"` so
      crates stay publishable.
- [x] `.gitignore`'s `Cargo.lock` entry removed (workspace ships a binary, so
      the lockfile belongs in version control).
      **Not yet fully done**: the file itself is still absent from the git
      index (`git ls-files -- Cargo.lock` is empty) — it needs a `git add
      Cargo.lock` once dependency churn from the current wave settles, so it
      isn't staged mid-edit by every agent that happens to touch a manifest.
- Residual non-Rust on the default feature set: **NONE** (ring eliminated in
  W1.5 below). Dev/bench-only: criterion → `alloca` (alloca.c), documented
  in deny.toml.

## W1 — Audited defect fixes (DONE — 65-agent implementation wave, 2026-08-25)

All 1,114 findings partitioned into file-exclusive buckets
(no two agents own the same file). Completed 65/65 agents with zero errors
(chain stages re-verify earlier stages' fixes, so per-item statuses total
above 1,114): every finding resolved as fixed, verified-already-correct,
not-a-bug, or explicitly deferred with reason. Followed by the followup
waves below (W1.7).

- [x] **Wave-1 buckets (37 × Sonnet, effort=max)** — every easy/medium bug,
      stub, error-handling, performance, docs-drift finding per crate.
      Highlights: DDPM sigmoid beta schedule NaN, Horn–Schunck alpha no-op,
      SH band-3 constants mis-assigned, atan2 yaw swap (frontal face = 90°),
      Umeyama reflection case, KVCache deep-clone per hit, ICP brute force
      despite kiddo, lazy sequences re-parsing whole files per frame, ~58
      silent-fallback error-handling holes, 5 production `unreachable!` +
      2 `panic!` + 1 `.expect()` (the only no-unwrap violations left).
- [x] **Hard buckets (20 × Opus)** — real implementations for: ControlNet
      (proper encoder copy + zero-convolutions), avatar/identity conditioning
      weight loading (no more random-weight inference), dynamic landmark
      embedding (real contour chains, stop corrupting iBUG jaw slots), weak-
      perspective rotation estimation, mesh symmetry map, DDPM strided-timestep
      posterior (critical: output stayed noise), batch generator (was returning
      uniform grey images), `oxigaf::pipeline::{render_from_file, export}`
      (were silent no-ops), CLI `convert` .pkl parser (was structurally unable
      to succeed), CLI `benchmark` real measurements (was timing `simulate_*`
      stubs), `setup` real asset URLs + sha256 verification.
- [x] **GPU chains (2 × Opus, sequential)** — forward: RadixSorter scratch
      overflow >262,144 Gaussians, tile_assign hardcoded tile_size, unbounded
      sort_keys writes, barrier-before-return UB corrupting edge tiles,
      12 MB/frame host clear-upload → GPU-side clear, radix passes over full
      capacity; backward: **gradient attributed to the wrong Gaussian** when
      early termination diverges within a tile (root cause of the relaxed 25%
      position-gradient test threshold), plus barrier UB.
- [x] **CLI wiring (3 × Opus, sequential)** — 41 modules (~57k lines,
      including never-compiled `scene_optimizer.rs`) unreachable from the
      binary; wired into a coherent subcommand taxonomy (21 command families
      under `commands/`, main.rs now a thin shim over the lib crate) with
      clap args, validation, JSON output, exit codes, shell completions.
- [x] **Docs buckets (4 × Sonnet)** — rewrite the four per-crate READMEs whose
      examples reference APIs that no longer exist (verified symbol-by-symbol
      drift), fix advertised-but-undefined `cuda`/`metal`/`accelerate`
      features, fill CHANGELOG `[0.1.2]`, complete per-crate publication
      metadata.

## W1.5 — ring/TLS elimination via COOLJAPAN stack (DONE)

- [x] Dropped `hf-hub` entirely (sole consumer was oxigaf-cli/src/assets.rs;
      zero references remain workspace-wide). hf-hub 1.0 is a
      reqwest/aws-lc-rs rewrite (worse: C/asm AWS-LC, no provider hook), so
      replacement — not upgrade — was the pure-Rust path.
- [x] Workspace deps: `ureq 3.4 (default-features = false, rustls-no-provider
      + rustls-webpki-roots)` — no ring, no gzip/flate2, pure-Rust Mozilla
      roots — plus `rustls 0.23 (default-features = false: std, tls12,
      logging)` and `oxitls-rustcrypto-provider 0.3` (COOLJAPAN
      rustls-rustcrypto fork, RUSTSEC-2026-0104 fixed).
- [x] assets.rs: RustCrypto provider installed once via
      `CryptoProvider::install_default` (std::sync::Once, tolerates an
      already-installed provider); `download_with_progress` reimplemented
      directly over the Hub `resolve` endpoint (redirects, `HF_TOKEN` bearer
      auth, `HF_ENDPOINT`/`HF_HUB_CACHE`/`HF_HOME` honored, HF-compatible
      cache layout, `.part` staging, Content-Length truncation check,
      path-traversal validation). Bonus: `download_file` no longer shells out
      to curl/wget — native streaming with real byte-accurate progress.
      Signatures unchanged (wave agents' call sites unaffected).
- [x] deny.toml: ring fully banned (wrappers deleted); "ureq" dropped from
      flate2 wrappers; residual-C note now "NONE on default features".
- [x] Verified: `cargo tree -i ring` → empty; `cargo deny check bans` → ok;
      every ureq/rustls/oxitls API call compile-checked in an isolated probe
      crate; live TLS handshake to huggingface.co succeeded end-to-end via
      the RustCrypto provider.

## W1.7 — Followup waves (DONE, 2026-08-25)

The W1 agents' 347 cross-file followups + 90 deferred items were triaged
(dedup → 44 work items; ~110 marked obsolete against current code), then
executed by further file-exclusive waves — 36 + 4 + 4 + 2 + 1 buckets — plus
three targeted convergence fixers. Highlights:

- [x] flame/render compile convergence (new mirror-matrix retargeting test
      API, wgpu 30 `ErrorScopeGuard` error-scope migration, Debug derives).
- [x] MSRV 1.85 → 1.87 (21 `incompatible_msrv` hits) + oxigaf-flame clippy
      `-D warnings` → zero, no `#[allow]` added.
- [x] `light_probe.rs` (1,942 lines, never compiled) + new unified
      `gltf.rs` writer declared & re-exported from oxigaf-render's crate
      root; CLI/oxigaf delegate to the single implementation.
- [x] Shipping `preprocess_sh*.wgsl` variant shaders' quaternion bug fixed
      atomically across all six shaders (the earlier fix only covered the
      fallback path).
- [x] SdX2 upsampler determinism (seeded sub-stream) locked in with
      mutation-verified regression tests; `DiffusionConfig::default()`
      CLIP/IP-Adapter width inconsistency fixed via a single
      `ip_adapter_context_dim()` knob (SD 1.5 / SDXL presets included);
      `clip_embed_dim` now genuinely sizes the CLIP tower.
- [x] oxigaf-diffusion fully polished: ZERO `#[allow]` in the crate, clippy
      + rustdoc `-D warnings` clean both feature sets, full suite
      3,213/3,213 passing end-to-end, 51 doctests.
- [x] oxigaf-bridge: pure-Rust `.pt`/`.pkl` ingest
      (`convert_pytorch_checkpoint` / `convert_flame_model`); Python
      `scripts/convert_*.py` deprecated; pre-existing safetensors 0.8 API
      breakage fixed; one dot-separated layer-naming convention.
- [x] 2000-line policy splits (diff_tool, cross_frame_consistency,
      rectified_flow, flame/render/cli split buckets); baseline flame test
      failures (emotion intensity, landmark-fitter parity) and the
      nondeterministic pinch-vertex boundary test fixed at the root.

## W2 — Convergence & verification (DONE — final certification 2026-08-26)

- [x] `cargo check --workspace --all-features --all-targets` → clean, zero
      warnings.
- [x] `cargo nextest run --workspace --all-features --no-fail-fast` →
      **15,289 / 15,289 passed** (28 skipped; baseline was 12,537 — the waves
      added ~2,750 tests).
- [x] `cargo clippy --workspace --all-features --all-targets -- -D warnings`
      → zero.
- [x] Doc tests: 194 passed, 0 failed. `RUSTDOCFLAGS="-D warnings" cargo doc
      --workspace --no-deps --all-features` → zero warnings.
- [x] `cargo fmt --all -- --check` → clean. `cargo deny check bans` → ok.
      `cargo tree -i ring` → empty; zero `onig` in Cargo.lock.
- [x] Orchestrator gatekeeper review: GPU backward shader (uniform loop
      bound + validity masks, documented), RadixSorter dynamic scratch,
      tile_assign uniform tile_size — all verified against the audited
      defects; policy sweep (2000-line cap, production dead_code allows,
      hardcoded paths) enforced; accepted structural allows documented under
      Deferred.

## Deferred / known-tradeoff items

- [ ] `zip` via oxicandle-core (.pth loading) — upstream oxiarc swap.
- [ ] `zip 6.0` via ndarray-npy `npz` — oxiarc-archive-backed .npz reader.
- [ ] flate2/miniz_oxide inside image's png/tiff/exr decoders — needs an
      oxiarc-deflate backend upstream; EXR output is a live CLI feature.
- [ ] GPU backward-pass gradient re-verification against finite differences
      on real GPU hardware, now that the wrong-Gaussian attribution fix has
      landed (expect the 25% position threshold to tighten).
- [x] Python `scripts/convert_*.py` — retirement assessed and implemented:
      oxigaf-bridge now ships pure-Rust `.pt`/`.pkl` ingest
      (`convert_pytorch_checkpoint` / `convert_flame_model` + examples); both
      scripts carry DEPRECATED docstrings pointing at the Rust replacements.
- [ ] Optional: wire the bridge's `.pt` ingest into `oxigaf convert` directly
      (needs an oxigaf-cli → oxigaf-bridge dependency edge + subcommand).
- [ ] Optional: expose `lr_schedule` / `gradient_clip` (enum-valued trainer
      knobs) in the CLI TOML schema — both currently fixed at trainer
      defaults; schema sketch recorded in the convergence fixer's report.
- Accepted (gatekeeper decision, 2026-08-25): pre-existing structural
  `#[allow]`s in production code are kept where the lint is a documented
  style trade-off — `too_many_arguments` on internal constructors (~15,
  spec-struct refactors would ripple through pub APIs), display-only
  `cast_precision_loss` (~10), `many_single_char_names` in math kernels with
  paper notation, and test-module `unwrap_used`/`expect_used` exemptions
  (the crates deny unwrap in production). `/tmp` string literals in tests
  are all non-writing path fixtures (no disk I/O) and likewise accepted.
  oxigaf-diffusion is the zero-`#[allow]` reference crate; production
  `#[allow(dead_code)]` sites were separately audited and eliminated.
