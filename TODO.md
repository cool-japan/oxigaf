# OxiGAF TODO

## Stubs to implement (added 2026-06-12 by /cooljapan-stub-check)

- [ ] `oxigaf-render`: `crates/oxigaf-render/tests/gradient_verification/test_position.rs:89` — implement GPU backward pass for position gradients and compare with analytical gradients (currently skipped)
  - Priority: P2 | Scope: large | Hint: none
- [ ] `oxigaf-render`: `crates/oxigaf-render/tests/gradient_verification/test_sh.rs:297` — implement `compute_sh_gradients` and enable gradient verification for spherical harmonics
  - Priority: P2 | Scope: large | Hint: none

## Stubs to implement (added 2026-06-22 by /cooljapan-stub-check)

- [ ] **oxigaf** `oxigaf-render`: `crates/oxigaf-render/tests/gradient_verification/test_sh.rs:297` — `TODO`: `Implement actual gradient verification when compute_sh_gradients is available`
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Implement `compute_sh_gradients` and assert the analytical spherical-harmonics gradients against finite-difference references.
  - **Risk:** SH basis ordering/normalization must match between analytical and FD paths or the comparison will falsely fail; pin the convention.
- [ ] **oxigaf** `oxigaf-render`: `crates/oxigaf-render/tests/gradient_verification/test_position.rs:89` — `TODO`: `When GPU backward pass is implemented, compare with analytical gradients`
  - **Priority:** P2  **Scope:** large  **Cross-project:** none
  - **Approach:** Once the GPU backward pass lands, compare the computed position gradients against the analytical gradients instead of only the current finite-check.
  - **Risk:** GPU-backend-gated; keep the finite-value fallback assertion active when the backward pass is unavailable.
