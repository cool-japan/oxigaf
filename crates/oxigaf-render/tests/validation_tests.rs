//! Tests for CPU-side validation helpers in `debug_readback`.

use oxigaf_render::debug_readback::{validate_buffer_count, validate_no_nan_inf};

// ── validate_no_nan_inf ───────────────────────────────────────────────────────

#[test]
fn test_validate_no_nan_inf_clean_data() {
    let data = [1.0_f32, 2.0, 3.0];
    assert!(validate_no_nan_inf(&data, "test_clean").is_ok());
}

#[test]
fn test_validate_no_nan_inf_empty_slice() {
    assert!(validate_no_nan_inf(&[], "empty").is_ok());
}

#[test]
fn test_validate_no_nan_inf_detects_nan() {
    let data = [1.0_f32, f32::NAN, 3.0];
    let result = validate_no_nan_inf(&data, "test_nan");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("NaN"), "error should mention NaN, got: {msg}");
    assert!(
        msg.contains("index 1"),
        "error should mention index 1, got: {msg}"
    );
}

#[test]
fn test_validate_no_nan_inf_detects_inf() {
    let data = [1.0_f32, f32::INFINITY, 3.0];
    let result = validate_no_nan_inf(&data, "test_inf");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("Inf"), "error should mention Inf, got: {msg}");
    assert!(
        msg.contains("index 1"),
        "error should mention index 1, got: {msg}"
    );
}

#[test]
fn test_validate_no_nan_inf_detects_neg_inf() {
    let data = [f32::NEG_INFINITY, 2.0_f32];
    let result = validate_no_nan_inf(&data, "neg_inf");
    assert!(result.is_err());
}

#[test]
fn test_validate_no_nan_inf_label_included_in_error() {
    let data = [f32::NAN];
    let result = validate_no_nan_inf(&data, "my_buffer");
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("my_buffer"), "label missing from error: {msg}");
}

// ── validate_buffer_count ─────────────────────────────────────────────────────

#[test]
fn test_validate_buffer_count_match() {
    assert!(validate_buffer_count(42, 42, "buf").is_ok());
}

#[test]
fn test_validate_buffer_count_zero_match() {
    assert!(validate_buffer_count(0, 0, "empty_buf").is_ok());
}

#[test]
fn test_validate_buffer_count_mismatch_too_few() {
    let result = validate_buffer_count(3, 5, "buf");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("expected 5") && msg.contains("got 3"),
        "unexpected msg: {msg}"
    );
}

#[test]
fn test_validate_buffer_count_mismatch_too_many() {
    let result = validate_buffer_count(10, 4, "grad_buf");
    assert!(result.is_err());
}

#[test]
fn test_validate_buffer_count_label_in_error() {
    let result = validate_buffer_count(1, 2, "positions_buf");
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("positions_buf"), "label missing: {msg}");
}
