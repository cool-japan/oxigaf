//! Unit tests for [`super`] — the checkpoint browser's parsing,
//! filtering, comparison and trend-fitting logic.
//!
//! Kept in its own file so `checkpoint_browser.rs` stays inside the
//! 2000-line ceiling.

use super::*;

// -----------------------------------------------------------------------
// parse_step_from_path
// -----------------------------------------------------------------------

#[test]
fn test_parse_step_ckpt_prefix() {
    assert_eq!(parse_step_from_path("ckpt_1000.json"), Some(1000));
}

#[test]
fn test_parse_step_step_prefix() {
    assert_eq!(parse_step_from_path("step_500"), Some(500));
}

#[test]
fn test_parse_step_checkpoint_dash() {
    assert_eq!(parse_step_from_path("checkpoint-200"), Some(200));
}

#[test]
fn test_parse_step_checkpoint_underscore() {
    assert_eq!(parse_step_from_path("checkpoint_300.bin"), Some(300));
}

#[test]
fn test_parse_step_model_prefix() {
    assert_eq!(parse_step_from_path("model_1000.json"), Some(1000));
}

#[test]
fn test_parse_step_checkpoint_step_compound() {
    assert_eq!(
        parse_step_from_path("checkpoint_step_12345.json"),
        Some(12345)
    );
}

#[test]
fn test_parse_step_no_number() {
    assert_eq!(parse_step_from_path("final_model.json"), None);
}

#[test]
fn test_parse_step_empty() {
    assert_eq!(parse_step_from_path(""), None);
}

#[test]
fn test_parse_step_with_directory() {
    assert_eq!(
        parse_step_from_path("/run/train/ckpt_2000.json"),
        Some(2000)
    );
}

#[test]
fn test_parse_step_fallback_trailing_number() {
    // "model_best_500" — last number after non-numeric tokens
    assert_eq!(parse_step_from_path("model_best_500"), Some(500));
}

#[test]
fn test_parse_step_prefers_step_over_embedded_date() {
    // The date "20260101" immediately follows the "checkpoint" keyword,
    // but "1000" (the real step) is a better, non-date-shaped candidate
    // elsewhere in the filename.
    assert_eq!(
        parse_step_from_path("checkpoint_20260101_1000.json"),
        Some(1000)
    );
}

#[test]
fn test_parse_step_prefers_step_over_trailing_date() {
    assert_eq!(parse_step_from_path("run_1000_20260101.json"), Some(1000));
}

#[test]
fn test_parse_step_falls_back_to_date_when_no_better_candidate() {
    // A lone date-shaped number is still better than nothing.
    assert_eq!(parse_step_from_path("backup_20260101.json"), Some(20260101));
}

#[test]
fn test_parse_step_prefers_step_over_long_timestamp() {
    // A 10-digit token is implausible as a step count (more likely a
    // Unix timestamp) and must lose to a shorter, plausible candidate.
    assert_eq!(
        parse_step_from_path("snapshot_1735689600_1000.json"),
        Some(1000)
    );
}

// -----------------------------------------------------------------------
// parse_psnr_from_path
// -----------------------------------------------------------------------

#[test]
fn test_parse_psnr_basic() {
    let result = parse_psnr_from_path("ckpt_1000_psnr_28.5.json");
    assert!(result.is_some());
    let v = result.unwrap();
    assert!((v - 28.5).abs() < 0.01, "expected 28.5, got {}", v);
}

#[test]
fn test_parse_psnr_no_psnr() {
    assert!(parse_psnr_from_path("ckpt_1000.json").is_none());
}

#[test]
fn test_parse_psnr_attached() {
    // "model_psnr28.5.bin"
    let result = parse_psnr_from_path("model_psnr28.5.bin");
    assert!(result.is_some());
    let v = result.unwrap();
    assert!((v - 28.5).abs() < 0.01, "expected 28.5, got {}", v);
}

#[test]
fn test_parse_psnr_dash_separator() {
    let result = parse_psnr_from_path("ckpt-psnr-32.1.json");
    assert!(result.is_some());
    let v = result.unwrap();
    assert!((v - 32.1).abs() < 0.01);
}

#[test]
fn test_parse_psnr_integer() {
    let result = parse_psnr_from_path("ckpt_psnr_30.json");
    assert!(result.is_some());
    let v = result.unwrap();
    assert!((v - 30.0).abs() < 0.01);
}

// -----------------------------------------------------------------------
// extract_tags_from_path
// -----------------------------------------------------------------------

#[test]
fn test_extract_tags_best() {
    let tags = extract_tags_from_path("ckpt_best_1000.json");
    assert!(tags.contains(&"best".to_string()));
}

#[test]
fn test_extract_tags_final() {
    let tags = extract_tags_from_path("model_final.bin");
    assert!(tags.contains(&"final".to_string()));
}

#[test]
fn test_extract_tags_latest() {
    let tags = extract_tags_from_path("checkpoint_latest.json");
    assert!(tags.contains(&"latest".to_string()));
}

#[test]
fn test_extract_tags_epoch() {
    let tags = extract_tags_from_path("ckpt_epoch_10_step_1000.json");
    assert!(tags.contains(&"epoch_10".to_string()));
}

#[test]
fn test_extract_tags_empty() {
    let tags = extract_tags_from_path("ckpt_1000.json");
    assert!(tags.is_empty());
}

#[test]
fn test_extract_tags_multiple() {
    let tags = extract_tags_from_path("model_best_final.json");
    assert!(tags.contains(&"best".to_string()));
    assert!(tags.contains(&"final".to_string()));
}

// -----------------------------------------------------------------------
// BrowserCheckpoint
// -----------------------------------------------------------------------

#[test]
fn test_from_path_step_parsed() {
    let c = BrowserCheckpoint::from_path("ckpt_step_5000.json");
    assert_eq!(c.step, 5000);
}

#[test]
fn test_from_path_zero_step_fallback() {
    let c = BrowserCheckpoint::from_path("model.json");
    assert_eq!(c.step, 0);
}

#[test]
fn test_try_from_path_ok() {
    let c = BrowserCheckpoint::try_from_path("ckpt_step_5000.json").expect("should parse");
    assert_eq!(c.step, 5000);
}

#[test]
fn test_try_from_path_errors_when_step_unparseable() {
    let result = BrowserCheckpoint::try_from_path("model.json");
    assert!(matches!(result, Err(BrowserError::ParseError(_))));
}

#[test]
fn test_is_best_true() {
    let c = BrowserCheckpoint::from_path("/checkpoints/ckpt_best_1000.json");
    assert!(c.is_best());
}

#[test]
fn test_is_best_false() {
    let c = BrowserCheckpoint::from_path("ckpt_1000.json");
    assert!(!c.is_best());
}

#[test]
fn test_is_final_true() {
    let c = BrowserCheckpoint::from_path("model_final.json");
    assert!(c.is_final());
}

#[test]
fn test_is_final_false() {
    let c = BrowserCheckpoint::from_path("ckpt_1000.json");
    assert!(!c.is_final());
}

#[test]
fn test_quality_score_with_psnr() {
    let mut c = BrowserCheckpoint::from_path("ckpt_1000.json");
    c.psnr = Some(30.0);
    let score = c.quality_score();
    assert!((score - 0.6).abs() < 1e-5, "expected 0.6, got {}", score);
}

#[test]
fn test_quality_score_with_loss() {
    let mut c = BrowserCheckpoint::from_path("ckpt_1000.json");
    c.loss = Some(0.5);
    let score = c.quality_score();
    assert!((score - 0.5).abs() < 1e-5, "expected 0.5, got {}", score);
}

#[test]
fn test_quality_score_no_metrics() {
    let c = BrowserCheckpoint::from_path("ckpt_1000.json");
    assert!((c.quality_score() - 0.0).abs() < 1e-5);
}

#[test]
fn test_quality_score_psnr_priority_over_loss() {
    let mut c = BrowserCheckpoint::from_path("ckpt_1000.json");
    c.psnr = Some(25.0);
    c.loss = Some(0.1);
    // Should use PSNR: 25/50 = 0.5, not 1-0.1 = 0.9
    let score = c.quality_score();
    assert!((score - 0.5).abs() < 1e-5, "expected 0.5, got {}", score);
}

// -----------------------------------------------------------------------
// CheckpointBrowser construction
// -----------------------------------------------------------------------

#[test]
fn test_browser_empty() {
    let browser = CheckpointBrowser::new(vec![], BrowserConfig::default());
    assert!(browser.is_empty());
    assert_eq!(browser.len(), 0);
}

#[test]
fn test_browser_from_paths() {
    let paths = vec!["ckpt_100.json".to_string(), "ckpt_200.json".to_string()];
    let browser = CheckpointBrowser::from_paths(paths, BrowserConfig::default());
    assert_eq!(browser.len(), 2);
}

#[test]
fn test_browser_total_size_bytes() {
    let mut c1 = BrowserCheckpoint::from_path("ckpt_100.json");
    c1.file_size_bytes = 1024;
    let mut c2 = BrowserCheckpoint::from_path("ckpt_200.json");
    c2.file_size_bytes = 2048;
    let browser = CheckpointBrowser::new(vec![c1, c2], BrowserConfig::default());
    assert_eq!(browser.total_size_bytes(), 3072);
}

#[test]
fn test_browser_step_range() {
    let c1 = BrowserCheckpoint::from_path("ckpt_100.json");
    let c2 = BrowserCheckpoint::from_path("ckpt_500.json");
    let c3 = BrowserCheckpoint::from_path("ckpt_300.json");
    let browser = CheckpointBrowser::new(vec![c1, c2, c3], BrowserConfig::default());
    assert_eq!(browser.step_range(), Some((100, 500)));
}

#[test]
fn test_browser_step_range_empty() {
    let browser = CheckpointBrowser::new(vec![], BrowserConfig::default());
    assert!(browser.step_range().is_none());
}

// -----------------------------------------------------------------------
// browse() — sort
// -----------------------------------------------------------------------

fn make_checkpoints_with_psnr() -> Vec<BrowserCheckpoint> {
    let mut c1 = BrowserCheckpoint::from_path("ckpt_100.json");
    c1.psnr = Some(25.0);
    let mut c2 = BrowserCheckpoint::from_path("ckpt_300.json");
    c2.psnr = Some(30.0);
    let mut c3 = BrowserCheckpoint::from_path("ckpt_200.json");
    c3.psnr = Some(27.5);
    vec![c1, c2, c3]
}

#[test]
fn test_browse_sort_by_step() {
    let config = BrowserConfig {
        sort_by: BrowserSort::ByStep,
        ..Default::default()
    };
    let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), config);
    let result = browser.browse();
    assert_eq!(result[0].step, 100);
    assert_eq!(result[1].step, 200);
    assert_eq!(result[2].step, 300);
}

#[test]
fn test_browse_sort_by_step_desc() {
    let config = BrowserConfig {
        sort_by: BrowserSort::ByStepDesc,
        ..Default::default()
    };
    let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), config);
    let result = browser.browse();
    assert_eq!(result[0].step, 300);
    assert_eq!(result[1].step, 200);
    assert_eq!(result[2].step, 100);
}

#[test]
fn test_browse_sort_by_psnr() {
    let config = BrowserConfig {
        sort_by: BrowserSort::ByPsnr,
        ..Default::default()
    };
    let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), config);
    let result = browser.browse();
    assert!((result[0].psnr.unwrap() - 30.0).abs() < 0.01);
}

#[test]
fn test_browse_max_display() {
    let config = BrowserConfig {
        max_display: 2,
        ..Default::default()
    };
    let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), config);
    let result = browser.browse();
    assert_eq!(result.len(), 2);
}

// -----------------------------------------------------------------------
// browse() — filter
// -----------------------------------------------------------------------

#[test]
fn test_browse_filter_step_range() {
    let mut config = BrowserConfig::default();
    config.filter.min_step = Some(150);
    config.filter.max_step = Some(250);
    let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), config);
    let result = browser.browse();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].step, 200);
}

#[test]
fn test_browse_filter_min_psnr() {
    let mut config = BrowserConfig::default();
    config.filter.min_psnr = Some(28.0);
    let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), config);
    let result = browser.browse();
    // Only step=300 (psnr=30.0) passes; step=200 (27.5) does not
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].step, 300);
}

#[test]
fn test_browse_filter_max_loss() {
    let mut c1 = BrowserCheckpoint::from_path("ckpt_100.json");
    c1.loss = Some(0.8);
    let mut c2 = BrowserCheckpoint::from_path("ckpt_200.json");
    c2.loss = Some(0.3);
    let mut config = BrowserConfig::default();
    config.filter.max_loss = Some(0.5);
    let browser = CheckpointBrowser::new(vec![c1, c2], config);
    let result = browser.browse();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].step, 200);
}

#[test]
fn test_browse_filter_tags_required() {
    let mut c1 = BrowserCheckpoint::from_path("ckpt_best_100.json");
    c1.psnr = Some(25.0);
    let mut c2 = BrowserCheckpoint::from_path("ckpt_200.json");
    c2.psnr = Some(27.0);
    let mut config = BrowserConfig::default();
    config.filter.tags_required = vec!["best".to_string()];
    let browser = CheckpointBrowser::new(vec![c1, c2], config);
    let result = browser.browse();
    assert_eq!(result.len(), 1);
    assert!(result[0].is_best());
}

#[test]
fn test_browse_filter_tags_excluded() {
    let c1 = BrowserCheckpoint::from_path("ckpt_best_100.json");
    let c2 = BrowserCheckpoint::from_path("ckpt_200.json");
    let mut config = BrowserConfig::default();
    config.filter.tags_excluded = vec!["best".to_string()];
    let browser = CheckpointBrowser::new(vec![c1, c2], config);
    let result = browser.browse();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].step, 200);
}

// -----------------------------------------------------------------------
// find_best
// -----------------------------------------------------------------------

#[test]
fn test_find_best_by_psnr() {
    let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), BrowserConfig::default());
    let best = browser.find_best().expect("should find best");
    assert!((best.psnr.unwrap() - 30.0).abs() < 0.01);
}

#[test]
fn test_find_best_by_loss_when_no_psnr() {
    let mut c1 = BrowserCheckpoint::from_path("ckpt_100.json");
    c1.loss = Some(0.8);
    let mut c2 = BrowserCheckpoint::from_path("ckpt_200.json");
    c2.loss = Some(0.2);
    let browser = CheckpointBrowser::new(vec![c1, c2], BrowserConfig::default());
    let best = browser.find_best().expect("should find best");
    assert_eq!(best.step, 200);
}

#[test]
fn test_find_best_empty() {
    let browser = CheckpointBrowser::new(vec![], BrowserConfig::default());
    assert!(browser.find_best().is_none());
}

// -----------------------------------------------------------------------
// find_latest
// -----------------------------------------------------------------------

#[test]
fn test_find_latest() {
    let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), BrowserConfig::default());
    let latest = browser.find_latest().expect("should find latest");
    assert_eq!(latest.step, 300);
}

#[test]
fn test_find_latest_empty() {
    let browser = CheckpointBrowser::new(vec![], BrowserConfig::default());
    assert!(browser.find_latest().is_none());
}

// -----------------------------------------------------------------------
// find_at_step / find_nearest_step
// -----------------------------------------------------------------------

#[test]
fn test_find_at_step_exact() {
    let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), BrowserConfig::default());
    let found = browser.find_at_step(200).expect("should find step 200");
    assert_eq!(found.step, 200);
}

#[test]
fn test_find_at_step_nearest() {
    let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), BrowserConfig::default());
    // 175 is between 100 and 200; nearest is 200
    let found = browser.find_at_step(175).expect("should find nearest");
    assert_eq!(found.step, 200);
}

#[test]
fn test_find_nearest_step() {
    let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), BrowserConfig::default());
    let found = browser.find_nearest_step(260).expect("should find nearest");
    assert_eq!(found.step, 300);
}

#[test]
fn test_find_at_step_exact_method_returns_exact_match() {
    let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), BrowserConfig::default());
    let found = browser
        .find_at_step_exact(200)
        .expect("step 200 should exist");
    assert_eq!(found.step, 200);
}

#[test]
fn test_find_at_step_exact_method_errors_without_falling_back_to_nearest() {
    // Unlike `find_at_step`, `find_at_step_exact` must not silently
    // substitute the nearest checkpoint (step 200) for a step that
    // isn't actually present.
    let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), BrowserConfig::default());
    let result = browser.find_at_step_exact(175);
    assert!(matches!(result, Err(BrowserError::CheckpointNotFound(_))));
}

// -----------------------------------------------------------------------
// at_percentile
// -----------------------------------------------------------------------

#[test]
fn test_at_percentile_zero() {
    let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), BrowserConfig::default());
    let ckpt = browser.at_percentile(0.0).expect("should return first");
    assert_eq!(ckpt.step, 100);
}

#[test]
fn test_at_percentile_one() {
    let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), BrowserConfig::default());
    let ckpt = browser.at_percentile(1.0).expect("should return last");
    assert_eq!(ckpt.step, 300);
}

#[test]
fn test_at_percentile_half() {
    let browser = CheckpointBrowser::new(make_checkpoints_with_psnr(), BrowserConfig::default());
    let ckpt = browser.at_percentile(0.5).expect("should return middle");
    assert_eq!(ckpt.step, 200);
}

#[test]
fn test_at_percentile_empty() {
    let browser = CheckpointBrowser::new(vec![], BrowserConfig::default());
    assert!(browser.at_percentile(0.5).is_none());
}

// -----------------------------------------------------------------------
// compare_checkpoints
// -----------------------------------------------------------------------

#[test]
fn test_compare_step_delta() {
    let a = BrowserCheckpoint::from_path("ckpt_100.json");
    let b = BrowserCheckpoint::from_path("ckpt_300.json");
    let diff = compare_checkpoints(&a, &b);
    assert_eq!(diff.step_delta, 200);
}

#[test]
fn test_compare_psnr_delta() {
    let mut a = BrowserCheckpoint::from_path("ckpt_100.json");
    a.psnr = Some(25.0);
    let mut b = BrowserCheckpoint::from_path("ckpt_200.json");
    b.psnr = Some(28.0);
    let diff = compare_checkpoints(&a, &b);
    assert!(diff.psnr_delta.is_some());
    assert!((diff.psnr_delta.unwrap() - 3.0).abs() < 0.01);
}

#[test]
fn test_compare_loss_delta() {
    let mut a = BrowserCheckpoint::from_path("ckpt_100.json");
    a.loss = Some(0.5);
    let mut b = BrowserCheckpoint::from_path("ckpt_200.json");
    b.loss = Some(0.3);
    let diff = compare_checkpoints(&a, &b);
    assert!(diff.loss_delta.is_some());
    assert!((diff.loss_delta.unwrap() - (-0.2)).abs() < 0.001);
}

#[test]
fn test_compare_tags_added_removed() {
    let a = BrowserCheckpoint::from_path("ckpt_best_100.json");
    let b = BrowserCheckpoint::from_path("ckpt_final_200.json");
    let diff = compare_checkpoints(&a, &b);
    assert!(diff.tags_added.contains(&"final".to_string()));
    assert!(diff.tags_removed.contains(&"best".to_string()));
}

#[test]
fn test_compare_size_delta() {
    let mut a = BrowserCheckpoint::from_path("ckpt_100.json");
    a.file_size_bytes = 1000;
    let mut b = BrowserCheckpoint::from_path("ckpt_200.json");
    b.file_size_bytes = 2500;
    let diff = compare_checkpoints(&a, &b);
    assert_eq!(diff.size_delta, 1500);
}

// -----------------------------------------------------------------------
// psnr_trend
// -----------------------------------------------------------------------

#[test]
fn test_psnr_trend_sorted_by_step() {
    let ckpts = make_checkpoints_with_psnr();
    let trend = psnr_trend(&ckpts);
    assert_eq!(trend.len(), 3);
    // Should be sorted by step ascending
    assert!(trend[0].0 < trend[1].0);
    assert!(trend[1].0 < trend[2].0);
}

#[test]
fn test_psnr_trend_excludes_no_psnr() {
    let mut ckpts = make_checkpoints_with_psnr();
    let no_psnr = BrowserCheckpoint::from_path("ckpt_400.json");
    ckpts.push(no_psnr);
    let trend = psnr_trend(&ckpts);
    assert_eq!(trend.len(), 3); // Only 3 have PSNR
}

#[test]
fn test_psnr_trend_empty() {
    let trend = psnr_trend(&[]);
    assert!(trend.is_empty());
}

// -----------------------------------------------------------------------
// find_psnr_elbow
// -----------------------------------------------------------------------

#[test]
fn test_find_psnr_elbow_basic() {
    // Create a set with diminishing returns
    let mut ckpts = Vec::new();
    let psnrs = [20.0f32, 25.0, 28.0, 29.5, 29.9, 30.0];
    for (i, &p) in psnrs.iter().enumerate() {
        let mut c = BrowserCheckpoint::from_path(&format!("ckpt_{}.json", (i + 1) * 1000));
        c.psnr = Some(p);
        ckpts.push(c);
    }
    let elbow = find_psnr_elbow(&ckpts);
    assert!(elbow.is_some());
    // The elbow should be somewhere in the middle of training
    let e = elbow.unwrap();
    assert!(e > 0);
}

#[test]
fn test_find_psnr_elbow_too_few() {
    let mut ckpts = Vec::new();
    for i in 0..2 {
        let mut c = BrowserCheckpoint::from_path(&format!("ckpt_{}.json", (i + 1) * 1000));
        c.psnr = Some(25.0 + i as f32);
        ckpts.push(c);
    }
    assert!(find_psnr_elbow(&ckpts).is_none());
}

// -----------------------------------------------------------------------
// estimate_steps_to_psnr
// -----------------------------------------------------------------------

#[test]
fn test_estimate_steps_to_psnr_improving() {
    let mut ckpts = Vec::new();
    // Linear PSNR: step=1000→psnr=20, step=2000→psnr=25
    let mut c1 = BrowserCheckpoint::from_path("ckpt_1000.json");
    c1.psnr = Some(20.0);
    let mut c2 = BrowserCheckpoint::from_path("ckpt_2000.json");
    c2.psnr = Some(25.0);
    ckpts.push(c1);
    ckpts.push(c2);
    // slope = 5/1000 per step, to reach 30 need 1000 more steps
    let estimate = estimate_steps_to_psnr(&ckpts, 30.0);
    assert!(estimate.is_some());
    let extra = estimate.unwrap();
    // Should be approximately 1000 more steps
    assert!(
        extra > 500 && extra < 2000,
        "expected ~1000 extra steps, got {}",
        extra
    );
}

#[test]
fn test_estimate_steps_to_psnr_declining() {
    let mut c1 = BrowserCheckpoint::from_path("ckpt_1000.json");
    c1.psnr = Some(30.0);
    let mut c2 = BrowserCheckpoint::from_path("ckpt_2000.json");
    c2.psnr = Some(25.0);
    let ckpts = vec![c1, c2];
    // Declining PSNR → None
    assert!(estimate_steps_to_psnr(&ckpts, 35.0).is_none());
}

#[test]
fn test_estimate_steps_to_psnr_too_few() {
    let mut c = BrowserCheckpoint::from_path("ckpt_1000.json");
    c.psnr = Some(25.0);
    assert!(estimate_steps_to_psnr(&[c], 30.0).is_none());
}

#[test]
fn test_estimate_steps_already_reached() {
    let mut c1 = BrowserCheckpoint::from_path("ckpt_1000.json");
    c1.psnr = Some(20.0);
    let mut c2 = BrowserCheckpoint::from_path("ckpt_2000.json");
    c2.psnr = Some(30.0);
    let ckpts = vec![c1, c2];
    // Target already reached
    let estimate = estimate_steps_to_psnr(&ckpts, 25.0);
    assert_eq!(estimate, Some(0));
}

#[test]
fn test_estimate_steps_to_psnr_precise_at_realistic_step_magnitude() {
    // Regression test for catastrophic cancellation: with step values
    // around 250,000 (routine for a long training run), the naive f32
    // formula `n*sum_xx - sum_x^2` subtracts two ~1e10-magnitude
    // quantities that are nearly equal, destroying the tiny (~10)
    // magnitude signal that actually encodes the slope. The exact
    // answer here is computable by hand: steps 250000..=250004 with
    // psnr 20.0, 20.1, ..., 20.4 is a perfect line of slope 0.1 and
    // intercept -24980.0, so reaching psnr=21.0 needs exactly 6 more
    // steps past the last observed one (250004).
    let mut ckpts = Vec::new();
    for i in 0..5u32 {
        let mut c = BrowserCheckpoint::from_path(&format!("ckpt_{}.json", 250_000 + i));
        c.psnr = Some(20.0 + i as f32 * 0.1);
        ckpts.push(c);
    }
    let estimate = estimate_steps_to_psnr(&ckpts, 21.0);
    assert_eq!(estimate, Some(6), "expected exactly 6 extra steps");
}

/// The quantisation tolerance must be small enough that a genuinely
/// fractional demand still rounds up: 4.5 steps is 5, not 4.
#[test]
fn test_estimate_steps_to_psnr_still_rounds_a_real_fraction_up() {
    let mut c1 = BrowserCheckpoint::from_path("ckpt_100.json");
    c1.psnr = Some(20.0);
    let mut c2 = BrowserCheckpoint::from_path("ckpt_200.json");
    c2.psnr = Some(30.0);
    // slope 0.1/step, so psnr 30.45 sits 4.5 steps past step 200.
    assert_eq!(estimate_steps_to_psnr(&[c1, c2], 30.45), Some(5));
}

/// A NaN PSNR (a corrupt or half-written metrics file) poisons every sum
/// in the fit, and `slope <= 0.0` is false for NaN — so the guard used to
/// pass and `NaN as usize` saturated to 0, reporting "already reached".
#[test]
fn test_estimate_steps_to_psnr_nan_metric_is_none() {
    let mut c1 = BrowserCheckpoint::from_path("ckpt_1000.json");
    c1.psnr = Some(f32::NAN);
    let mut c2 = BrowserCheckpoint::from_path("ckpt_2000.json");
    c2.psnr = Some(25.0);
    assert!(estimate_steps_to_psnr(&[c1, c2], 30.0).is_none());
}

#[test]
fn test_estimate_steps_to_psnr_degenerate_identical_steps_is_none() {
    // All checkpoints claim the same step: the fit is undefined (zero
    // variance in x), and must not panic or return a nonsense value.
    let mut c1 = BrowserCheckpoint::from_path("ckpt_1000.json");
    c1.psnr = Some(20.0);
    let mut c2 = BrowserCheckpoint::from_path("ckpt_1000_b.json");
    c2.step = 1000;
    c2.psnr = Some(25.0);
    assert!(estimate_steps_to_psnr(&[c1, c2], 30.0).is_none());
}

// -----------------------------------------------------------------------
// checkpoint_spacing_stats
// -----------------------------------------------------------------------

#[test]
fn test_spacing_stats_regular() {
    let paths = [
        "ckpt_100.json",
        "ckpt_200.json",
        "ckpt_300.json",
        "ckpt_400.json",
    ];
    let ckpts: Vec<_> = paths
        .iter()
        .map(|p| BrowserCheckpoint::from_path(p))
        .collect();
    let stats = checkpoint_spacing_stats(&ckpts);
    assert!(stats.is_regular, "evenly spaced should be regular");
    assert!((stats.mean_step_gap - 100.0).abs() < 0.01);
    assert_eq!(stats.min_step_gap, 100);
    assert_eq!(stats.max_step_gap, 100);
    assert_eq!(stats.total_steps, 300);
}

#[test]
fn test_spacing_stats_irregular() {
    let paths = ["ckpt_100.json", "ckpt_200.json", "ckpt_800.json"];
    let ckpts: Vec<_> = paths
        .iter()
        .map(|p| BrowserCheckpoint::from_path(p))
        .collect();
    let stats = checkpoint_spacing_stats(&ckpts);
    assert!(!stats.is_regular, "irregular spacing should not be regular");
}

#[test]
fn test_spacing_stats_single() {
    let ckpts = vec![BrowserCheckpoint::from_path("ckpt_100.json")];
    let stats = checkpoint_spacing_stats(&ckpts);
    assert_eq!(stats.total_steps, 0);
}

#[test]
fn test_spacing_stats_empty() {
    let stats = checkpoint_spacing_stats(&[]);
    assert!(stats.is_regular);
    assert_eq!(stats.total_steps, 0);
}

// -----------------------------------------------------------------------
// describe_checkpoint
// -----------------------------------------------------------------------

#[test]
fn test_describe_checkpoint_non_empty() {
    let mut c = BrowserCheckpoint::from_path("ckpt_1000.json");
    c.psnr = Some(28.5);
    let desc = describe_checkpoint(&c);
    assert!(!desc.is_empty());
    assert!(desc.contains("step=1000"));
    assert!(desc.contains("psnr=28.50"));
}

#[test]
fn test_describe_checkpoint_minimal() {
    let c = BrowserCheckpoint::from_path("ckpt_0.json");
    let desc = describe_checkpoint(&c);
    assert!(!desc.is_empty());
    assert!(desc.contains("step=0"));
}

// -----------------------------------------------------------------------
// format_checkpoint_table
// -----------------------------------------------------------------------

#[test]
fn test_format_checkpoint_table_non_empty() {
    let ckpts = make_checkpoints_with_psnr();
    let refs: Vec<&BrowserCheckpoint> = ckpts.iter().collect();
    let table = format_checkpoint_table(&refs);
    assert!(!table.is_empty());
    assert!(table.contains("Step"));
    assert!(table.contains("PSNR"));
}

#[test]
fn test_format_checkpoint_table_empty() {
    let table = format_checkpoint_table(&[]);
    assert!(!table.is_empty()); // Still has header
    assert!(table.contains("Step"));
}

// -----------------------------------------------------------------------
// format_checkpoint_diff
// -----------------------------------------------------------------------

#[test]
fn test_format_checkpoint_diff_non_empty() {
    let a = BrowserCheckpoint::from_path("ckpt_100.json");
    let b = BrowserCheckpoint::from_path("ckpt_200.json");
    let diff = compare_checkpoints(&a, &b);
    let formatted = format_checkpoint_diff(&diff);
    assert!(!formatted.is_empty());
    assert!(formatted.contains("Step delta"));
}

// -----------------------------------------------------------------------
// format_spacing_stats
// -----------------------------------------------------------------------

#[test]
fn test_format_spacing_stats_non_empty() {
    let paths = ["ckpt_100.json", "ckpt_200.json", "ckpt_300.json"];
    let ckpts: Vec<_> = paths
        .iter()
        .map(|p| BrowserCheckpoint::from_path(p))
        .collect();
    let stats = checkpoint_spacing_stats(&ckpts);
    let formatted = format_spacing_stats(&stats);
    assert!(!formatted.is_empty());
    assert!(formatted.contains("mean"));
}

// -----------------------------------------------------------------------
// BrowserError display
// -----------------------------------------------------------------------

#[test]
fn test_error_no_checkpoints_display() {
    let e = BrowserError::NoCheckpoints("/tmp/ckpts".to_string());
    let msg = format!("{}", e);
    assert!(msg.contains("/tmp/ckpts"));
}

#[test]
fn test_error_too_few_checkpoints_display() {
    let e = BrowserError::TooFewCheckpoints(1);
    let msg = format!("{}", e);
    assert!(msg.contains("1"));
}

#[test]
fn test_error_checkpoint_not_found() {
    let e = BrowserError::CheckpointNotFound("ckpt_999.json".to_string());
    let msg = format!("{}", e);
    assert!(msg.contains("ckpt_999.json"));
}

// -----------------------------------------------------------------------
// browse() — sort by loss and quality score
// -----------------------------------------------------------------------

#[test]
fn test_browse_sort_by_loss() {
    let mut c1 = BrowserCheckpoint::from_path("ckpt_100.json");
    c1.loss = Some(0.8);
    let mut c2 = BrowserCheckpoint::from_path("ckpt_200.json");
    c2.loss = Some(0.2);
    let mut c3 = BrowserCheckpoint::from_path("ckpt_300.json");
    c3.loss = Some(0.5);
    let config = BrowserConfig {
        sort_by: BrowserSort::ByLoss,
        ..Default::default()
    };
    let browser = CheckpointBrowser::new(vec![c1, c2, c3], config);
    let result = browser.browse();
    assert!((result[0].loss.unwrap() - 0.2).abs() < 0.001);
}

#[test]
fn test_browse_sort_by_quality_score() {
    let mut c1 = BrowserCheckpoint::from_path("ckpt_100.json");
    c1.psnr = Some(20.0);
    let mut c2 = BrowserCheckpoint::from_path("ckpt_200.json");
    c2.psnr = Some(35.0);
    let config = BrowserConfig {
        sort_by: BrowserSort::ByQualityScore,
        ..Default::default()
    };
    let browser = CheckpointBrowser::new(vec![c1, c2], config);
    let result = browser.browse();
    assert_eq!(result[0].step, 200); // higher PSNR = higher quality
}

#[test]
fn test_browse_sort_by_file_size() {
    let mut c1 = BrowserCheckpoint::from_path("ckpt_100.json");
    c1.file_size_bytes = 500;
    let mut c2 = BrowserCheckpoint::from_path("ckpt_200.json");
    c2.file_size_bytes = 2000;
    let config = BrowserConfig {
        sort_by: BrowserSort::ByFileSize,
        ..Default::default()
    };
    let browser = CheckpointBrowser::new(vec![c1, c2], config);
    let result = browser.browse();
    assert_eq!(result[0].step, 200); // largest first
}
