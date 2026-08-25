// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

use super::*;
use std::f32::consts::PI;

// -----------------------------------------------------------------------
// Test helpers
// -----------------------------------------------------------------------

fn make_frame(step: u64, az: f32, el: f32, blink: f32, ts: f64) -> GazeFrame {
    let gaze = GazeDirection::new(az, el, 0.0);
    GazeFrame {
        step,
        left_gaze: gaze,
        right_gaze: gaze,
        blink_left: blink,
        blink_right: blink,
        timestamp_ms: ts,
    }
}

fn default_config() -> GazeControllerConfig {
    GazeControllerConfig::default()
}

// -----------------------------------------------------------------------
// GazeDirection
// -----------------------------------------------------------------------

#[test]
fn test_gaze_direction_primary() {
    let p = GazeDirection::primary();
    assert_eq!(p.azimuth, 0.0);
    assert_eq!(p.elevation, 0.0);
    assert_eq!(p.vergence, 0.0);
}

#[test]
fn test_gaze_direction_default_matches_primary() {
    let d = GazeDirection::default();
    let p = GazeDirection::primary();
    assert_eq!(d.azimuth, p.azimuth);
    assert_eq!(d.elevation, p.elevation);
}

#[test]
fn test_gaze_direction_to_cartesian_primary() {
    let p = GazeDirection::primary();
    let c = p.to_cartesian();
    // primary gaze: az=0, el=0 → (0, 0, 1)
    assert!((c[0]).abs() < 1e-5, "x should be ~0, got {}", c[0]);
    assert!((c[1]).abs() < 1e-5, "y should be ~0, got {}", c[1]);
    assert!((c[2] - 1.0).abs() < 1e-5, "z should be ~1, got {}", c[2]);
}

#[test]
fn test_gaze_direction_to_cartesian_right() {
    // az = pi/2, el = 0 → gaze to the right
    let g = GazeDirection::new(PI / 2.0, 0.0, 0.0);
    let c = g.to_cartesian();
    assert!((c[0] - 1.0).abs() < 1e-5, "x should be ~1, got {}", c[0]);
    assert!((c[1]).abs() < 1e-5, "y should be ~0, got {}", c[1]);
}

#[test]
fn test_gaze_direction_to_cartesian_unit_length() {
    let g = GazeDirection::new(0.3, -0.2, 0.5);
    let c = g.to_cartesian();
    let len = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
    assert!(
        (len - 1.0).abs() < 1e-5,
        "Cartesian vector should be unit, len={len}"
    );
}

// -----------------------------------------------------------------------
// GazeFrame helpers
// -----------------------------------------------------------------------

#[test]
fn test_gaze_frame_cyclopean() {
    let f = make_frame(0, 0.1, -0.1, 0.0, 0.0);
    let c = f.cyclopean_gaze();
    assert!((c.azimuth - 0.1).abs() < 1e-6);
    assert!((c.elevation - (-0.1)).abs() < 1e-6);
}

#[test]
fn test_gaze_frame_mean_blink() {
    let mut f = make_frame(0, 0.0, 0.0, 0.0, 0.0);
    f.blink_left = 0.6;
    f.blink_right = 0.4;
    assert!((f.mean_blink() - 0.5).abs() < 1e-6);
}

#[test]
fn test_gaze_frame_monocular() {
    let g = GazeDirection::new(0.2, 0.1, 1.5);
    let f = GazeFrame::monocular(5, g, 83.3);
    assert_eq!(f.step, 5);
    assert_eq!(f.blink_left, 0.0);
    assert_eq!(f.blink_right, 0.0);
    assert!((f.left_gaze.azimuth - 0.2).abs() < 1e-6);
}

// -----------------------------------------------------------------------
// Listing's law
// -----------------------------------------------------------------------

#[test]
fn test_listing_axis_primary_forward() {
    // primary = [0,0,1], target slightly to the right [sin(0.1), 0, cos(0.1)]
    let primary = [0.0_f32, 0.0, 1.0];
    let angle = 0.1_f32;
    let target = [angle.sin(), 0.0, angle.cos()];
    let axis = gz_listing_axis(primary, target).expect("listing axis");
    // Axis should be in Listing's plane → dot with primary ≈ 0.
    let dot = axis[0] * primary[0] + axis[1] * primary[1] + axis[2] * primary[2];
    assert!(dot.abs() < 1e-5, "axis dot primary should be ~0, got {dot}");
    // Axis should be approximately [0, 1, 0] for rightward gaze (rotation around Y).
    // The sign may differ; check absolute value.
    assert!(
        (axis[1].abs() - 1.0).abs() < 0.05,
        "expect ~Y axis, got {axis:?}"
    );
}

#[test]
fn test_listing_axis_zero_primary_error() {
    let result = gz_listing_axis([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    assert!(result.is_err(), "should fail for zero primary");
}

#[test]
fn test_listing_axis_zero_target_error() {
    let result = gz_listing_axis([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]);
    assert!(result.is_err(), "should fail for zero target");
}

#[test]
fn test_listing_axis_antiparallel_error() {
    let result = gz_listing_axis([0.0, 0.0, 1.0], [0.0, 0.0, -1.0]);
    assert!(result.is_err(), "antiparallel should fail");
}

#[test]
fn test_listing_rotation_identity_for_equal_dirs() {
    let primary = [0.0_f32, 0.0, 1.0];
    let q = gz_listing_rotation(primary, primary).expect("identity listing rotation");
    // Should be identity quaternion [0, 0, 0, 1].
    assert!((q[3] - 1.0).abs() < 1e-5, "w should be ~1, got {}", q[3]);
    assert!(q[0].abs() < 1e-5);
    assert!(q[1].abs() < 1e-5);
    assert!(q[2].abs() < 1e-5);
}

#[test]
fn test_listing_rotation_unit_quaternion() {
    let primary = [0.0_f32, 0.0, 1.0];
    let target = [0.1_f32.sin(), 0.0, 0.1_f32.cos()];
    let q = gz_listing_rotation(primary, target).expect("listing rotation");
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "quaternion should be unit, norm={norm}"
    );
}

#[test]
fn test_listing_rotation_zero_primary_error() {
    let result = gz_listing_rotation([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    assert!(result.is_err());
}

#[test]
fn test_listing_rotation_zero_target_error() {
    let result = gz_listing_rotation([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]);
    assert!(result.is_err());
}

#[test]
fn test_listing_rotation_axis_in_listing_plane() {
    // The rotation axis of the quaternion should be perpendicular to primary.
    let primary = [0.0_f32, 0.0, 1.0];
    let target = [0.0_f32, 0.2_f32.sin(), 0.2_f32.cos()];
    let q = gz_listing_rotation(primary, target).expect("listing rotation");
    // Axis component is [qx, qy, qz].
    let dot = q[0] * primary[0] + q[1] * primary[1] + q[2] * primary[2];
    assert!(
        dot.abs() < 1e-4,
        "rotation axis not in Listing plane, dot={dot}"
    );
}

// -----------------------------------------------------------------------
// Angular velocity
// -----------------------------------------------------------------------

#[test]
fn test_angular_velocity_empty() {
    let vel = gz_angular_velocity(&[], 60.0);
    assert!(vel.is_empty());
}

#[test]
fn test_angular_velocity_single_frame() {
    let frames = vec![make_frame(0, 0.0, 0.0, 0.0, 0.0)];
    let vel = gz_angular_velocity(&frames, 60.0);
    assert!(vel.is_empty());
}

#[test]
fn test_angular_velocity_zero_for_identical() {
    let frames: Vec<GazeFrame> = (0..5)
        .map(|i| make_frame(i, 0.0, 0.0, 0.0, i as f64 / 60.0 * 1000.0))
        .collect();
    let vel = gz_angular_velocity(&frames, 60.0);
    for v in &vel {
        assert!(
            v.abs() < 1e-5,
            "velocity should be zero for identical frames, got {v}"
        );
    }
}

#[test]
fn test_angular_velocity_nonzero_for_moving() {
    // Jump of ~10 degrees between frame 0 and frame 1.
    let deg10 = (10.0_f32).to_radians();
    let frames = vec![
        make_frame(0, 0.0, 0.0, 0.0, 0.0),
        make_frame(1, deg10, 0.0, 0.0, 1000.0 / 60.0),
    ];
    let vel = gz_angular_velocity(&frames, 60.0);
    assert_eq!(vel.len(), 1);
    // At 60fps with 10 deg jump: velocity ≈ 10 * 60 = 600 dps.
    assert!(vel[0] > 500.0, "expected high velocity, got {}", vel[0]);
}

#[test]
fn test_angular_velocity_output_length() {
    let frames: Vec<GazeFrame> = (0..10).map(|i| make_frame(i, 0.0, 0.0, 0.0, 0.0)).collect();
    let vel = gz_angular_velocity(&frames, 60.0);
    assert_eq!(vel.len(), frames.len() - 1);
}

// -----------------------------------------------------------------------
// Saccade detection
// -----------------------------------------------------------------------

#[test]
fn test_detect_saccades_empty() {
    let s = gz_detect_saccades(&[], 60.0, 30.0, 10.0);
    assert!(s.is_empty());
}

#[test]
fn test_detect_saccades_all_slow() {
    // Tiny movement: should not be detected as saccade.
    let frames: Vec<GazeFrame> = (0..20)
        .map(|i| make_frame(i, 0.001 * i as f32, 0.0, 0.0, 0.0))
        .collect();
    let s = gz_detect_saccades(&frames, 60.0, 30.0, 10.0);
    assert!(
        s.is_empty(),
        "no saccade expected for slow movement, got {:?} saccades",
        s.len()
    );
}

#[test]
fn test_detect_saccades_large_jump() {
    let fps = 100.0_f32;
    let threshold = 30.0_f32; // dps
                              // A 20-frame run at 5 deg/frame × 100fps = 500 dps (above threshold).
    let step_deg = 5.0_f32.to_radians();
    let mut frames: Vec<GazeFrame> = Vec::new();
    // 10 static frames.
    for i in 0..10 {
        frames.push(make_frame(i, 0.0, 0.0, 0.0, 0.0));
    }
    // 20 fast frames.
    for i in 0..20 {
        frames.push(make_frame(10 + i, step_deg * i as f32, 0.0, 0.0, 0.0));
    }
    let s = gz_detect_saccades(&frames, fps, threshold, 1.0);
    assert!(!s.is_empty(), "expected saccade detection");
    assert!(s[0].peak_velocity_dps > threshold);
}

#[test]
fn test_detect_saccades_min_duration_filter() {
    // Single-frame saccade should be filtered if min_duration_ms is high.
    let fps = 60.0;
    let jump = (30.0_f32).to_radians();
    let frames = vec![
        make_frame(0, 0.0, 0.0, 0.0, 0.0),
        make_frame(1, jump, 0.0, 0.0, 0.0),
        make_frame(2, jump, 0.0, 0.0, 0.0),
    ];
    // min_duration_ms = 1000 ms → single frame saccade (16.67 ms) should be filtered.
    let s = gz_detect_saccades(&frames, fps, 30.0, 1000.0);
    assert!(
        s.is_empty(),
        "saccade should be filtered by min_duration_ms"
    );
}

// -----------------------------------------------------------------------
// Fixation detection
// -----------------------------------------------------------------------

#[test]
fn test_detect_fixations_empty() {
    let f = gz_detect_fixations(&[], 60.0, 30.0, 100.0);
    assert!(f.is_empty());
}

#[test]
fn test_detect_fixations_stationary_sequence() {
    // All frames at same position → should be a fixation.
    let fps = 60.0;
    let frames: Vec<GazeFrame> = (0..60)
        .map(|i| make_frame(i, 0.0, 0.0, 0.0, i as f64 * 1000.0 / f64::from(fps)))
        .collect();
    let f = gz_detect_fixations(&frames, fps, 30.0, 100.0);
    assert!(
        !f.is_empty(),
        "expected at least one fixation for stationary gaze"
    );
}

#[test]
fn test_detect_fixations_centroid_accuracy() {
    // 30 frames at (az=0.1, el=0.05) → centroid should match.
    let fps = 60.0;
    let az = 0.1_f32;
    let el = 0.05_f32;
    let frames: Vec<GazeFrame> = (0..30).map(|i| make_frame(i, az, el, 0.0, 0.0)).collect();
    let f = gz_detect_fixations(&frames, fps, 30.0, 100.0);
    if !f.is_empty() {
        assert!(
            (f[0].centroid_az - az).abs() < 0.01,
            "centroid_az expected {}, got {}",
            az,
            f[0].centroid_az
        );
        assert!(
            (f[0].centroid_el - el).abs() < 0.01,
            "centroid_el expected {}, got {}",
            el,
            f[0].centroid_el
        );
    }
}

// -----------------------------------------------------------------------
// Blink detection
// -----------------------------------------------------------------------

#[test]
fn test_detect_blinks_empty() {
    let b = gz_detect_blinks(&[], 60.0, 0.5);
    assert!(b.is_empty());
}

#[test]
fn test_detect_blinks_no_blink() {
    let frames: Vec<GazeFrame> = (0..20).map(|i| make_frame(i, 0.0, 0.0, 0.0, 0.0)).collect();
    let b = gz_detect_blinks(&frames, 60.0, 0.5);
    assert!(b.is_empty(), "no blink expected, got {}", b.len());
}

#[test]
fn test_detect_blinks_single_blink() {
    let fps = 60.0;
    // Construct a sequence: open, blink, open.
    let mut frames: Vec<GazeFrame> = (0..10).map(|i| make_frame(i, 0.0, 0.0, 0.0, 0.0)).collect();
    for i in 10..15_u64 {
        frames.push(make_frame(i, 0.0, 0.0, 0.8, 0.0)); // blink
    }
    for i in 15..25_u64 {
        frames.push(make_frame(i, 0.0, 0.0, 0.0, 0.0)); // open
    }
    let b = gz_detect_blinks(&frames, fps, 0.5);
    assert_eq!(b.len(), 1, "expected 1 blink, got {}", b.len());
    assert_eq!(b[0].start_step, 10);
}

// Regression: `BlinkPhase::Opening` was constructed nowhere in the
// crate; a completed excursion is finalised at its falling edge.
#[test]
fn test_detect_blinks_completed_excursion_phase_is_opening() {
    let frames: Vec<GazeFrame> = (0..10)
        .map(|i| {
            make_frame(
                i,
                0.0,
                0.0,
                if (5..8).contains(&i) { 0.8 } else { 0.0 },
                0.0,
            )
        })
        .collect();
    let b = gz_detect_blinks(&frames, 60.0, 0.5);
    assert_eq!(b[0].phase, BlinkPhase::Opening);
}

#[test]
fn test_detect_blinks_threshold_below_not_detected() {
    let fps = 60.0;
    // Blink value is 0.4 (below 0.5 threshold).
    let mut frames: Vec<GazeFrame> = (0..10).map(|i| make_frame(i, 0.0, 0.0, 0.0, 0.0)).collect();
    for i in 10..15_u64 {
        frames.push(make_frame(i, 0.0, 0.0, 0.4, 0.0));
    }
    let b = gz_detect_blinks(&frames, fps, 0.5);
    assert!(b.is_empty(), "blink below threshold should not be detected");
}

#[test]
fn test_detect_blinks_still_open_at_end() {
    // Blink starts but recording ends during blink → BlinkPhase::Closed.
    let fps = 60.0;
    let mut frames: Vec<GazeFrame> = (0..5).map(|i| make_frame(i, 0.0, 0.0, 0.0, 0.0)).collect();
    for i in 5..10_u64 {
        frames.push(make_frame(i, 0.0, 0.0, 0.9, 0.0));
    }
    let b = gz_detect_blinks(&frames, fps, 0.5);
    assert_eq!(b.len(), 1);
    assert_eq!(b[0].phase, BlinkPhase::Closed);
}

// -----------------------------------------------------------------------
// Dispersion
// -----------------------------------------------------------------------

#[test]
fn test_dispersion_empty() {
    assert_eq!(gz_dispersion(&[]), 0.0);
}

#[test]
fn test_dispersion_single() {
    let g = GazeDirection::new(0.1, 0.2, 0.0);
    assert_eq!(gz_dispersion(&[g]), 0.0);
}

#[test]
fn test_dispersion_identical() {
    let g = GazeDirection::new(0.1, 0.2, 0.0);
    let slice = vec![g; 10];
    assert!(
        gz_dispersion(&slice) < 1e-4,
        "identical gaze → near-zero dispersion"
    );
}

#[test]
fn test_dispersion_spread() {
    let g1 = GazeDirection::new(0.0, 0.0, 0.0);
    let g2 = GazeDirection::new((10.0_f32).to_radians(), 0.0, 0.0);
    let d = gz_dispersion(&[g1, g2]);
    assert!(
        (d - 10.0).abs() < 0.1,
        "expected ~10 deg dispersion, got {d}"
    );
}

// -----------------------------------------------------------------------
// Blink waveform
// -----------------------------------------------------------------------

#[test]
fn test_blink_waveform_zero_at_start() {
    assert!((gz_blink_waveform(0.0, 100.0)).abs() < 1e-5);
}

#[test]
fn test_blink_waveform_zero_at_end() {
    assert!((gz_blink_waveform(100.0, 100.0)).abs() < 1e-5);
}

#[test]
fn test_blink_waveform_peak_at_half() {
    let v = gz_blink_waveform(50.0, 100.0);
    assert!(
        (v - 1.0).abs() < 1e-5,
        "waveform should peak at 1.0, got {v}"
    );
}

#[test]
fn test_blink_waveform_monotone_first_half() {
    let duration = 200.0;
    let mut prev = 0.0_f32;
    for i in 0..=50_usize {
        let t = i as f32 * 2.0; // 0..100ms (first half)
        let v = gz_blink_waveform(t, duration);
        assert!(
            v >= prev - 1e-5,
            "waveform should monotonically increase in first half at t={t}"
        );
        prev = v;
    }
}

#[test]
fn test_blink_waveform_monotone_second_half() {
    let duration = 200.0;
    let mut prev = 1.0_f32;
    for i in 50_usize..=100 {
        let t = i as f32 * 2.0; // 100..200ms (second half)
        let v = gz_blink_waveform(t, duration);
        assert!(
            v <= prev + 1e-5,
            "waveform should monotonically decrease in second half at t={t}"
        );
        prev = v;
    }
}

#[test]
fn test_blink_waveform_zero_duration() {
    assert_eq!(gz_blink_waveform(0.0, 0.0), 0.0);
}

#[test]
fn test_blink_waveform_out_of_range() {
    // t > duration should clamp to 0 (end of blink).
    let v = gz_blink_waveform(999.0, 100.0);
    assert!(
        v.abs() < 1e-5,
        "waveform beyond duration should be ~0, got {v}"
    );
}

// -----------------------------------------------------------------------
// Blink synthesis
// -----------------------------------------------------------------------

#[test]
fn test_synthesize_blinks_output_length() {
    let out = gz_synthesize_blinks(600, 60.0, 15.0, 150.0, 42);
    assert_eq!(out.len(), 600);
}

#[test]
fn test_synthesize_blinks_values_in_range() {
    let out = gz_synthesize_blinks(600, 60.0, 15.0, 150.0, 42);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&v),
            "blink value out of range at {i}: {v}"
        );
    }
}

#[test]
fn test_synthesize_blinks_has_blinks() {
    // At 60fps, 10min duration, 15 blinks/min → expect ~150 blinks.
    let n_steps = 60 * 60 * 10; // 10 min at 60fps
    let out = gz_synthesize_blinks(n_steps, 60.0, 15.0, 150.0, 1234);
    let n_peaks: usize = out
        .windows(3)
        .filter(|w| w[1] > w[0] && w[1] > w[2] && w[1] > 0.5)
        .count();
    // Expect at least 50 and no more than 400 blink peaks for 10 min.
    assert!(
        (50..=400).contains(&n_peaks),
        "expected 50-400 blink peaks in 10min, got {n_peaks}"
    );
}

#[test]
fn test_synthesize_blinks_zero_steps() {
    let out = gz_synthesize_blinks(0, 60.0, 15.0, 150.0, 1);
    assert!(out.is_empty());
}

#[test]
fn test_synthesize_blinks_different_seeds_differ() {
    let a = gz_synthesize_blinks(600, 60.0, 15.0, 150.0, 111);
    let b = gz_synthesize_blinks(600, 60.0, 15.0, 150.0, 222);
    // At least some values should differ.
    let differs = a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6);
    assert!(
        differs,
        "different seeds should produce different sequences"
    );
}

// -----------------------------------------------------------------------
// GazeController
// -----------------------------------------------------------------------

#[test]
fn test_controller_new_ok() {
    let config = default_config();
    let ctrl = GazeController::new(config).expect("controller creation should succeed");
    assert!(ctrl.history.is_empty());
}

#[test]
fn test_controller_new_invalid_fps() {
    let mut config = default_config();
    config.fps = -1.0;
    assert!(GazeController::new(config).is_err());
}

#[test]
fn test_controller_push_frame() {
    let config = default_config();
    let mut ctrl = GazeController::new(config).expect("new");
    ctrl.push_frame(make_frame(0, 0.0, 0.0, 0.0, 0.0));
    assert_eq!(ctrl.history.len(), 1);
}

#[test]
fn test_controller_history_cap() {
    let config = default_config();
    let mut ctrl = GazeController::new(config).expect("new");
    for i in 0..1200_u64 {
        ctrl.push_frame(make_frame(i, 0.0, 0.0, 0.0, 0.0));
    }
    assert!(
        ctrl.history.len() <= HISTORY_CAP,
        "history exceeds cap: {}",
        ctrl.history.len()
    );
}

#[test]
fn test_controller_history_cap_exact() {
    let config = default_config();
    let mut ctrl = GazeController::new(config).expect("new");
    for i in 0..1000_u64 {
        ctrl.push_frame(make_frame(i, 0.0, 0.0, 0.0, 0.0));
    }
    assert_eq!(ctrl.history.len(), HISTORY_CAP);
    // Push one more.
    ctrl.push_frame(make_frame(1000, 0.0, 0.0, 0.0, 0.0));
    assert_eq!(ctrl.history.len(), HISTORY_CAP);
}

#[test]
fn test_controller_mean_gaze_empty_error() {
    let config = default_config();
    let ctrl = GazeController::new(config).expect("new");
    assert!(ctrl.mean_gaze(5).is_err());
}

#[test]
fn test_controller_mean_gaze_zero_n_error() {
    let config = default_config();
    let mut ctrl = GazeController::new(config).expect("new");
    ctrl.push_frame(make_frame(0, 0.1, 0.0, 0.0, 0.0));
    assert!(ctrl.mean_gaze(0).is_err());
}

#[test]
fn test_controller_mean_gaze_correct() {
    let config = default_config();
    let mut ctrl = GazeController::new(config).expect("new");
    ctrl.push_frame(make_frame(0, 0.1, 0.0, 0.0, 0.0));
    ctrl.push_frame(make_frame(1, 0.3, 0.0, 0.0, 0.0));
    let mg = ctrl.mean_gaze(2).expect("mean gaze");
    assert!(
        (mg.azimuth - 0.2).abs() < 1e-5,
        "expected az=0.2, got {}",
        mg.azimuth
    );
}

#[test]
fn test_controller_mean_gaze_n_clamped_to_history() {
    let config = default_config();
    let mut ctrl = GazeController::new(config).expect("new");
    ctrl.push_frame(make_frame(0, 0.0, 0.0, 0.0, 0.0));
    // n_last=10 with 1 frame → should use 1 frame without error.
    let mg = ctrl.mean_gaze(10).expect("mean gaze clamped");
    assert!((mg.azimuth).abs() < 1e-5);
}

#[test]
fn test_controller_update_events_no_panic() {
    let config = default_config();
    let mut ctrl = GazeController::new(config).expect("new");
    for i in 0..30_u64 {
        ctrl.push_frame(make_frame(i, 0.0, 0.0, 0.0, i as f64));
    }
    ctrl.update_events(); // should not panic
}

// Regression: `update_events` used to reuse `fixation_min_duration_ms`
// (100ms) for saccades too, discarding a genuine ~33ms saccade.
#[test]
fn test_controller_update_events_detects_short_saccade() {
    let mut ctrl = GazeController::new(default_config()).expect("new");
    ctrl.push_frame(make_frame(0, 0.0, 0.0, 0.0, 0.0));
    ctrl.push_frame(make_frame(1, 5.0_f32.to_radians(), 0.0, 0.0, 0.0));
    ctrl.push_frame(make_frame(2, 5.0_f32.to_radians(), 0.0, 0.0, 0.0));
    ctrl.update_events();
    assert!(!ctrl.saccades().is_empty());
}

#[test]
fn test_controller_synthesize_blinks_uses_config_length() {
    let ctrl = GazeController::new(default_config()).expect("new");
    assert_eq!(ctrl.synthesize_blinks(120, 7).len(), 120);
}

#[test]
fn test_controller_current_fixation_empty() {
    let config = default_config();
    let ctrl = GazeController::new(config).expect("new");
    assert!(ctrl.current_fixation().is_none());
}

#[test]
fn test_controller_is_blinking_empty() {
    let config = default_config();
    let ctrl = GazeController::new(config).expect("new");
    assert!(!ctrl.is_blinking());
}

// -----------------------------------------------------------------------
// Vergence
// -----------------------------------------------------------------------

#[test]
fn test_vergence_from_iod_parallel_eyes() {
    let g = GazeDirection::new(0.0, 0.0, 0.0);
    // Parallel eyes → vergence = 0 (infinity).
    let v = gz_vergence_from_iod(&g, &g, 64.0).expect("vergence");
    assert_eq!(v, 0.0);
}

#[test]
fn test_vergence_from_iod_known_geometry() {
    // IOD = 64 mm = 0.064 m. Object at 1 m.
    // Convergence angle ≈ 2 * atan(0.032 / 1.0) ≈ 3.66 deg ≈ 0.0639 rad.
    // Left eye converges inward (positive az), right eye converges inward (negative az).
    let iod_mm = 64.0_f32;
    let dist_m = 1.0_f32;
    let half_iod = iod_mm / 1000.0 / 2.0;
    let half_angle = (half_iod / dist_m).atan();
    let left = GazeDirection::new(-half_angle, 0.0, 0.0);
    let right = GazeDirection::new(half_angle, 0.0, 0.0);
    let v = gz_vergence_from_iod(&left, &right, iod_mm).expect("vergence");
    assert!((v - dist_m).abs() < 0.01, "expected ~1m, got {v}");
}

#[test]
fn test_convergence_angle_deg_infinity() {
    let angle = gz_convergence_angle_deg(0.0, 0.064);
    assert_eq!(angle, 0.0);
}

#[test]
fn test_convergence_angle_deg_one_meter() {
    // At 1m with 64mm IOD: ~3.67 deg.
    let angle = gz_convergence_angle_deg(1.0, 0.064);
    assert!(
        angle > 3.0 && angle < 4.5,
        "expected ~3.67 deg, got {angle}"
    );
}

#[test]
fn test_convergence_angle_deg_increases_as_distance_decreases() {
    let a1 = gz_convergence_angle_deg(2.0, 0.064);
    let a2 = gz_convergence_angle_deg(0.5, 0.064);
    assert!(
        a2 > a1,
        "closer object should have larger convergence angle"
    );
}

#[test]
fn test_convergence_angle_zero_iod() {
    let angle = gz_convergence_angle_deg(1.0, 0.0);
    assert_eq!(angle, 0.0);
}

// -----------------------------------------------------------------------
// Statistics
// -----------------------------------------------------------------------

#[test]
fn test_compute_stats_empty_controller() {
    let config = default_config();
    let ctrl = GazeController::new(config).expect("new");
    let stats = gz_compute_stats(&ctrl, 60.0);
    assert_eq!(stats.n_frames, 0);
    assert_eq!(stats.n_saccades, 0);
    assert_eq!(stats.n_fixations, 0);
    assert_eq!(stats.n_blinks, 0);
    assert_eq!(stats.mean_fixation_dur_ms, 0.0);
}

#[test]
fn test_compute_stats_frame_count() {
    let config = default_config();
    let mut ctrl = GazeController::new(config).expect("new");
    for i in 0..30_u64 {
        ctrl.push_frame(make_frame(i, 0.0, 0.0, 0.0, 0.0));
    }
    ctrl.update_events();
    let stats = gz_compute_stats(&ctrl, 60.0);
    assert_eq!(stats.n_frames, 30);
}

#[test]
fn test_compute_stats_mean_vergence() {
    let config = default_config();
    let mut ctrl = GazeController::new(config).expect("new");
    // 10 frames with vergence 2.0.
    for i in 0..10_u64 {
        let gaze = GazeDirection::new(0.0, 0.0, 2.0);
        ctrl.push_frame(GazeFrame::monocular(i, gaze, 0.0));
    }
    let stats = gz_compute_stats(&ctrl, 60.0);
    assert!(
        (stats.mean_vergence_m - 2.0).abs() < 1e-5,
        "mean vergence should be 2.0"
    );
}

#[test]
fn test_format_stats_contains_fields() {
    let stats = GazeStats {
        n_frames: 100,
        n_saccades: 5,
        n_fixations: 3,
        n_blinks: 2,
        mean_fixation_dur_ms: 200.0,
        mean_saccade_amplitude_deg: 8.5,
        blink_rate_per_min: 12.0,
        mean_vergence_m: 1.5,
    };
    let s = gz_format_stats(&stats);
    assert!(s.contains("100"), "should contain frame count");
    assert!(s.contains("saccades"), "should mention saccades");
    assert!(s.contains("blinks"), "should mention blinks");
}

// -----------------------------------------------------------------------
// Edge cases
// -----------------------------------------------------------------------

#[test]
fn test_dispersion_two_equal_points() {
    let g = GazeDirection::new(0.0, 0.0, 0.0);
    assert!(gz_dispersion(&[g, g]) < 1e-5);
}

#[test]
fn test_angular_velocity_two_identical_frames() {
    let frames = vec![
        make_frame(0, 0.0, 0.0, 0.0, 0.0),
        make_frame(1, 0.0, 0.0, 0.0, 0.0),
    ];
    let vel = gz_angular_velocity(&frames, 60.0);
    assert_eq!(vel.len(), 1);
    assert!(vel[0].abs() < 1e-5);
}

#[test]
fn test_blink_synthesis_rate_reasonable() {
    // 60 fps, 1 min (3600 frames), 10 blinks/min → ~10 blinks.
    let out = gz_synthesize_blinks(3600, 60.0, 10.0, 150.0, 99);
    // Count transitions from below-0.5 to above-0.5.
    let blink_starts = out.windows(2).filter(|w| w[0] < 0.5 && w[1] >= 0.5).count();
    // Allow large tolerance due to exponential ISI variance.
    assert!(
        (3..=25).contains(&blink_starts),
        "expected 3-25 blinks for 10/min rate in 1min, got {blink_starts}"
    );
}

#[test]
fn test_controller_history_ordering_preserved() {
    let config = default_config();
    let mut ctrl = GazeController::new(config).expect("new");
    for i in 0..10_u64 {
        ctrl.push_frame(make_frame(i, i as f32 * 0.01, 0.0, 0.0, 0.0));
    }
    // History should preserve insertion order.
    for (idx, frame) in ctrl.history.iter().enumerate() {
        assert_eq!(
            frame.step, idx as u64,
            "frame ordering mismatch at index {idx}"
        );
    }
}

#[test]
fn test_listing_rotation_angle_correctness() {
    // primary=[0,0,1], target=[sin(30°), 0, cos(30°)].
    // Expected rotation angle around Y = 30°.
    let primary = [0.0_f32, 0.0, 1.0];
    let deg30 = (30.0_f32).to_radians();
    let target = [deg30.sin(), 0.0, deg30.cos()];
    let q = gz_listing_rotation(primary, target).expect("listing rotation");
    // The rotation angle = 2 * acos(qw).
    let angle = 2.0 * q[3].clamp(-1.0, 1.0).acos();
    assert!(
        (angle - deg30).abs() < 1e-4,
        "expected rotation angle ~30 deg ({deg30} rad), got {angle} rad"
    );
}

#[test]
fn test_vergence_symmetry() {
    // Swapping left and right should give same result.
    let left = GazeDirection::new(-0.02, 0.0, 0.0);
    let right = GazeDirection::new(0.02, 0.0, 0.0);
    let v1 = gz_vergence_from_iod(&left, &right, 64.0).expect("v1");
    let v2 = gz_vergence_from_iod(&right, &left, 64.0).expect("v2");
    assert!((v1 - v2).abs() < 1e-6, "vergence should be symmetric");
}
