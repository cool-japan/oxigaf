// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

use super::*;

// --- Test helpers ---

fn make_frame(idx: usize, ts: f32, yaw: f32, pitch: f32, roll: f32) -> HeadTrackFrame {
    HeadTrackFrame::new(
        idx,
        ts,
        HeadTrackPose {
            yaw,
            pitch,
            roll,
            tx: 0.0,
            ty: 0.0,
            tz: 100.0,
            confidence: 0.9,
        },
    )
}

fn simple_trajectory(n: usize) -> HeadTrajectory {
    let frames: Vec<HeadTrackFrame> = (0..n)
        .map(|i| make_frame(i, i as f32 * 33.0, i as f32 * 0.05, 0.0, 0.0))
        .collect();
    HeadTrajectory::new(frames).expect("valid trajectory")
}

// --- HeadTrackFrame ---

#[test]
fn test_frame_new_valid() {
    let f = HeadTrackFrame::new(
        0,
        0.0,
        HeadTrackPose {
            yaw: 0.1,
            pitch: 0.2,
            roll: 0.3,
            tx: 1.0,
            ty: 2.0,
            tz: 500.0,
            confidence: 0.95,
        },
    );
    assert!(f.is_valid);
    assert_eq!(f.confidence, 0.95);
}

#[test]
fn test_frame_invalid() {
    let f = HeadTrackFrame::invalid(5, 165.0);
    assert!(!f.is_valid);
    assert_eq!(f.frame_idx, 5);
    assert_eq!(f.confidence, 0.0);
}

#[test]
fn test_euler_angles() {
    let f = make_frame(0, 0.0, 0.1, 0.2, 0.3);
    assert_eq!(f.euler_angles(), [0.1, 0.2, 0.3]);
}

#[test]
fn test_translation() {
    let f = HeadTrackFrame::new(
        0,
        0.0,
        HeadTrackPose {
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            tx: 3.0,
            ty: 4.0,
            tz: 0.0,
            confidence: 1.0,
        },
    );
    assert_eq!(f.translation(), [3.0, 4.0, 0.0]);
}

#[test]
fn test_rotation_magnitude() {
    let f = make_frame(0, 0.0, 3.0, 4.0, 0.0);
    assert!((f.rotation_magnitude() - 5.0).abs() < 1e-5);
}

// --- HeadTrackerConfig::validate ---

#[test]
fn test_config_default_valid() {
    assert!(HeadTrackerConfig::default().validate().is_ok());
}

#[test]
fn test_config_zero_outlier_threshold() {
    let cfg = HeadTrackerConfig {
        outlier_threshold: 0.0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_bad_confidence_threshold() {
    let cfg = HeadTrackerConfig {
        confidence_threshold: -0.1,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_zero_max_history() {
    let cfg = HeadTrackerConfig {
        max_history: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

// --- HeadTracker ---

#[test]
fn test_tracker_new_ok() {
    assert!(HeadTracker::new(HeadTrackerConfig::default()).is_ok());
}

#[test]
fn test_tracker_update_adds_to_history() {
    let mut tracker = HeadTracker::new(HeadTrackerConfig::default()).expect("ok");
    let f = make_frame(0, 0.0, 0.1, 0.0, 0.0);
    tracker.update(f);
    assert_eq!(tracker.history_len(), 1);
}

#[test]
fn test_tracker_current() {
    let mut tracker = HeadTracker::new(HeadTrackerConfig::default()).expect("ok");
    assert!(tracker.current().is_none());
    tracker.update(make_frame(0, 0.0, 0.1, 0.0, 0.0));
    assert!(tracker.current().is_some());
}

#[test]
fn test_tracker_filtered_current() {
    let cfg = HeadTrackerConfig {
        filter: TrackingFilter::ExponentialMovingAverage { alpha: 0.5 },
        ..Default::default()
    };
    let mut tracker = HeadTracker::new(cfg).expect("ok");
    tracker.update(make_frame(0, 0.0, 1.0, 0.0, 0.0));
    tracker.update(make_frame(1, 33.0, 0.0, 0.0, 0.0));
    let fc = tracker.filtered_current().expect("some");
    // EMA: second step yaw = 0.5*0 + 0.5*1 = 0.5
    assert!((fc.yaw - 0.5).abs() < 1e-5);
}

#[test]
fn test_tracker_reset() {
    let mut tracker = HeadTracker::new(HeadTrackerConfig::default()).expect("ok");
    tracker.update(make_frame(0, 0.0, 0.0, 0.0, 0.0));
    tracker.reset();
    assert_eq!(tracker.history_len(), 0);
    assert!(tracker.current().is_none());
}

#[test]
fn test_tracker_valid_fraction() {
    let mut tracker = HeadTracker::new(HeadTrackerConfig::default()).expect("ok");
    tracker.update(make_frame(0, 0.0, 0.1, 0.0, 0.0));
    tracker.update(HeadTrackFrame::invalid(1, 33.0));
    assert!((tracker.valid_fraction() - 0.5).abs() < 1e-5);
}

#[test]
fn test_tracker_max_history_trim() {
    let cfg = HeadTrackerConfig {
        max_history: 3,
        ..Default::default()
    };
    let mut tracker = HeadTracker::new(cfg).expect("ok");
    for i in 0..10usize {
        tracker.update(make_frame(i, i as f32 * 33.0, 0.0, 0.0, 0.0));
    }
    assert_eq!(tracker.history_len(), 3);
}

#[test]
fn test_tracker_confidence_and_outlier_gating() {
    let mut tracker = HeadTracker::new(HeadTrackerConfig::default()).expect("ok");

    // Low confidence: `is_valid` must be downgraded even though
    // `HeadTrackFrame::new` sets it `true`.
    let low_conf = HeadTrackFrame::new(
        0,
        0.0,
        HeadTrackPose {
            confidence: 0.05,
            ..Default::default()
        },
    );
    assert!(!tracker.update(low_conf).is_valid);

    // First confident frame becomes the outlier-detection baseline.
    tracker.update(make_frame(1, 33.0, 0.0, 0.0, 0.0));

    // Huge yaw jump exceeds the default outlier_threshold (PI/6 rad):
    // must be downgraded to invalid rather than accepted as genuine
    // frame-to-frame motion.
    assert!(!tracker.update(make_frame(2, 66.0, 3.0, 0.0, 0.0)).is_valid);

    // A small jump from the *last accepted* baseline (frame 1, not the
    // rejected frame 2) must be accepted.
    assert!(tracker.update(make_frame(3, 99.0, 0.02, 0.0, 0.0)).is_valid);
}

#[test]
fn test_tracker_interpolate_gaps_honours_max_gap_frames() {
    let cfg = HeadTrackerConfig {
        max_gap_frames: 2,
        ..Default::default()
    };
    let mut tracker = HeadTracker::new(cfg).expect("ok");
    let low_conf = |idx: usize, ts: f32| {
        HeadTrackFrame::new(
            idx,
            ts,
            HeadTrackPose {
                confidence: 0.05,
                ..Default::default()
            },
        )
    };
    // Gap of length 2 (<= max_gap_frames): should be interpolated.
    tracker.update(make_frame(0, 0.0, 0.0, 0.0, 0.0));
    tracker.update(low_conf(1, 33.0));
    tracker.update(low_conf(2, 66.0));
    tracker.update(make_frame(3, 99.0, 0.0, 0.0, 0.0));
    // Gap of length 3 (> max_gap_frames): should be left invalid.
    tracker.update(low_conf(4, 132.0));
    tracker.update(low_conf(5, 165.0));
    tracker.update(low_conf(6, 198.0));
    tracker.update(make_frame(7, 231.0, 0.0, 0.0, 0.0));

    let filled = tracker.interpolate_gaps().expect("ok");
    assert!(filled.frames[1].is_valid && filled.frames[2].is_valid);
    assert!(!filled.frames[4].is_valid && !filled.frames[5].is_valid && !filled.frames[6].is_valid);
}

#[test]
fn test_tracker_one_euro_filters_all_six_channels() {
    let cfg = HeadTrackerConfig {
        filter: TrackingFilter::OneEuro {
            min_cutoff: 1.0,
            beta: 0.0,
        },
        ..Default::default()
    };
    let mut tracker = HeadTracker::new(cfg).expect("ok");
    let pose_a = HeadTrackPose {
        yaw: 1.0,
        pitch: 1.0,
        roll: 1.0,
        tx: 1.0,
        ty: 1.0,
        tz: 1.0,
        confidence: 0.9,
    };
    let pose_b = HeadTrackPose {
        confidence: 0.9,
        ..Default::default()
    };
    tracker.update(HeadTrackFrame::new(0, 0.0, pose_a));
    let f = tracker.update(HeadTrackFrame::new(1, 33.0, pose_b));
    // Every channel must land strictly between the old (1.0) and new
    // (0.0) raw values -- the pre-fix filter passed pitch/roll/tx/ty/tz
    // through unsmoothed (i.e. straight to 0.0).
    for (name, v) in [
        ("pitch", f.pitch),
        ("roll", f.roll),
        ("tx", f.tx),
        ("ty", f.ty),
        ("tz", f.tz),
    ] {
        assert!(v > 0.0 && v < 1.0, "{name} should be smoothed, got {v}");
    }
}

#[test]
fn test_tracker_one_euro_uses_real_frame_interval() {
    // A larger real dt must let the filter track a new sample more
    // closely than a small dt does; a fixed 33 ms assumption would
    // make both cases identical.
    let make_tracker = || {
        HeadTracker::new(HeadTrackerConfig {
            filter: TrackingFilter::OneEuro {
                min_cutoff: 1.0,
                beta: 0.0,
            },
            ..Default::default()
        })
        .expect("ok")
    };

    let mut near = make_tracker();
    near.update(make_frame(0, 0.0, 1.0, 0.0, 0.0));
    let near_out = near.update(make_frame(1, 33.0, 0.0, 0.0, 0.0)).yaw;

    let mut far = make_tracker();
    far.update(make_frame(0, 0.0, 1.0, 0.0, 0.0));
    let far_out = far.update(make_frame(1, 1000.0, 0.0, 0.0, 0.0)).yaw;

    assert!(
        far_out < near_out,
        "near={near_out}, far={far_out} (far should track the new sample more closely)"
    );
}

#[test]
fn test_tracker_sma_averages_exactly_window_raw_samples() {
    let cfg = HeadTrackerConfig {
        filter: TrackingFilter::SimpleMovingAverage { window: 3 },
        ..Default::default()
    };
    let mut tracker = HeadTracker::new(cfg).expect("ok");
    tracker.update(make_frame(0, 0.0, 0.0, 0.0, 0.0));
    tracker.update(make_frame(1, 33.0, 3.0, 0.0, 0.0));
    let out = tracker.update(make_frame(2, 66.0, 6.0, 0.0, 0.0));
    // Mean of the 3 raw samples (0, 3, 6) = 3.0. The pre-fix formula
    // averaged `window + 1` = 4 already-smoothed samples instead.
    assert!((out.yaw - 3.0).abs() < 1e-5, "got {}", out.yaw);
}

#[test]
fn test_tracker_sma_window_one_is_identity() {
    let cfg = HeadTrackerConfig {
        filter: TrackingFilter::SimpleMovingAverage { window: 1 },
        ..Default::default()
    };
    let mut tracker = HeadTracker::new(cfg).expect("ok");
    tracker.update(make_frame(0, 0.0, 1.0, 0.0, 0.0));
    let out = tracker.update(make_frame(1, 33.0, 5.0, 0.0, 0.0));
    assert!(
        (out.yaw - 5.0).abs() < 1e-5,
        "window=1 must reproduce the raw frame, got {}",
        out.yaw
    );
}

// --- HeadTrajectory ---

#[test]
fn test_trajectory_empty_error() {
    assert!(HeadTrajectory::new(vec![]).is_err());
}

#[test]
fn test_trajectory_duration_ms() {
    let traj = simple_trajectory(10);
    let expected = 9.0 * 33.0;
    assert!((traj.duration_ms() - expected).abs() < 1e-3);
}

#[test]
fn test_trajectory_fps() {
    let traj = simple_trajectory(31);
    // 30 intervals * 33ms = 990ms; 30/0.99s ≈ 30.3
    let fps = traj.fps();
    assert!(fps > 25.0 && fps < 35.0);
}

#[test]
fn test_trajectory_yaw_range_only_valid() {
    let mut frames: Vec<HeadTrackFrame> = (0..5)
        .map(|i| make_frame(i, i as f32 * 33.0, i as f32 * 0.1, 0.0, 0.0))
        .collect();
    // Make frame 0 invalid (yaw=0 but not counted)
    frames[0] = HeadTrackFrame::invalid(0, 0.0);
    let traj = HeadTrajectory::new(frames).expect("ok");
    let (min, max) = traj.yaw_range();
    // Valid frames: idx 1..4, yaw 0.1..0.4
    assert!((min - 0.1).abs() < 1e-5);
    assert!((max - 0.4).abs() < 1e-5);
}

#[test]
fn test_trajectory_coverage() {
    let frames = vec![
        make_frame(0, 0.0, 0.0, 0.0, 0.0),
        HeadTrackFrame::invalid(1, 33.0),
    ];
    let traj = HeadTrajectory::new(frames).expect("ok");
    assert!((traj.coverage() - 0.5).abs() < 1e-5);
}

// --- ema_smooth_trajectory ---

#[test]
fn test_ema_alpha1_identical() {
    let traj = simple_trajectory(5);
    let smoothed = ema_smooth_trajectory(&traj, 1.0).expect("ok");
    for (a, b) in traj.frames.iter().zip(smoothed.frames.iter()) {
        assert!((a.yaw - b.yaw).abs() < 1e-6);
    }
}

#[test]
fn test_ema_alpha05_smoothed() {
    let frames: Vec<HeadTrackFrame> = vec![
        make_frame(0, 0.0, 0.0, 0.0, 0.0),
        make_frame(1, 33.0, 1.0, 0.0, 0.0),
        make_frame(2, 66.0, 0.0, 0.0, 0.0),
    ];
    let traj = HeadTrajectory::new(frames).expect("ok");
    let smoothed = ema_smooth_trajectory(&traj, 0.5).expect("ok");
    // Frame 1: 0.5*1 + 0.5*0 = 0.5
    assert!((smoothed.frames[1].yaw - 0.5).abs() < 1e-5);
    // Frame 2: 0.5*0 + 0.5*0.5 = 0.25
    assert!((smoothed.frames[2].yaw - 0.25).abs() < 1e-5);
}

// --- sma_smooth_trajectory ---

#[test]
fn test_sma_window1_identical() {
    let traj = simple_trajectory(5);
    let smoothed = sma_smooth_trajectory(&traj, 1).expect("ok");
    for (a, b) in traj.frames.iter().zip(smoothed.frames.iter()) {
        assert!((a.yaw - b.yaw).abs() < 1e-6);
    }
}

#[test]
fn test_sma_window2_averaged() {
    let frames: Vec<HeadTrackFrame> = vec![
        make_frame(0, 0.0, 0.0, 0.0, 0.0),
        make_frame(1, 33.0, 1.0, 0.0, 0.0),
    ];
    let traj = HeadTrajectory::new(frames).expect("ok");
    let smoothed = sma_smooth_trajectory(&traj, 2).expect("ok");
    // Frame 0: only itself → 0.0
    assert!((smoothed.frames[0].yaw - 0.0).abs() < 1e-5);
    // Frame 1: (0+1)/2 = 0.5
    assert!((smoothed.frames[1].yaw - 0.5).abs() < 1e-5);
}

// --- one_euro_filter_sequence ---

#[test]
fn test_one_euro_same_length_output() {
    let vals = vec![0.0f32, 0.1, 0.2, 0.1, 0.0];
    let ts: Vec<f32> = (0..5).map(|i| i as f32 * 33.0).collect();
    let out = one_euro_filter_sequence(&vals, &ts, 1.0, 0.0).expect("ok");
    assert_eq!(out.len(), vals.len());
}

#[test]
fn test_one_euro_mismatched_lengths_error() {
    let vals = vec![0.0f32, 0.1, 0.2];
    let ts = vec![0.0f32, 33.0]; // shorter
    assert!(one_euro_filter_sequence(&vals, &ts, 1.0, 0.0).is_err());
}

#[test]
fn test_one_euro_first_sample_unchanged() {
    let vals = vec![0.42f32, 0.1];
    let ts = vec![0.0f32, 33.0];
    let out = one_euro_filter_sequence(&vals, &ts, 1.0, 0.0).expect("ok");
    assert!((out[0] - 0.42).abs() < 1e-5);
}

// --- interpolate_missing_frames ---

#[test]
fn test_interpolate_all_valid_unchanged() {
    let traj = simple_trajectory(4);
    let out = interpolate_missing_frames(&traj, usize::MAX).expect("ok");
    for (a, b) in traj.frames.iter().zip(out.frames.iter()) {
        assert!((a.yaw - b.yaw).abs() < 1e-6);
    }
}

#[test]
fn test_interpolate_one_missing() {
    let frames = vec![
        make_frame(0, 0.0, 0.0, 0.0, 0.0),
        HeadTrackFrame::invalid(1, 33.0),
        make_frame(2, 66.0, 1.0, 0.0, 0.0),
    ];
    let traj = HeadTrajectory::new(frames).expect("ok");
    let out = interpolate_missing_frames(&traj, usize::MAX).expect("ok");
    assert!(out.frames[1].is_valid);
    // t = 1/2 of the span → yaw = 0.5
    assert!((out.frames[1].yaw - 0.5).abs() < 1e-5);
}

#[test]
fn test_interpolate_all_invalid_stays_invalid() {
    // No valid frame anywhere: must not be reported as a fully
    // tracked, perfectly still trajectory.
    let frames = vec![
        HeadTrackFrame::invalid(0, 0.0),
        HeadTrackFrame::invalid(1, 33.0),
        HeadTrackFrame::invalid(2, 66.0),
    ];
    let traj = HeadTrajectory::new(frames).expect("ok");
    let out = interpolate_missing_frames(&traj, usize::MAX).expect("ok");
    assert!(out.frames.iter().all(|f| !f.is_valid));
}

#[test]
fn test_interpolate_respects_max_gap_frames() {
    let mut frames = vec![make_frame(0, 0.0, 0.0, 0.0, 0.0)];
    for i in 1..=3 {
        frames.push(HeadTrackFrame::invalid(i, i as f32 * 33.0));
    }
    frames.push(make_frame(4, 132.0, 1.0, 0.0, 0.0));
    let traj = HeadTrajectory::new(frames).expect("ok");

    // Gap length 3 > max_gap_frames 2: left invalid.
    let capped = interpolate_missing_frames(&traj, 2).expect("ok");
    assert!(capped.frames[1..4].iter().all(|f| !f.is_valid));

    // Same gap, generous cap: filled in.
    let filled = interpolate_missing_frames(&traj, 3).expect("ok");
    assert!(filled.frames[1..4].iter().all(|f| f.is_valid));
}

// --- detect_pose_jumps ---

#[test]
fn test_no_jumps() {
    let traj = simple_trajectory(5);
    let jumps = detect_pose_jumps(&traj, 1.0); // large threshold
    assert!(jumps.is_empty());
}

#[test]
fn test_one_large_jump_detected() {
    let frames = vec![
        make_frame(0, 0.0, 0.0, 0.0, 0.0),
        make_frame(1, 33.0, 2.0, 0.0, 0.0), // Δ = 2.0 rad
    ];
    let traj = HeadTrajectory::new(frames).expect("ok");
    let jumps = detect_pose_jumps(&traj, 1.0);
    assert_eq!(jumps, vec![1]);
}

// --- rotation_velocity ---

#[test]
fn test_rotation_velocity_single_frame_error() {
    let traj = HeadTrajectory::new(vec![make_frame(0, 0.0, 0.0, 0.0, 0.0)]).expect("ok");
    assert!(rotation_velocity(&traj).is_err());
}

#[test]
fn test_rotation_velocity_two_frames() {
    let frames = vec![
        make_frame(0, 0.0, 0.0, 0.0, 0.0),
        make_frame(1, 100.0, 1.0, 0.0, 0.0), // Δ=1 rad over 100ms
    ];
    let traj = HeadTrajectory::new(frames).expect("ok");
    let vels = rotation_velocity(&traj).expect("ok");
    assert_eq!(vels.len(), 1);
    assert!((vels[0] - 0.01).abs() < 1e-5); // 1/100
}

// --- compute_trajectory_stats ---

#[test]
fn test_stats_empty_error() {
    // Can't construct empty trajectory, so test indirectly via manual hack
    // Instead, test that a valid trajectory gives non-error stats
    let traj = simple_trajectory(5);
    assert!(compute_trajectory_stats(&traj, 1.0).is_ok());
}

#[test]
fn test_stats_valid_fields() {
    let traj = simple_trajectory(10);
    let stats = compute_trajectory_stats(&traj, 1.0).expect("ok");
    assert_eq!(stats.total_frames, 10);
    assert_eq!(stats.valid_frames, 10);
    assert!(stats.duration_ms > 0.0);
    assert!(stats.mean_confidence > 0.0);
}

// --- segment_by_motion ---

#[test]
fn test_segment_still_trajectory() {
    // All same pose → velocity = 0 → all still
    let frames: Vec<HeadTrackFrame> = (0..5)
        .map(|i| make_frame(i, i as f32 * 33.0, 0.0, 0.0, 0.0))
        .collect();
    let traj = HeadTrajectory::new(frames).expect("ok");
    let segs = segment_by_motion(&traj, 0.01).expect("ok");
    // All segments should be still
    for (_, _, is_motion) in &segs {
        assert!(!is_motion);
    }
}

#[test]
fn test_segment_single_frame_error() {
    let traj = HeadTrajectory::new(vec![make_frame(0, 0.0, 0.0, 0.0, 0.0)]).expect("ok");
    assert!(segment_by_motion(&traj, 0.01).is_err());
}

#[test]
fn test_segment_by_motion_indices_are_consistent_frame_positions() {
    // Alternating jump / flat runs so multiple transitions occur.
    let frames = vec![
        make_frame(0, 0.0, 0.0, 0.0, 0.0),
        make_frame(1, 33.0, 5.0, 0.0, 0.0),   // big jump: motion
        make_frame(2, 66.0, 5.0, 0.0, 0.0),   // flat: still
        make_frame(3, 99.0, 5.0, 0.0, 0.0),   // flat: still
        make_frame(4, 132.0, 12.0, 0.0, 0.0), // big jump: motion
        make_frame(5, 165.0, 12.0, 0.0, 0.0), // flat: still
    ];
    let traj = HeadTrajectory::new(frames).expect("ok");
    let segs = segment_by_motion(&traj, 0.01).expect("ok");

    assert!(segs.len() >= 2, "expected multiple segments, got {segs:?}");
    assert_eq!(
        segs.first().expect("non-empty").0,
        0,
        "first segment starts at frame 0"
    );
    assert_eq!(
        segs.last().expect("non-empty").1,
        traj.len() - 1,
        "last segment ends at the final frame index"
    );
    for w in segs.windows(2) {
        assert_eq!(
            w[0].1, w[1].0,
            "consecutive segments share their boundary frame"
        );
        assert!(w[0].1 > w[0].0, "every segment spans at least one frame");
    }
    assert!(
        segs.iter().any(|&(_, _, m)| m) && segs.iter().any(|&(_, _, m)| !m),
        "expected both motion and still segments, got {segs:?}"
    );
}

// --- head_coverage_score ---

#[test]
fn test_coverage_all_same_pose_low_score() {
    let frames: Vec<HeadTrackFrame> = (0..10)
        .map(|i| make_frame(i, i as f32 * 33.0, 0.0, 0.0, 0.0))
        .collect();
    let traj = HeadTrajectory::new(frames).expect("ok");
    // All same pose → all land in one bin → 1/(4*4) = 0.0625
    let score = head_coverage_score(&traj, 4, 4).expect("ok");
    assert!(score <= 0.1);
}

#[test]
fn test_coverage_varied_poses_higher_score() {
    let frames: Vec<HeadTrackFrame> = (0..16)
        .map(|i| {
            let yaw = (i as f32) * 0.1;
            let pitch = (i % 4) as f32 * 0.1;
            make_frame(i, i as f32 * 33.0, yaw, pitch, 0.0)
        })
        .collect();
    let traj = HeadTrajectory::new(frames).expect("ok");
    let score = head_coverage_score(&traj, 4, 4).expect("ok");
    assert!(score > 0.1);
}

// --- slice_trajectory ---

#[test]
fn test_slice_valid_range() {
    let traj = simple_trajectory(10);
    let sliced = slice_trajectory(&traj, 2, 5).expect("ok");
    assert_eq!(sliced.len(), 4); // frames 2,3,4,5
}

#[test]
fn test_slice_out_of_range_error() {
    let traj = simple_trajectory(5);
    assert!(slice_trajectory(&traj, 0, 10).is_err());
}

// --- resample_trajectory ---

#[test]
fn test_resample_same_fps_approx_same_count() {
    let traj = simple_trajectory(31); // ~30 fps over ~1 second
    let resampled = resample_trajectory(&traj, 30.0).expect("ok");
    // Should have approximately the same number of frames
    let n = resampled.len();
    assert!((25..=40).contains(&n), "got {n} frames");
}

#[test]
fn test_resample_invalid_fps_error() {
    let traj = simple_trajectory(5);
    assert!(resample_trajectory(&traj, 0.0).is_err());
    assert!(resample_trajectory(&traj, -1.0).is_err());
}

#[test]
fn test_resample_single_frame_error() {
    let traj = HeadTrajectory::new(vec![make_frame(0, 0.0, 0.0, 0.0, 0.0)]).expect("ok");
    assert!(resample_trajectory(&traj, 30.0).is_err());
}
