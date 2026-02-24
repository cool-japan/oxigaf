//! Integration tests for interactive mode functionality.

use assert_cmd::Command;
use std::sync::atomic::Ordering;

#[test]
fn interactive_controller_default_state() {
    use oxigaf_cli::InteractiveController;

    let controller = InteractiveController::new();

    // Verify all atomic flags start in the correct state
    assert!(
        !controller.paused.load(Ordering::Relaxed),
        "Controller should not be paused initially"
    );
    assert_eq!(
        controller.lr_adjustment.load(Ordering::Relaxed),
        100,
        "Learning rate adjustment should be 100 (1.0x) initially"
    );
    assert!(
        !controller.verbose_toggle.load(Ordering::Relaxed),
        "Verbose should be off initially"
    );
    assert!(
        !controller.save_requested.load(Ordering::Relaxed),
        "Save should not be requested initially"
    );
    assert!(
        !controller.quit_requested.load(Ordering::Relaxed),
        "Quit should not be requested initially"
    );
}

#[test]
fn interactive_controller_pause_toggle() {
    use oxigaf_cli::InteractiveController;

    let controller = InteractiveController::new();

    // Test pause toggle
    let was_paused = controller.paused.fetch_xor(true, Ordering::Relaxed);
    assert!(!was_paused, "Should not be paused initially");

    let is_paused = controller.paused.load(Ordering::Relaxed);
    assert!(is_paused, "Should be paused after toggle");

    // Toggle back
    controller.paused.fetch_xor(true, Ordering::Relaxed);
    let is_paused = controller.paused.load(Ordering::Relaxed);
    assert!(!is_paused, "Should not be paused after second toggle");
}

#[test]
fn interactive_controller_lr_adjustment_bounds() {
    use oxigaf_cli::InteractiveController;

    let controller = InteractiveController::new();

    // Test upper bound (200 = 2.0x)
    controller.lr_adjustment.store(195, Ordering::Relaxed);
    let current = controller.lr_adjustment.load(Ordering::Relaxed);
    let new = current.saturating_add(10).min(200);
    controller.lr_adjustment.store(new, Ordering::Relaxed);
    assert_eq!(
        controller.lr_adjustment.load(Ordering::Relaxed),
        200,
        "LR adjustment should be capped at 200"
    );

    // Test lower bound (10 = 0.1x)
    controller.lr_adjustment.store(15, Ordering::Relaxed);
    let current = controller.lr_adjustment.load(Ordering::Relaxed);
    let new = current.saturating_sub(10).max(10);
    controller.lr_adjustment.store(new, Ordering::Relaxed);
    assert_eq!(
        controller.lr_adjustment.load(Ordering::Relaxed),
        10,
        "LR adjustment should be capped at 10"
    );
}

#[test]
fn interactive_controller_save_request() {
    use oxigaf_cli::InteractiveController;

    let controller = InteractiveController::new();

    // Request save
    controller.save_requested.store(true, Ordering::Relaxed);
    assert!(
        controller.save_requested.load(Ordering::Relaxed),
        "Save should be requested"
    );

    // Simulate consumption with swap
    let was_requested = controller.save_requested.swap(false, Ordering::Relaxed);
    assert!(was_requested, "Save request should have been set");
    assert!(
        !controller.save_requested.load(Ordering::Relaxed),
        "Save request should be cleared after swap"
    );
}

#[test]
#[allow(deprecated)]
fn interactive_flag_accepted() {
    // Test that the CLI accepts the --interactive flag
    Command::cargo_bin("oxigaf")
        .unwrap() // Safe: test will fail if binary not found, which is expected
        .args(["train", "--interactive", "--help"])
        .assert()
        .success();
}

#[test]
fn interactive_controller_thread_safety() {
    use oxigaf_cli::InteractiveController;
    use std::sync::Arc;
    use std::thread;

    let controller = Arc::new(InteractiveController::new());

    // Spawn multiple threads that modify the controller state
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let ctrl = Arc::clone(&controller);
            thread::spawn(move || {
                // Toggle pause
                ctrl.paused.fetch_xor(true, Ordering::Relaxed);

                // Adjust LR
                if i % 2 == 0 {
                    let current = ctrl.lr_adjustment.load(Ordering::Relaxed);
                    let new = current.saturating_add(10).min(200);
                    ctrl.lr_adjustment.store(new, Ordering::Relaxed);
                } else {
                    let current = ctrl.lr_adjustment.load(Ordering::Relaxed);
                    let new = current.saturating_sub(10).max(10);
                    ctrl.lr_adjustment.store(new, Ordering::Relaxed);
                }

                // Toggle verbose
                ctrl.verbose_toggle.fetch_xor(true, Ordering::Relaxed);
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap_or_else(|_| panic!("Thread panicked")); // Safe: test failure on panic
    }

    // Verify the controller is still in a valid state
    let lr_adj = controller.lr_adjustment.load(Ordering::Relaxed);
    assert!(
        (10..=200).contains(&lr_adj),
        "LR adjustment should be within valid range after concurrent modifications"
    );
}
