//! Interactive mode for live parameter adjustment during training.
//!
//! Provides keyboard controls to pause/resume training, adjust learning rates,
//! toggle verbose logging, save checkpoints, and quit gracefully.

use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode},
};
use std::io::Write;
use std::sync::{
    atomic::{AtomicBool, AtomicU8, Ordering},
    Arc,
};
use std::time::Duration;

/// Controller for interactive training mode with keyboard controls.
///
/// Manages shared state between the keyboard listener thread and the main
/// training loop using atomic operations for thread-safe communication.
pub struct InteractiveController {
    /// Pause/resume training. Read by the training loop's step gate.
    pub paused: Arc<AtomicBool>,
    /// Learning rate adjustment multiplier (100 = 1.0x, range: 10-200),
    /// updated by the `[Up]`/`[Down]` keys.
    ///
    /// Public and ready for a training loop to read, but **nothing applies
    /// it yet**: [`oxigaf_trainer::GaussianOptimizer`] clones its
    /// `OptimizerConfig` into a private field with no setter, so the per-step
    /// learning rates cannot be scaled from outside the trainer. Mutating
    /// `Trainer::config.optimizer` does not help — the optimiser never reads
    /// it again after construction. Wiring this up needs a
    /// `set_lr_scale`/`config_mut` accessor on `GaussianOptimizer` first; see
    /// [`InteractiveController::print_controls`].
    pub lr_adjustment: Arc<AtomicU8>,
    /// Toggle verbose logging, flipped by the `[v]` key.
    ///
    /// Read once per training step by `crate::pipeline::run_reconstruction`:
    /// while set, every iteration is logged (whatever `log_interval` says,
    /// including `0`) together with the individual loss terms.
    pub verbose_toggle: Arc<AtomicBool>,
    /// Request checkpoint save. Read (and cleared) by the training loop.
    pub save_requested: Arc<AtomicBool>,
    /// Request graceful quit. Read by the training loop's step gate.
    pub quit_requested: Arc<AtomicBool>,
}

impl InteractiveController {
    /// Create a new interactive controller with default state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            paused: Arc::new(AtomicBool::new(false)),
            lr_adjustment: Arc::new(AtomicU8::new(100)), // 1.0x multiplier
            verbose_toggle: Arc::new(AtomicBool::new(false)),
            save_requested: Arc::new(AtomicBool::new(false)),
            quit_requested: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start the keyboard listener thread.
    ///
    /// Spawns a background thread that listens for keyboard events and updates
    /// the controller's atomic state. The thread runs until quit is requested.
    ///
    /// # Note
    ///
    /// This enables terminal raw mode. Raw mode is restored automatically
    /// when this `InteractiveController` is dropped (see the `Drop` impl) —
    /// on every exit path (training finishes normally, an error propagates,
    /// or the user presses `q`), not only the quit key. Callers do not need
    /// to call `crossterm::terminal::disable_raw_mode()` themselves as long
    /// as the controller is held (not leaked) for the interactive session.
    pub fn start_keyboard_listener(&self) {
        let paused = Arc::clone(&self.paused);
        let lr_adj = Arc::clone(&self.lr_adjustment);
        let verbose = Arc::clone(&self.verbose_toggle);
        let save = Arc::clone(&self.save_requested);
        let quit = Arc::clone(&self.quit_requested);

        std::thread::spawn(move || {
            // Try to enable raw mode
            if enable_raw_mode().is_err() {
                tracing::error!("Failed to enable terminal raw mode for interactive controls");
                return;
            }

            loop {
                // Poll for events with timeout
                match event::poll(Duration::from_millis(100)) {
                    Ok(true) => {
                        // Event available, read it
                        match event::read() {
                            // Only react to the initial key-down. Without this
                            // filter, Windows (and Unix terminals that opted
                            // into the Kitty keyboard protocol's
                            // `REPORT_EVENT_TYPES`) deliver a `Press` *and* a
                            // `Release` event per physical keypress, so every
                            // handler would run twice — most visibly, the
                            // `fetch_xor`-based toggles (`paused`,
                            // `verbose_toggle`) would flip on Press and flip
                            // right back on Release, silently cancelling out.
                            Ok(Event::Key(key_event)) if key_event.kind == KeyEventKind::Press => {
                                handle_key_event(
                                    key_event, &paused, &lr_adj, &verbose, &save, &quit,
                                );
                            }
                            Ok(_) => {} // Ignore other events (and non-Press key events)
                            Err(e) => {
                                tracing::error!("Failed to read keyboard event: {}", e);
                            }
                        }
                    }
                    Ok(false) => {} // No event available
                    Err(e) => {
                        tracing::error!("Failed to poll for keyboard events: {}", e);
                    }
                }

                // Check quit flag
                if quit.load(Ordering::Relaxed) {
                    let _ = disable_raw_mode();
                    break;
                }
            }
        });
    }

    /// Print the interactive controls help message.
    ///
    /// `[Space]`, `[v]`, `[s]` and `[q]`
    /// (`paused`/`verbose_toggle`/`save_requested`/`quit_requested`) are all
    /// consumed by `pipeline.rs`'s training loop and take effect on the run
    /// in progress.
    ///
    /// `[Up/Down]` is still called out as "recorded" rather than claimed
    /// outright: pressing it updates [`lr_adjustment`](Self::lr_adjustment)
    /// and prints a confirmation, but no training loop can act on it yet —
    /// see that field's documentation for the trainer-side accessor it is
    /// waiting on. Labelling it as working would be the worse failure: the
    /// user would believe they had rescued a diverging run.
    pub fn print_controls(&self) {
        println!();
        println!("Interactive Controls:");
        println!("  [Space]    Pause/Resume training");
        println!("  [Up/Down]  Increase/Decrease learning rate (recorded; trainer wiring pending)");
        println!("  [v]        Toggle verbose logging");
        println!("  [s]        Save checkpoint now");
        println!("  [q]        Quit gracefully");
        println!();
    }
}

impl Default for InteractiveController {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for InteractiveController {
    /// Restore the terminal on every exit path, not just the quit key.
    ///
    /// [`start_keyboard_listener`](Self::start_keyboard_listener) spawns a
    /// **detached** background thread (no `JoinHandle` is retained) whose
    /// loop previously called `disable_raw_mode()` only from the branch that
    /// runs when the user presses `q`. If training instead finished
    /// normally, returned an error, or the process was torn down for any
    /// other reason, that thread was simply killed along with the rest of
    /// the process mid-loop, and the terminal was left in raw mode (no
    /// local echo, no line editing) after `oxigaf` exited — the user's
    /// shell looked broken until they blindly typed `reset` or `stty
    /// sane`. `InteractiveController` is held by its owner for exactly the
    /// interactive session's scope (constructed once per `oxigaf train
    /// --interactive` invocation, never cloned), so restoring here runs
    /// reliably whenever that scope ends: success, an error propagated with
    /// `?`, or an unwinding panic all drop it exactly once.
    ///
    /// Also requests the listener thread stop polling (it notices within
    /// its ~100ms poll interval) so an orphaned thread does not keep
    /// running — and, on the `q`-key exit path specifically, does not race
    /// past this call still believing it owns raw-mode cleanup — if the
    /// caller does more work after the interactive session ends.
    fn drop(&mut self) {
        self.quit_requested.store(true, Ordering::Relaxed);
        // Errors are expected and ignored here (e.g. raw mode was never
        // entered because `enable_raw_mode()` failed in the listener
        // thread, or stdout is not a TTY at all in a test/CI harness) —
        // this is best-effort cleanup, matching the ignored `Result` the
        // listener thread's own `q`-key path already used.
        let _ = disable_raw_mode();
    }
}

/// Print a status line that stays readable while the terminal may be in
/// raw mode.
///
/// Raw mode (entered by [`InteractiveController::start_keyboard_listener`])
/// disables the terminal's output post-processing (`ONLCR`), so a bare `\n`
/// moves the cursor down a row without returning it to column 0 — each
/// subsequent status line staircased one column further right instead of
/// starting flush at the left margin. An explicit `\r\n`, plus an immediate
/// flush (a raw-mode TTY is not guaranteed to line-buffer the way a
/// cooked-mode one does), keeps these lines readable and prevents them from
/// colliding with whatever else (e.g. a progress bar) is drawing to the
/// same terminal.
fn raw_mode_println(msg: &str) {
    print!("\r\n{msg}\r\n");
    let _ = std::io::stdout().flush();
}

/// Handle keyboard events and update controller state.
fn handle_key_event(
    key: KeyEvent,
    paused: &Arc<AtomicBool>,
    lr_adj: &Arc<AtomicU8>,
    verbose: &Arc<AtomicBool>,
    save: &Arc<AtomicBool>,
    quit: &Arc<AtomicBool>,
) {
    match key.code {
        KeyCode::Char(' ') => {
            // Toggle pause state
            let was_paused = paused.fetch_xor(true, Ordering::Relaxed);
            let now_paused = !was_paused;
            if now_paused {
                raw_mode_println("Training paused");
            } else {
                raw_mode_println("Training resumed");
            }
        }
        KeyCode::Up => {
            // Increase learning rate (max 2.0x)
            let current = lr_adj.load(Ordering::Relaxed);
            let new = current.saturating_add(10).min(200);
            lr_adj.store(new, Ordering::Relaxed);
            raw_mode_println(&format!("Learning rate: {:.1}x", f64::from(new) / 100.0));
        }
        KeyCode::Down => {
            // Decrease learning rate (min 0.1x)
            let current = lr_adj.load(Ordering::Relaxed);
            let new = current.saturating_sub(10).max(10);
            lr_adj.store(new, Ordering::Relaxed);
            raw_mode_println(&format!("Learning rate: {:.1}x", f64::from(new) / 100.0));
        }
        KeyCode::Char('v') | KeyCode::Char('V') => {
            // Toggle verbose logging
            let was_verbose = verbose.fetch_xor(true, Ordering::Relaxed);
            let now_verbose = !was_verbose;
            raw_mode_println(&format!(
                "Verbose: {}",
                if now_verbose { "ON" } else { "OFF" }
            ));
        }
        KeyCode::Char('s') | KeyCode::Char('S') => {
            // Request checkpoint save
            save.store(true, Ordering::Relaxed);
            raw_mode_println("Checkpoint save requested...");
        }
        KeyCode::Char('q') | KeyCode::Char('Q') => {
            // Request graceful quit
            quit.store(true, Ordering::Relaxed);
            raw_mode_println("Graceful shutdown requested...");
        }
        _ => {} // Ignore other keys
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// Broader end-to-end/thread-safety coverage lives in
// `crates/oxigaf-cli/tests/interactive_tests.rs`; these are focused unit
// tests for the specific bugs fixed directly in this file — `handle_key_event`
// itself previously had zero coverage anywhere.

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyModifiers;

    fn key(code: KeyCode) -> KeyEvent {
        // `KeyEvent::new` defaults `kind` to `KeyEventKind::Press`, matching
        // what `start_keyboard_listener`'s filter now requires.
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    type Atomics = (
        Arc<AtomicBool>,
        Arc<AtomicU8>,
        Arc<AtomicBool>,
        Arc<AtomicBool>,
        Arc<AtomicBool>,
    );

    /// Fresh (paused, lr_adjustment, verbose, save, quit) atomics at the
    /// same defaults `InteractiveController::new` uses.
    fn fresh_atomics() -> Atomics {
        (
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicU8::new(100)),
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
        )
    }

    // -----------------------------------------------------------------------
    // InteractiveController::new / Default / Drop
    // -----------------------------------------------------------------------

    #[test]
    fn new_has_expected_defaults() {
        let ctrl = InteractiveController::new();
        assert!(!ctrl.paused.load(Ordering::Relaxed));
        assert_eq!(ctrl.lr_adjustment.load(Ordering::Relaxed), 100);
        assert!(!ctrl.verbose_toggle.load(Ordering::Relaxed));
        assert!(!ctrl.save_requested.load(Ordering::Relaxed));
        assert!(!ctrl.quit_requested.load(Ordering::Relaxed));
    }

    #[test]
    fn default_matches_new() {
        let ctrl = InteractiveController::default();
        assert_eq!(ctrl.lr_adjustment.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn drop_sets_quit_requested_and_does_not_panic() {
        // Regression: previously *only* the listener thread's own 'q'-key
        // branch ever called `disable_raw_mode()`; every other way of
        // ending an interactive session (training finishes normally, an
        // error propagates, the process is killed) left the terminal in
        // raw mode. `Drop` must fire reliably instead — clone the Arc
        // *before* dropping so its post-drop state is observable — and it
        // must not panic even though there is no real TTY in a test
        // harness (so the internal `disable_raw_mode()` call is expected to
        // fail; that failure must be swallowed, exactly like the listener
        // thread's own already-ignored `let _ = disable_raw_mode();`).
        let ctrl = InteractiveController::new();
        let quit = Arc::clone(&ctrl.quit_requested);
        assert!(!quit.load(Ordering::Relaxed));
        drop(ctrl);
        assert!(
            quit.load(Ordering::Relaxed),
            "Drop must request the listener thread stop"
        );
    }

    // -----------------------------------------------------------------------
    // handle_key_event: pure state-transition logic, no TTY required.
    // -----------------------------------------------------------------------

    #[test]
    fn space_toggles_paused() {
        let (paused, lr, verbose, save, quit) = fresh_atomics();
        handle_key_event(
            key(KeyCode::Char(' ')),
            &paused,
            &lr,
            &verbose,
            &save,
            &quit,
        );
        assert!(paused.load(Ordering::Relaxed), "first space should pause");
        handle_key_event(
            key(KeyCode::Char(' ')),
            &paused,
            &lr,
            &verbose,
            &save,
            &quit,
        );
        assert!(
            !paused.load(Ordering::Relaxed),
            "second space should resume"
        );
    }

    #[test]
    fn up_increases_lr_adjustment_and_caps_at_200() {
        let (paused, lr, verbose, save, quit) = fresh_atomics();
        lr.store(195, Ordering::Relaxed);
        handle_key_event(key(KeyCode::Up), &paused, &lr, &verbose, &save, &quit);
        assert_eq!(
            lr.load(Ordering::Relaxed),
            200,
            "should cap at 200, not wrap"
        );
        handle_key_event(key(KeyCode::Up), &paused, &lr, &verbose, &save, &quit);
        assert_eq!(lr.load(Ordering::Relaxed), 200, "should stay capped at 200");
    }

    #[test]
    fn down_decreases_lr_adjustment_and_floors_at_10() {
        let (paused, lr, verbose, save, quit) = fresh_atomics();
        lr.store(15, Ordering::Relaxed);
        handle_key_event(key(KeyCode::Down), &paused, &lr, &verbose, &save, &quit);
        assert_eq!(
            lr.load(Ordering::Relaxed),
            10,
            "should floor at 10, not underflow"
        );
        handle_key_event(key(KeyCode::Down), &paused, &lr, &verbose, &save, &quit);
        assert_eq!(
            lr.load(Ordering::Relaxed),
            10,
            "should stay floored at 10 (saturating_sub, no wraparound)"
        );
    }

    #[test]
    fn v_and_shift_v_toggle_verbose() {
        let (paused, lr, verbose, save, quit) = fresh_atomics();
        handle_key_event(
            key(KeyCode::Char('v')),
            &paused,
            &lr,
            &verbose,
            &save,
            &quit,
        );
        assert!(verbose.load(Ordering::Relaxed));
        handle_key_event(
            key(KeyCode::Char('V')),
            &paused,
            &lr,
            &verbose,
            &save,
            &quit,
        );
        assert!(!verbose.load(Ordering::Relaxed));
    }

    #[test]
    fn s_and_shift_s_request_save() {
        let (paused, lr, verbose, save, quit) = fresh_atomics();
        handle_key_event(
            key(KeyCode::Char('S')),
            &paused,
            &lr,
            &verbose,
            &save,
            &quit,
        );
        assert!(save.load(Ordering::Relaxed));
    }

    #[test]
    fn q_and_shift_q_request_quit() {
        let (paused, lr, verbose, save, quit) = fresh_atomics();
        handle_key_event(
            key(KeyCode::Char('q')),
            &paused,
            &lr,
            &verbose,
            &save,
            &quit,
        );
        assert!(quit.load(Ordering::Relaxed));
    }

    #[test]
    fn unrecognised_key_changes_nothing() {
        let (paused, lr, verbose, save, quit) = fresh_atomics();
        handle_key_event(
            key(KeyCode::Char('z')),
            &paused,
            &lr,
            &verbose,
            &save,
            &quit,
        );
        assert!(!paused.load(Ordering::Relaxed));
        assert_eq!(lr.load(Ordering::Relaxed), 100);
        assert!(!verbose.load(Ordering::Relaxed));
        assert!(!save.load(Ordering::Relaxed));
        assert!(!quit.load(Ordering::Relaxed));
    }

    // -----------------------------------------------------------------------
    // raw_mode_println: must not panic even without a real TTY.
    // -----------------------------------------------------------------------

    #[test]
    fn raw_mode_println_does_not_panic() {
        raw_mode_println("test message");
    }
}
