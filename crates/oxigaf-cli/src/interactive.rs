//! Interactive mode for live parameter adjustment during training.
//!
//! Provides keyboard controls to pause/resume training, adjust learning rates,
//! toggle verbose logging, save checkpoints, and quit gracefully.

use crossterm::{
    event::{self, Event, KeyCode, KeyEvent},
    terminal::{disable_raw_mode, enable_raw_mode},
};
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
    /// Pause/resume training
    pub paused: Arc<AtomicBool>,
    /// Learning rate adjustment multiplier (100 = 1.0x, range: 10-200)
    pub lr_adjustment: Arc<AtomicU8>,
    /// Toggle verbose logging
    pub verbose_toggle: Arc<AtomicBool>,
    /// Request checkpoint save
    pub save_requested: Arc<AtomicBool>,
    /// Request graceful quit
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
    /// This enables terminal raw mode. Make sure to call
    /// `crossterm::terminal::disable_raw_mode()` on cleanup.
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
                            Ok(Event::Key(key_event)) => {
                                handle_key_event(
                                    key_event, &paused, &lr_adj, &verbose, &save, &quit,
                                );
                            }
                            Ok(_) => {} // Ignore other events
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
    pub fn print_controls(&self) {
        println!();
        println!("Interactive Controls:");
        println!("  [Space]    Pause/Resume training");
        println!("  [Up/Down]  Increase/Decrease learning rate");
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
                println!("\nTraining paused");
            } else {
                println!("\nTraining resumed");
            }
        }
        KeyCode::Up => {
            // Increase learning rate (max 2.0x)
            let current = lr_adj.load(Ordering::Relaxed);
            let new = current.saturating_add(10).min(200);
            lr_adj.store(new, Ordering::Relaxed);
            println!("\nLearning rate: {:.1}x", f64::from(new) / 100.0);
        }
        KeyCode::Down => {
            // Decrease learning rate (min 0.1x)
            let current = lr_adj.load(Ordering::Relaxed);
            let new = current.saturating_sub(10).max(10);
            lr_adj.store(new, Ordering::Relaxed);
            println!("\nLearning rate: {:.1}x", f64::from(new) / 100.0);
        }
        KeyCode::Char('v') | KeyCode::Char('V') => {
            // Toggle verbose logging
            let was_verbose = verbose.fetch_xor(true, Ordering::Relaxed);
            let now_verbose = !was_verbose;
            println!("\nVerbose: {}", if now_verbose { "ON" } else { "OFF" });
        }
        KeyCode::Char('s') | KeyCode::Char('S') => {
            // Request checkpoint save
            save.store(true, Ordering::Relaxed);
            println!("\nCheckpoint save requested...");
        }
        KeyCode::Char('q') | KeyCode::Char('Q') => {
            // Request graceful quit
            quit.store(true, Ordering::Relaxed);
            println!("\nGraceful shutdown requested...");
        }
        _ => {} // Ignore other keys
    }
}
