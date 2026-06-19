//! Preview controller — maps keyboard/mouse events to camera actions.
//!
//! Pure-Rust, no winit, no GPU dependency. Designed for use with any event
//! source that can produce [`KeyCode`], mouse button, and scroll events.

use crate::arcball::ArcballCamera;

// ---------------------------------------------------------------------------
// KeyCode
// ---------------------------------------------------------------------------

/// Simple key-code abstraction with no winit dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyCode {
    Q,
    Escape,
    R,
    S,
    Space,
    Right,
    Left,
    Up,
    Down,
    Plus,
    Minus,
    W,
    A,
    D,
    E,
    X,
    F1,
    F2,
    F3,
}

// ---------------------------------------------------------------------------
// CameraAction
// ---------------------------------------------------------------------------

/// Camera input actions derived from keyboard or mouse events.
#[derive(Debug, Clone, PartialEq)]
pub enum CameraAction {
    Orbit { delta_yaw: f32, delta_pitch: f32 },
    Dolly { delta: f32 },
    Pan { delta_x: f32, delta_y: f32 },
    Reset,
    Screenshot,
    Quit,
    ToggleAnimation,
    NextFrame,
    PreviousFrame,
    SpeedUp,
    SlowDown,
}

// ---------------------------------------------------------------------------
// KeyBindings
// ---------------------------------------------------------------------------

/// Key binding configuration for preview controls.
pub struct KeyBindings {
    pub quit_keys: Vec<KeyCode>,
    pub reset_key: KeyCode,
    pub screenshot_key: KeyCode,
    pub toggle_animation_key: KeyCode,
    pub next_frame_key: KeyCode,
    pub prev_frame_key: KeyCode,
}

impl KeyBindings {
    /// Return the [`CameraAction`] mapped to the given key, if any.
    ///
    /// Priority: quit_keys → reset → screenshot → toggle_animation →
    /// next_frame → prev_frame → speed → hardcoded orbit/pan keys.
    pub fn action_for_key(&self, key: KeyCode) -> Option<CameraAction> {
        if self.quit_keys.contains(&key) {
            return Some(CameraAction::Quit);
        }
        if key == self.reset_key {
            return Some(CameraAction::Reset);
        }
        if key == self.screenshot_key {
            return Some(CameraAction::Screenshot);
        }
        if key == self.toggle_animation_key {
            return Some(CameraAction::ToggleAnimation);
        }
        if key == self.next_frame_key {
            return Some(CameraAction::NextFrame);
        }
        if key == self.prev_frame_key {
            return Some(CameraAction::PreviousFrame);
        }

        // Speed control via Plus/Minus
        match key {
            KeyCode::Plus => return Some(CameraAction::SpeedUp),
            KeyCode::Minus => return Some(CameraAction::SlowDown),
            _ => {}
        }

        // WASD / arrow orbit and pan
        const ORBIT_STEP: f32 = 0.05;
        const PAN_STEP: f32 = 0.1;
        match key {
            KeyCode::W => Some(CameraAction::Orbit {
                delta_yaw: 0.0,
                delta_pitch: ORBIT_STEP,
            }),
            KeyCode::X => Some(CameraAction::Orbit {
                delta_yaw: 0.0,
                delta_pitch: -ORBIT_STEP,
            }),
            KeyCode::A => Some(CameraAction::Orbit {
                delta_yaw: -ORBIT_STEP,
                delta_pitch: 0.0,
            }),
            KeyCode::D => Some(CameraAction::Orbit {
                delta_yaw: ORBIT_STEP,
                delta_pitch: 0.0,
            }),
            KeyCode::E => Some(CameraAction::Dolly { delta: -0.5 }),
            KeyCode::Up => Some(CameraAction::Pan {
                delta_x: 0.0,
                delta_y: PAN_STEP,
            }),
            KeyCode::Down => Some(CameraAction::Pan {
                delta_x: 0.0,
                delta_y: -PAN_STEP,
            }),
            _ => None,
        }
    }
}

impl Default for KeyBindings {
    fn default() -> Self {
        Self {
            quit_keys: vec![KeyCode::Q, KeyCode::Escape],
            reset_key: KeyCode::R,
            screenshot_key: KeyCode::S,
            toggle_animation_key: KeyCode::Space,
            next_frame_key: KeyCode::Right,
            prev_frame_key: KeyCode::Left,
        }
    }
}

// ---------------------------------------------------------------------------
// PreviewConfig
// ---------------------------------------------------------------------------

/// Configuration for the preview window and interaction sensitivities.
pub struct PreviewConfig {
    pub width: u32,
    pub height: u32,
    pub title: String,
    pub target_fps: u32,
    /// Mouse sensitivity for orbit dragging (radians per pixel).
    pub mouse_sensitivity: f32,
    /// Scroll sensitivity for dolly zoom.
    pub scroll_sensitivity: f32,
    /// Pan sensitivity for middle-mouse drag.
    pub pan_sensitivity: f32,
    pub show_stats: bool,
    pub show_axes: bool,
}

impl PreviewConfig {
    /// Create a new config with the given dimensions and title.
    #[must_use]
    pub fn new(width: u32, height: u32, title: impl Into<String>) -> Self {
        Self {
            width,
            height,
            title: title.into(),
            target_fps: 60,
            mouse_sensitivity: 0.005,
            scroll_sensitivity: 0.1,
            pan_sensitivity: 0.001,
            show_stats: true,
            show_axes: true,
        }
    }
}

impl Default for PreviewConfig {
    fn default() -> Self {
        Self::new(1280, 720, "OxiGAF Preview")
    }
}

// ---------------------------------------------------------------------------
// PreviewController
// ---------------------------------------------------------------------------

/// Controls camera state and playback from keyboard/mouse events.
pub struct PreviewController {
    pub camera: ArcballCamera,
    pub config: PreviewConfig,
    pub bindings: KeyBindings,
    pub is_animating: bool,
    pub current_frame: usize,
    pub total_frames: usize,
    pub playback_speed: f32,
    is_left_mouse_down: bool,
    last_mouse_x: f32,
    last_mouse_y: f32,
}

impl PreviewController {
    /// Create a new controller with default camera and bindings.
    #[must_use]
    pub fn new(config: PreviewConfig) -> Self {
        Self {
            camera: ArcballCamera::default(),
            config,
            bindings: KeyBindings::default(),
            is_animating: false,
            current_frame: 0,
            total_frames: 0,
            playback_speed: 1.0,
            is_left_mouse_down: false,
            last_mouse_x: 0.0,
            last_mouse_y: 0.0,
        }
    }

    /// Handle a keyboard event.
    ///
    /// Applies the action to internal state and returns it (or `None` if the
    /// key is unbound).
    pub fn handle_key(&mut self, key: KeyCode) -> Option<CameraAction> {
        let action = self.bindings.action_for_key(key)?;
        self.apply_action(action.clone());
        Some(action)
    }

    /// Record the start of a left-mouse drag.
    pub fn handle_mouse_button_down(&mut self, x: f32, y: f32) {
        self.is_left_mouse_down = true;
        self.last_mouse_x = x;
        self.last_mouse_y = y;
    }

    /// Record the end of a left-mouse drag.
    pub fn handle_mouse_button_up(&mut self) {
        self.is_left_mouse_down = false;
    }

    /// Handle mouse movement.
    ///
    /// If a drag is in progress, computes an orbit delta and applies it.
    /// Returns `Some(CameraAction::Orbit{…})` while dragging, otherwise `None`.
    pub fn handle_mouse_move(&mut self, x: f32, y: f32) -> Option<CameraAction> {
        if !self.is_left_mouse_down {
            self.last_mouse_x = x;
            self.last_mouse_y = y;
            return None;
        }

        let dx = x - self.last_mouse_x;
        let dy = y - self.last_mouse_y;
        self.last_mouse_x = x;
        self.last_mouse_y = y;

        let delta_yaw = dx * self.config.mouse_sensitivity;
        let delta_pitch = -dy * self.config.mouse_sensitivity;
        self.camera.orbit(delta_yaw, delta_pitch);
        Some(CameraAction::Orbit {
            delta_yaw,
            delta_pitch,
        })
    }

    /// Handle a scroll event, applying a dolly zoom.
    pub fn handle_scroll(&mut self, delta: f32) -> CameraAction {
        let actual_delta = -delta * self.config.scroll_sensitivity;
        self.camera.dolly(actual_delta);
        CameraAction::Dolly {
            delta: actual_delta,
        }
    }

    /// Advance playback state by `dt_seconds` when animating.
    ///
    /// Frame index wraps modulo `total_frames`. Does nothing if `total_frames`
    /// is zero or animation is paused.
    pub fn tick(&mut self, dt_seconds: f32) {
        if !self.is_animating || self.total_frames == 0 {
            return;
        }
        // Accumulate fractional frames using playback_speed (frames/second relative).
        // We interpret playback_speed * target_fps as frames-per-second rate.
        // Cast target_fps (u32) to f32 directly; values in the 1-1000 range are
        // exactly representable.
        #[allow(clippy::cast_precision_loss)]
        let frames_per_second = self.playback_speed * (self.config.target_fps as f32);
        let frames_advanced = (frames_per_second * dt_seconds) as usize;
        if frames_advanced > 0 {
            self.current_frame = (self.current_frame + frames_advanced) % self.total_frames;
        }
    }

    /// Format a short status string with camera position and frame info.
    ///
    /// Guaranteed to be under 80 characters.
    #[must_use]
    pub fn format_stats(&self) -> String {
        let pos = self.camera.position();
        let frame_info = if self.total_frames > 0 {
            format!("{}/{}", self.current_frame + 1, self.total_frames)
        } else {
            "no frames".to_string()
        };
        // Keep it brief to stay under 80 chars
        format!(
            "pos({:.2},{:.2},{:.2}) yaw={:.2} frm:{}",
            pos[0], pos[1], pos[2], self.camera.yaw, frame_info,
        )
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn apply_action(&mut self, action: CameraAction) {
        match action {
            CameraAction::Orbit {
                delta_yaw,
                delta_pitch,
            } => {
                self.camera.orbit(delta_yaw, delta_pitch);
            }
            CameraAction::Dolly { delta } => {
                self.camera.dolly(delta);
            }
            CameraAction::Pan { delta_x, delta_y } => {
                self.camera.pan(delta_x, delta_y);
            }
            CameraAction::Reset => {
                self.camera.reset();
            }
            CameraAction::ToggleAnimation => {
                self.is_animating = !self.is_animating;
            }
            CameraAction::NextFrame => {
                if self.total_frames > 0 {
                    self.current_frame = (self.current_frame + 1) % self.total_frames;
                }
            }
            CameraAction::PreviousFrame => {
                if self.total_frames > 0 {
                    self.current_frame = self
                        .current_frame
                        .checked_sub(1)
                        .unwrap_or(self.total_frames - 1);
                }
            }
            CameraAction::SpeedUp => {
                self.playback_speed = (self.playback_speed * 2.0).min(8.0);
            }
            CameraAction::SlowDown => {
                self.playback_speed = (self.playback_speed * 0.5).max(0.125);
            }
            // These are returned to the caller to handle externally
            CameraAction::Screenshot | CameraAction::Quit => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_controller() -> PreviewController {
        PreviewController::new(PreviewConfig::default())
    }

    #[test]
    fn test_preview_config_default() {
        let cfg = PreviewConfig::default();
        assert_eq!(cfg.width, 1280);
        assert_eq!(cfg.height, 720);
        assert_eq!(cfg.title, "OxiGAF Preview");
        assert_eq!(cfg.target_fps, 60);
        assert!((cfg.mouse_sensitivity - 0.005).abs() < 1e-6);
        assert!((cfg.scroll_sensitivity - 0.1).abs() < 1e-6);
        assert!((cfg.pan_sensitivity - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_key_bindings_quit_on_q() {
        let bindings = KeyBindings::default();
        assert_eq!(
            bindings.action_for_key(KeyCode::Q),
            Some(CameraAction::Quit)
        );
    }

    #[test]
    fn test_key_bindings_quit_on_escape() {
        let bindings = KeyBindings::default();
        assert_eq!(
            bindings.action_for_key(KeyCode::Escape),
            Some(CameraAction::Quit)
        );
    }

    #[test]
    fn test_key_bindings_unknown_key_returns_none() {
        let bindings = KeyBindings::default();
        // F1 is not bound in default bindings (not in quit_keys, reset, etc.)
        assert_eq!(bindings.action_for_key(KeyCode::F1), None);
    }

    #[test]
    fn test_handle_key_reset() {
        let mut ctrl = make_controller();
        // Orbit first so state is non-default
        ctrl.camera.orbit(1.0, 0.5);
        let action = ctrl.handle_key(KeyCode::R);
        assert_eq!(action, Some(CameraAction::Reset));
        // Camera should be back to default
        let default = ArcballCamera::default();
        assert!((ctrl.camera.yaw - default.yaw).abs() < 1e-4);
        assert!((ctrl.camera.pitch - default.pitch).abs() < 1e-4);
    }

    #[test]
    fn test_handle_key_quit() {
        let mut ctrl = make_controller();
        let action = ctrl.handle_key(KeyCode::Q);
        assert_eq!(action, Some(CameraAction::Quit));
    }

    #[test]
    fn test_handle_mouse_drag_orbits_camera() {
        let mut ctrl = make_controller();
        let initial_yaw = ctrl.camera.yaw;
        ctrl.handle_mouse_button_down(100.0, 100.0);
        let result = ctrl.handle_mouse_move(200.0, 100.0); // 100px right
                                                           // Should return an Orbit action
        assert!(matches!(result, Some(CameraAction::Orbit { .. })));
        // yaw should have changed
        assert!((ctrl.camera.yaw - initial_yaw).abs() > 1e-4);
    }

    #[test]
    fn test_handle_scroll_dollies_camera() {
        let mut ctrl = make_controller();
        let initial_dist = ctrl.camera.distance;
        let action = ctrl.handle_scroll(1.0); // scroll up → dolly in
        assert!(matches!(action, CameraAction::Dolly { .. }));
        // distance should have changed
        assert!((ctrl.camera.distance - initial_dist).abs() > 1e-6);
    }

    #[test]
    fn test_tick_advances_frame_when_animating() {
        let mut ctrl = make_controller();
        ctrl.is_animating = true;
        ctrl.total_frames = 120;
        ctrl.current_frame = 0;
        ctrl.playback_speed = 1.0;
        // Tick for 1 second at 60fps target → should advance 60 frames out of 120
        ctrl.tick(1.0);
        assert!(
            ctrl.current_frame > 0,
            "frame should have advanced (got {})",
            ctrl.current_frame
        );
    }

    #[test]
    fn test_tick_does_not_advance_when_not_animating() {
        let mut ctrl = make_controller();
        ctrl.is_animating = false;
        ctrl.total_frames = 60;
        ctrl.current_frame = 10;
        ctrl.tick(10.0);
        assert_eq!(
            ctrl.current_frame, 10,
            "frame should not advance when paused"
        );
    }

    #[test]
    fn test_format_stats_length() {
        let ctrl = make_controller();
        let stats = ctrl.format_stats();
        assert!(
            stats.len() < 80,
            "stats string too long ({} chars): {stats}",
            stats.len()
        );
    }

    #[test]
    fn test_toggle_animation() {
        let mut ctrl = make_controller();
        assert!(!ctrl.is_animating);
        let action = ctrl.handle_key(KeyCode::Space);
        assert_eq!(action, Some(CameraAction::ToggleAnimation));
        assert!(ctrl.is_animating);

        // Toggle again
        let action2 = ctrl.handle_key(KeyCode::Space);
        assert_eq!(action2, Some(CameraAction::ToggleAnimation));
        assert!(!ctrl.is_animating);
    }
}
