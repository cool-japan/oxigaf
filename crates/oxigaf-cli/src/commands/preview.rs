//! `oxigaf preview` — drive a model's camera and re-render it live.
//!
//! Wires [`crate::preview`] (and, through it, [`crate::arcball`]).
//!
//! # What this actually does
//!
//! OxiGAF ships no windowing backend, so there is no OS window to draw into.
//! `preview` instead drives the same [`PreviewController`] a windowed viewer
//! would drive, and writes every frame to an image file:
//!
//! * `<output-dir>/preview.png` is rewritten after every camera change, so a
//!   normal image viewer left open on it behaves like a live viewport.
//! * the screenshot key (`s` by default) additionally writes a numbered
//!   `preview_000.png`, `preview_001.png`, … that is never overwritten.
//!
//! Frames come from [`crate::export::render_point_cloud`], the same software
//! rasteriser `oxigaf render` uses.
//!
//! # Modes
//!
//! * Interactive (default) — raw-mode keyboard input via crossterm.
//! * `--script <file>` — replay a command list with no terminal at all. This
//!   is what makes the camera logic testable and scriptable in CI.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Args;
use crossterm::event::{self, Event, KeyCode as TerminalKey, KeyEventKind};
use crossterm::terminal::enable_raw_mode;
use serde_json::json;

use oxigaf::render::gaussian::GaussianModel;

use crate::commands::model_io::load_scene;
use crate::commands::runtime::RawModeGuard;
use crate::commands::{emit, CmdContext};
use crate::preview::{CameraAction, KeyCode, PreviewConfig, PreviewController};

/// Arguments for `oxigaf preview`.
#[derive(Debug, Args)]
pub struct PreviewArgs {
    /// Model to preview (`.ply` or `.json` checkpoint).
    pub model: PathBuf,

    /// Frame width in pixels.
    #[arg(long, default_value = "800")]
    pub width: u32,

    /// Frame height in pixels.
    #[arg(long, default_value = "600")]
    pub height: u32,

    /// Initial camera distance from the target.
    #[arg(long, default_value = "0.6")]
    pub distance: f32,

    /// Initial azimuth in degrees.
    #[arg(long, default_value = "0.0")]
    pub yaw: f32,

    /// Initial elevation in degrees.
    #[arg(long, default_value = "10.0")]
    pub pitch: f32,

    /// Directory the rendered frames are written to.
    ///
    /// `preview.png` inside it is the live viewport and is rewritten on
    /// every camera change — unlike every other writing command in this CLI
    /// it is deliberately not guarded by `--force`, because continuously
    /// overwriting that one file *is* the feature. Numbered screenshots are
    /// never overwritten. Under `--dry-run` nothing is written at all.
    #[arg(long, default_value = ".")]
    pub output_dir: PathBuf,

    /// Replay this command file instead of reading the keyboard.
    #[arg(long)]
    pub script: Option<PathBuf>,

    /// Stop the interactive loop after this many events (0 = unlimited).
    #[arg(long, default_value = "0")]
    pub max_events: usize,

    /// Number of animation frames the playback controls cycle through.
    #[arg(long, default_value = "0")]
    pub frames: usize,
}

/// Map a terminal key event onto the preview module's key abstraction.
///
/// Returns `None` for keys the preview has no binding for, so an unrelated
/// keypress is ignored rather than mis-handled.
fn map_terminal_key(key: TerminalKey) -> Option<KeyCode> {
    match key {
        TerminalKey::Esc => Some(KeyCode::Escape),
        TerminalKey::Left => Some(KeyCode::Left),
        TerminalKey::Right => Some(KeyCode::Right),
        TerminalKey::Up => Some(KeyCode::Up),
        TerminalKey::Down => Some(KeyCode::Down),
        TerminalKey::F(1) => Some(KeyCode::F1),
        TerminalKey::F(2) => Some(KeyCode::F2),
        TerminalKey::F(3) => Some(KeyCode::F3),
        TerminalKey::Char(character) => parse_key_name(&character.to_lowercase().to_string()),
        _ => None,
    }
}

/// The one-line control summary printed before the interactive loop.
///
/// Kept in step with [`crate::preview::KeyBindings::action_for_key`], which
/// is the only place that decides what a key does: the arrow keys step
/// through animation frames (`next_frame_key`/`prev_frame_key` are matched
/// *before* the orbit/pan block) and `+`/`-` change playback speed, so
/// advertising them as "orbit/pan" and "zoom" sent users to keys that do
/// something else entirely. Orbiting is `wasd`; zooming is `e` and `F3`.
fn controls_hint() -> &'static str {
    "[q] quit  [r] reset  [s] screenshot  [wasd] orbit  [e/F3] zoom  \
     [up/down/F1/F2] pan  [space] play/pause  [left/right] frame  \
     [+/-] speed"
}

/// Map a script token onto the preview module's key abstraction.
fn parse_key_name(name: &str) -> Option<KeyCode> {
    match name {
        "q" => Some(KeyCode::Q),
        "esc" | "escape" => Some(KeyCode::Escape),
        "r" => Some(KeyCode::R),
        "s" => Some(KeyCode::S),
        " " | "space" => Some(KeyCode::Space),
        "right" => Some(KeyCode::Right),
        "left" => Some(KeyCode::Left),
        "up" => Some(KeyCode::Up),
        "down" => Some(KeyCode::Down),
        "+" | "=" | "plus" => Some(KeyCode::Plus),
        "-" | "minus" => Some(KeyCode::Minus),
        "w" => Some(KeyCode::W),
        "a" => Some(KeyCode::A),
        "d" => Some(KeyCode::D),
        "e" => Some(KeyCode::E),
        "x" => Some(KeyCode::X),
        "f1" => Some(KeyCode::F1),
        "f2" => Some(KeyCode::F2),
        "f3" => Some(KeyCode::F3),
        _ => None,
    }
}

/// Copy `model` with every Gaussian centre moved by `-target`.
fn shifted_model(model: &GaussianModel, target: [f32; 3]) -> GaussianModel {
    let mut shifted = model.clone();
    for gaussian in &mut shifted.gaussians {
        gaussian.position[0] -= target[0];
        gaussian.position[1] -= target[1];
        gaussian.position[2] -= target[2];
    }
    shifted
}

/// Render the current camera view of `model` to an image.
///
/// [`crate::pipeline::orbit_camera`] always looks at the world origin, so a
/// panned arcball target is honoured by shifting the scene by `-target`
/// instead — the two are equivalent for a point-cloud render, and the shift
/// only costs a copy when the user has actually panned.
fn render_frame(
    model: &GaussianModel,
    controller: &PreviewController,
    width: u32,
    height: u32,
) -> image::RgbImage {
    let camera = crate::pipeline::orbit_camera(
        controller.camera.yaw.to_degrees(),
        controller.camera.pitch.to_degrees(),
        controller.camera.distance,
        width,
        height,
    );
    let target = controller.camera.target;
    if target == [0.0, 0.0, 0.0] {
        return crate::export::render_point_cloud(model, &camera);
    }
    crate::export::render_point_cloud(&shifted_model(model, target), &camera)
}

/// Live-updating viewport frame plus the numbered screenshots.
struct FrameSink {
    live_path: PathBuf,
    output_dir: PathBuf,
    width: u32,
    height: u32,
    screenshots: usize,
    frames_written: usize,
}

impl FrameSink {
    fn new(output_dir: PathBuf, width: u32, height: u32) -> Self {
        Self {
            live_path: output_dir.join("preview.png"),
            output_dir,
            width,
            height,
            screenshots: 0,
            frames_written: 0,
        }
    }

    /// Rewrite the live viewport frame.
    fn refresh(&mut self, model: &GaussianModel, controller: &PreviewController) -> Result<()> {
        let frame = render_frame(model, controller, self.width, self.height);
        frame
            .save(&self.live_path)
            .with_context(|| format!("Failed to write {}", self.live_path.display()))?;
        self.frames_written += 1;
        Ok(())
    }

    /// Write a numbered screenshot that later frames will not overwrite.
    fn screenshot(
        &mut self,
        model: &GaussianModel,
        controller: &PreviewController,
    ) -> Result<PathBuf> {
        let path = self
            .output_dir
            .join(format!("preview_{:03}.png", self.screenshots));
        let frame = render_frame(model, controller, self.width, self.height);
        frame
            .save(&path)
            .with_context(|| format!("Failed to write {}", path.display()))?;
        self.screenshots += 1;
        self.frames_written += 1;
        Ok(path)
    }
}

/// Apply one action's side effects beyond the controller's own state update.
///
/// Returns `true` when the caller should stop.
fn apply_side_effects(
    action: &CameraAction,
    model: &GaussianModel,
    controller: &PreviewController,
    sink: &mut FrameSink,
    ctx: &CmdContext,
) -> Result<bool> {
    match action {
        CameraAction::Quit => return Ok(true),
        CameraAction::Screenshot => {
            let path = sink.screenshot(model, controller)?;
            if ctx.human() {
                println!("Saved {}", path.display());
            }
        }
        _ => sink.refresh(model, controller)?,
    }
    Ok(false)
}

/// Parse one numeric script argument, naming the offending line on failure.
fn parse_float(
    token: Option<&str>,
    what: &str,
    command: &str,
    script: &Path,
    line_number: usize,
) -> Result<f32> {
    let Some(text) = token else {
        anyhow::bail!(
            "{}:{line_number}: {command} needs a {what}",
            script.display()
        );
    };
    let value: f32 = text.parse().with_context(|| {
        format!(
            "{}:{line_number}: {text:?} is not a number",
            script.display()
        )
    })?;
    if !value.is_finite() {
        anyhow::bail!("{}:{line_number}: {text:?} is not finite", script.display());
    }
    Ok(value)
}

/// Replay a script file. Blank lines and `#` comments are ignored.
///
/// | Command | Effect |
/// |---------|--------|
/// | `key <name>` | feed a key to the controller |
/// | `scroll <delta>` | dolly by a scroll delta |
/// | `mouse-down <x> <y>` / `mouse-up` | start/stop an orbit drag |
/// | `pan-down <x> <y>` / `pan-up` | start/stop a pan drag |
/// | `move <x> <y>` | move the mouse (orbits or pans while dragging) |
/// | `tick <seconds>` | advance animation playback |
/// | `shot` | write a numbered screenshot |
fn replay_script(
    script: &Path,
    model: &GaussianModel,
    controller: &mut PreviewController,
    sink: &mut FrameSink,
    ctx: &CmdContext,
) -> Result<usize> {
    let text = std::fs::read_to_string(script)
        .with_context(|| format!("Failed to read {}", script.display()))?;

    let mut applied = 0usize;
    for (number, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut tokens = line.split_whitespace();
        let Some(command) = tokens.next() else {
            continue;
        };
        let line_number = number + 1;

        let action = match command {
            "key" => {
                let Some(name) = tokens.next() else {
                    anyhow::bail!("{}:{line_number}: key needs a name", script.display());
                };
                let Some(code) = parse_key_name(&name.to_lowercase()) else {
                    anyhow::bail!("{}:{line_number}: unknown key {name:?}", script.display());
                };
                controller.handle_key(code)
            }
            "scroll" => {
                let delta = parse_float(tokens.next(), "delta", command, script, line_number)?;
                Some(controller.handle_scroll(delta))
            }
            "mouse-down" => {
                let x = parse_float(tokens.next(), "x", command, script, line_number)?;
                let y = parse_float(tokens.next(), "y", command, script, line_number)?;
                controller.handle_mouse_button_down(x, y);
                None
            }
            "mouse-up" => {
                controller.handle_mouse_button_up();
                None
            }
            "pan-down" => {
                let x = parse_float(tokens.next(), "x", command, script, line_number)?;
                let y = parse_float(tokens.next(), "y", command, script, line_number)?;
                controller.handle_middle_mouse_button_down(x, y);
                None
            }
            "pan-up" => {
                controller.handle_middle_mouse_button_up();
                None
            }
            "move" => {
                let x = parse_float(tokens.next(), "x", command, script, line_number)?;
                let y = parse_float(tokens.next(), "y", command, script, line_number)?;
                controller.handle_mouse_move(x, y)
            }
            "tick" => {
                let seconds = parse_float(tokens.next(), "seconds", command, script, line_number)?;
                controller.tick(seconds);
                None
            }
            "shot" => Some(CameraAction::Screenshot),
            other => anyhow::bail!(
                "{}:{line_number}: unknown command {other:?}",
                script.display()
            ),
        };

        applied += 1;
        if let Some(action) = action {
            if apply_side_effects(&action, model, controller, sink, ctx)? {
                break;
            }
        }
    }
    Ok(applied)
}

/// Read the keyboard until the user quits.
fn interactive_loop(
    model: &GaussianModel,
    controller: &mut PreviewController,
    sink: &mut FrameSink,
    ctx: &CmdContext,
    max_events: usize,
) -> Result<usize> {
    enable_raw_mode().context("Failed to put the terminal into raw mode for preview controls")?;
    // Restores the terminal on every exit path, `?` included.
    let _guard = RawModeGuard;

    let mut handled = 0usize;
    loop {
        if max_events > 0 && handled >= max_events {
            break;
        }
        if !event::poll(Duration::from_millis(100)).context("Failed to poll for key events")? {
            continue;
        }
        let Event::Key(key_event) = event::read().context("Failed to read a key event")? else {
            continue;
        };
        // Only the initial key-down: terminals that report key-release would
        // otherwise run every handler twice.
        if key_event.kind != KeyEventKind::Press {
            continue;
        }
        let Some(code) = map_terminal_key(key_event.code) else {
            continue;
        };
        let Some(action) = controller.handle_key(code) else {
            continue;
        };
        handled += 1;
        if apply_side_effects(&action, model, controller, sink, ctx)? {
            break;
        }
        if ctx.human() {
            // Raw mode disables the implicit carriage return, and a `print!`
            // without a newline stays in the line buffer until flushed.
            print!("\r{}          ", controller.format_stats());
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
    }
    Ok(handled)
}

/// Run `oxigaf preview`.
///
/// # Errors
///
/// Returns an error for an unreadable model, an unwritable output directory,
/// a malformed script, or a terminal that refuses raw mode.
pub fn run(args: PreviewArgs, ctx: CmdContext) -> Result<()> {
    if args.width == 0 || args.height == 0 {
        anyhow::bail!("--width and --height must both be at least 1");
    }
    if !(args.distance.is_finite() && args.distance > 0.0) {
        anyhow::bail!("--distance must be finite and positive");
    }

    let model = load_scene(&args.model)?;
    if model.is_empty() {
        anyhow::bail!("Model contains no Gaussians: {}", args.model.display());
    }

    let mut config = PreviewConfig::new(args.width, args.height, "OxiGAF Preview");
    config.show_stats = ctx.human();
    let mut controller = PreviewController::new(config);
    controller.camera.distance = args.distance;
    controller.camera.yaw = args.yaw.to_radians();
    controller.camera.pitch = args.pitch.to_radians();
    controller.total_frames = args.frames;

    if ctx.dry_run {
        emit(
            &ctx,
            "preview",
            json!({
                "dry_run": true,
                "model": args.model.display().to_string(),
                "n_gaussians": model.len(),
                "resolution": [args.width, args.height],
                "output_dir": args.output_dir.display().to_string(),
                "mode": if args.script.is_some() { "script" } else { "interactive" },
            }),
            &[],
            || {
                println!(
                    "[dry-run] would preview {} ({} Gaussians) into {}",
                    args.model.display(),
                    model.len(),
                    args.output_dir.display()
                );
            },
        );
        return Ok(());
    }

    std::fs::create_dir_all(&args.output_dir).with_context(|| {
        format!(
            "Failed to create output directory: {}",
            args.output_dir.display()
        )
    })?;

    let mut sink = FrameSink::new(args.output_dir.clone(), args.width, args.height);
    sink.refresh(&model, &controller)?;

    let events = match args.script {
        Some(ref script) => replay_script(script, &model, &mut controller, &mut sink, &ctx)?,
        None => {
            if ctx.human() {
                println!(
                    "Live frame: {}  —  {}",
                    sink.live_path.display(),
                    controls_hint()
                );
            }
            interactive_loop(&model, &mut controller, &mut sink, &ctx, args.max_events)?
        }
    };

    let live_path = sink.live_path.clone();
    let artifacts: Vec<(&str, &Path)> = vec![("frame", live_path.as_path())];
    let payload = json!({
        "model": args.model.display().to_string(),
        "n_gaussians": model.len(),
        "resolution": [args.width, args.height],
        "events": events,
        "frames_written": sink.frames_written,
        "screenshots": sink.screenshots,
        "live_frame": live_path.display().to_string(),
        "camera": {
            "yaw_deg": controller.camera.yaw.to_degrees(),
            "pitch_deg": controller.camera.pitch.to_degrees(),
            "distance": controller.camera.distance,
            "target": controller.camera.target,
        },
        "current_frame": controller.current_frame,
        "total_frames": controller.total_frames,
    });

    emit(&ctx, "preview", payload, &artifacts, || {
        println!();
        println!("{}", controller.format_stats());
        println!(
            "Wrote {} frame(s) ({} screenshot(s)) to {}",
            sink.frames_written,
            sink.screenshots,
            args.output_dir.display()
        );
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verbosity::Verbosity;
    use oxigaf::render::gaussian::GaussianAttributes;

    fn quiet_ctx() -> CmdContext {
        CmdContext::new(Verbosity::Quiet, true, false)
    }

    fn tiny_model() -> GaussianModel {
        GaussianModel {
            gaussians: (0..4)
                .map(|i| GaussianAttributes {
                    position: [i as f32 * 0.05, 0.0, 0.0],
                    _pad0: 0.0,
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [-3.0, -3.0, -3.0],
                    opacity: 3.0,
                })
                .collect(),
            sh_coeffs: vec![0.5; 4 * 3],
            sh_degree: 0,
            face_indices: Vec::new(),
            barycentric: Vec::new(),
            local_offsets: Vec::new(),
            is_rigid: Vec::new(),
        }
    }

    fn controller() -> PreviewController {
        PreviewController::new(PreviewConfig::new(32, 24, "test"))
    }

    #[test]
    fn terminal_keys_map_onto_preview_keys() {
        assert_eq!(map_terminal_key(TerminalKey::Esc), Some(KeyCode::Escape));
        assert_eq!(map_terminal_key(TerminalKey::Char('Q')), Some(KeyCode::Q));
        assert_eq!(map_terminal_key(TerminalKey::F(2)), Some(KeyCode::F2));
        assert_eq!(map_terminal_key(TerminalKey::Char('z')), None);
        assert_eq!(map_terminal_key(TerminalKey::Tab), None);
    }

    #[test]
    fn script_key_names_cover_every_binding() {
        for name in [
            "q", "esc", "r", "s", "space", "left", "right", "up", "down", "plus", "minus", "w",
            "a", "d", "e", "x", "f1", "f2", "f3",
        ] {
            assert!(parse_key_name(name).is_some(), "{name} should parse");
        }
        assert!(parse_key_name("nope").is_none());
    }

    #[test]
    fn a_script_drives_the_camera_and_writes_frames() {
        let dir = std::env::temp_dir().join("oxigaf_preview_script");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let script = dir.join("moves.txt");
        // `d`, not `right`: the arrow keys are bound to frame stepping in
        // `KeyBindings::action_for_key`, which matches `next_frame_key`
        // before it reaches the orbit block. `d` is the orbit-right key.
        std::fs::write(
            &script,
            "# orbit right, zoom in, screenshot\nkey d\nscroll 1\nshot\n",
        )
        .expect("write script");

        let model = tiny_model();
        let mut ctrl = controller();
        let before_yaw = ctrl.camera.yaw;
        let before_distance = ctrl.camera.distance;
        let mut sink = FrameSink::new(dir.clone(), 32, 24);

        let applied =
            replay_script(&script, &model, &mut ctrl, &mut sink, &quiet_ctx()).expect("replay");
        assert_eq!(applied, 3);
        assert!(ctrl.camera.yaw != before_yaw, "orbit must move the camera");
        assert!(
            ctrl.camera.distance < before_distance,
            "a positive scroll must zoom in"
        );
        assert_eq!(sink.screenshots, 1);
        assert!(dir.join("preview_000.png").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The arrow keys step through animation frames rather than orbiting —
    /// the counterpart to the test above, so a future rebinding cannot make
    /// both of them silently vacuous.
    #[test]
    fn the_arrow_keys_step_through_frames_without_moving_the_camera() {
        let dir = std::env::temp_dir().join("oxigaf_preview_frames");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let script = dir.join("frames.txt");
        std::fs::write(&script, "key right\nkey right\nkey left\n").expect("write script");

        let model = tiny_model();
        let mut ctrl = controller();
        // Frame stepping is a deliberate no-op until an animation is
        // loaded, so the controller needs a frame count to step through.
        ctrl.total_frames = 4;
        let before_yaw = ctrl.camera.yaw;
        let before_pitch = ctrl.camera.pitch;
        let mut sink = FrameSink::new(dir.clone(), 32, 24);

        let applied =
            replay_script(&script, &model, &mut ctrl, &mut sink, &quiet_ctx()).expect("replay");
        assert_eq!(applied, 3);
        assert_eq!(ctrl.current_frame, 1, "two forward, one back");
        assert_eq!(ctrl.camera.yaw, before_yaw, "frame keys must not orbit");
        assert_eq!(ctrl.camera.pitch, before_pitch);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Regression: the interactive banner advertised "[arrows] orbit/pan"
    /// and "[+/-] zoom", neither of which is what those keys do. A help line
    /// that names the wrong keys is worse than none.
    #[test]
    fn the_controls_hint_names_the_keys_that_are_actually_bound() {
        let hint = controls_hint();
        for expected in ["[wasd] orbit", "[e/F3] zoom", "[left/right] frame"] {
            assert!(
                hint.contains(expected),
                "hint is missing {expected}: {hint}"
            );
        }
        assert!(!hint.contains("orbit/pan"), "the old wrong hint is back");

        // Every key the hint names must resolve to a real binding, and the
        // action must be the one advertised.
        let mut ctrl = controller();
        let orbit = parse_key_name("d").and_then(|k| ctrl.handle_key(k));
        assert!(matches!(orbit, Some(CameraAction::Orbit { .. })));
        let zoom = parse_key_name("e").and_then(|k| ctrl.handle_key(k));
        assert!(matches!(zoom, Some(CameraAction::Dolly { .. })));
        let speed = parse_key_name("+").and_then(|k| ctrl.handle_key(k));
        assert!(matches!(speed, Some(CameraAction::SpeedUp)));
        let frame = parse_key_name("right").and_then(|k| ctrl.handle_key(k));
        assert!(matches!(frame, Some(CameraAction::NextFrame)));
    }

    #[test]
    fn a_script_stops_at_the_quit_key() {
        let dir = std::env::temp_dir().join("oxigaf_preview_quit");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let script = dir.join("quit.txt");
        std::fs::write(&script, "key q\nkey right\n").expect("write script");

        let model = tiny_model();
        let mut ctrl = controller();
        let before_yaw = ctrl.camera.yaw;
        let mut sink = FrameSink::new(dir.clone(), 32, 24);
        let applied =
            replay_script(&script, &model, &mut ctrl, &mut sink, &quiet_ctx()).expect("replay");
        assert_eq!(applied, 1, "the quit key must end the replay");
        assert_eq!(ctrl.camera.yaw, before_yaw);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_script_rejects_unknown_commands() {
        let dir = std::env::temp_dir().join("oxigaf_preview_bad");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let script = dir.join("bad.txt");
        std::fs::write(&script, "teleport 1 2 3\n").expect("write script");

        let model = tiny_model();
        let mut ctrl = controller();
        let mut sink = FrameSink::new(dir.clone(), 32, 24);
        assert!(replay_script(&script, &model, &mut ctrl, &mut sink, &quiet_ctx()).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_script_rejects_a_missing_argument() {
        let dir = std::env::temp_dir().join("oxigaf_preview_missing");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let script = dir.join("missing.txt");
        std::fs::write(&script, "scroll\n").expect("write script");

        let model = tiny_model();
        let mut ctrl = controller();
        let mut sink = FrameSink::new(dir.clone(), 32, 24);
        assert!(replay_script(&script, &model, &mut ctrl, &mut sink, &quiet_ctx()).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn panning_shifts_every_gaussian_by_the_target() {
        let model = tiny_model();
        let shifted = shifted_model(&model, [10.0, -2.0, 0.5]);
        assert_eq!(shifted.len(), model.len());
        for (moved, original) in shifted.gaussians.iter().zip(model.gaussians.iter()) {
            assert!((moved.position[0] - (original.position[0] - 10.0)).abs() < 1e-6);
            assert!((moved.position[1] - (original.position[1] + 2.0)).abs() < 1e-6);
            assert!((moved.position[2] - (original.position[2] - 0.5)).abs() < 1e-6);
        }
    }

    #[test]
    fn a_rendered_frame_has_the_requested_size() {
        let model = tiny_model();
        let mut ctrl = controller();
        assert_eq!(render_frame(&model, &ctrl, 32, 24).dimensions(), (32, 24));
        ctrl.camera.target = [10.0, 0.0, 0.0];
        assert_eq!(render_frame(&model, &ctrl, 32, 24).dimensions(), (32, 24));
    }
}
