//! `oxigaf video` — turn a directory of rendered frames into a deliverable.
//!
//! Glue over [`crate::video_export`].
//!
//! | Subcommand | Library entry point |
//! |------------|---------------------|
//! | `build`  | [`crate::video_export::FrameCollector`] |
//! | `viewer` | [`crate::video_export::generate_html_viewer`] |
//!
//! The input is whatever `oxigaf render` (or `oxigaf preview`) wrote: a
//! directory of PNG/JPEG frames, taken in file-name order. `build`
//! re-encodes them as a frame sequence or a single animated GIF and writes
//! the JSON manifest that describes the result; `viewer` writes a
//! self-contained HTML page that plays them back.
//!
//! # No transcoding OxiGAF cannot do
//!
//! There is no MP4/H.264 arm here, and there deliberately is not one: every
//! usable H.264 encoder is a C library, which the Pure-Rust policy rules
//! out. GIF is the animated format this crate can actually write end to end
//! (the pure-Rust `image`/`gif` codec stack), and the manifest gives an
//! external encoder everything it needs to build an MP4 if a caller wants
//! one. Accepting `--format mp4` and quietly emitting a GIF would be worse
//! than not offering it.
//!
//! # Exit codes
//!
//! Unusable inputs are [`crate::error::CliError::InputInvalid`] →
//! [`crate::error::EXIT_IO_ERROR`]; encoding failures come back as
//! [`crate::error::CliError::VideoExport`] → [`crate::error::EXIT_EXPORT_ERROR`].

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{Args, Subcommand, ValueEnum, ValueHint};
use serde_json::json;

use crate::commands::image_io::{image_files, input_invalid, load_rgba};
use crate::commands::{emit, prepare_output, CmdContext};
use crate::progress_types::BatchProgress;
use crate::video_export::{
    generate_html_viewer, FrameCollector, FrameMetadata, HtmlViewerConfig, VideoExportConfig,
    VideoFormat, VideoManifest,
};

/// `oxigaf video <command>`.
#[derive(Debug, Args)]
pub struct VideoArgs {
    #[command(subcommand)]
    pub command: VideoCommand,
}

/// Video-export subcommands.
#[derive(Debug, Subcommand)]
pub enum VideoCommand {
    /// Re-encode a frame directory and write its manifest.
    Build(BuildArgs),

    /// Write a self-contained HTML page that plays a frame directory back.
    Viewer(ViewerArgs),
}

/// Output encoding for `oxigaf video build`.
#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
pub enum VideoOutputFormat {
    /// One PNG per frame (lossless).
    #[default]
    Png,
    /// One JPEG per frame (lossy, honours `--quality`).
    Jpg,
    /// A single animated GIF.
    Gif,
    /// Write only the JSON manifest, no image files.
    Manifest,
}

impl VideoOutputFormat {
    /// Map onto the library's format enum.
    fn to_video_format(self) -> VideoFormat {
        match self {
            Self::Png => VideoFormat::FrameSequence {
                extension: "png".to_string(),
            },
            Self::Jpg => VideoFormat::FrameSequence {
                extension: "jpg".to_string(),
            },
            Self::Gif => VideoFormat::Gif,
            Self::Manifest => VideoFormat::Manifest,
        }
    }
}

/// How frame paths are written into the generated HTML page.
#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
pub enum PathMode {
    /// Relative to the HTML file, so the page and its frames can be moved
    /// or served together. Falls back to absolute for frames outside the
    /// page's directory tree.
    #[default]
    Relative,
    /// Absolute paths — the page only works from this machine.
    Absolute,
}

/// Arguments for `oxigaf video build`.
#[derive(Debug, Args)]
pub struct BuildArgs {
    /// Directory of rendered frames, taken in file-name order.
    #[arg(long, value_hint = ValueHint::DirPath)]
    pub frames: PathBuf,

    /// Directory to write the output and the manifest into.
    #[arg(short, long, value_hint = ValueHint::DirPath)]
    pub output: PathBuf,

    /// Output encoding.
    #[arg(long, value_enum, default_value = "png")]
    pub format: VideoOutputFormat,

    /// Playback frame rate, used for timestamps, duration and GIF delay.
    #[arg(long, default_value = "30")]
    pub fps: u32,

    /// Prefix for the written file names and for the manifest.
    #[arg(long, default_value = "frame")]
    pub prefix: String,

    /// JPEG quality in `1..=100`; ignored for the other formats.
    #[arg(long, default_value = "90")]
    pub quality: u8,

    /// Mark the GIF as playing once instead of looping.
    #[arg(long)]
    pub no_loop: bool,

    /// Also write a self-contained HTML viewer here.
    ///
    /// Only meaningful with `--format png` or `--format jpg`: the viewer
    /// swaps one `<img src>` per frame, and the other two formats leave no
    /// per-frame files to point at. Use `oxigaf video viewer` to build a
    /// page for a frame directory that already exists.
    #[arg(long, value_name = "FILE", value_hint = ValueHint::FilePath)]
    pub html: Option<PathBuf>,

    /// Overwrite `--html` if it already exists.
    #[arg(long)]
    pub force: bool,
}

/// Arguments for `oxigaf video viewer`.
#[derive(Debug, Args)]
pub struct ViewerArgs {
    /// Directory of frames to play back.
    #[arg(long, value_hint = ValueHint::DirPath)]
    pub frames: PathBuf,

    /// HTML file to write.
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    pub output: PathBuf,

    /// Playback frame rate.
    #[arg(long, default_value = "30")]
    pub fps: u32,

    /// Page title.
    #[arg(long, default_value = "OxiGAF Frame Viewer")]
    pub title: String,

    /// Stop at the last frame instead of looping.
    #[arg(long)]
    pub no_loop: bool,

    /// Omit the play/pause/step buttons.
    #[arg(long)]
    pub no_controls: bool,

    /// How frame paths are written into the page.
    #[arg(long, value_enum, default_value = "relative")]
    pub path_mode: PathMode,

    /// Overwrite the output file if it already exists.
    #[arg(long)]
    pub force: bool,
}

/// Run the `video` family.
///
/// # Errors
///
/// Propagates unreadable frame directories, mixed frame resolutions, and
/// encoder failures.
pub fn run(args: VideoArgs, ctx: CmdContext) -> Result<()> {
    match args.command {
        VideoCommand::Build(build_args) => cmd_build(build_args, &ctx),
        VideoCommand::Viewer(viewer_args) => cmd_viewer(viewer_args, &ctx),
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// The frames of `dir`, with the dimensions they must all share.
///
/// Reads the first frame to establish the size; the rest are checked as they
/// are decoded so a mixed-resolution directory fails with the offending file
/// named rather than deep inside the encoder.
fn frame_paths(dir: &Path) -> Result<Vec<PathBuf>> {
    let files = image_files(dir)?;
    if files.is_empty() {
        return Err(input_invalid(dir, "contains no image files"));
    }
    Ok(files)
}

/// Read one frame, requiring it to match `expected` when that is known.
///
/// The size mismatch is reported through [`input_invalid`]'s `reason`:
/// `CliError::InputInvalid`'s `Display` renders it directly (`error.rs`), so
/// "every frame must be one size" — useless without the two sizes — reaches
/// the user without a second `anyhow` context layer repeating it. The typed
/// error underneath still carries the exit code.
fn read_frame(path: &Path, expected: Option<(u32, u32)>) -> Result<(Vec<u8>, u32, u32)> {
    let (pixels, width, height) = load_rgba(path)?;
    if let Some((w, h)) = expected {
        if (w, h) != (width, height) {
            let sentence = format!(
                "{} is {width}×{height} but the first frame is {w}×{h}; \
                 every frame of a sequence must be one size",
                path.display()
            );
            return Err(input_invalid(path, sentence));
        }
    }
    Ok((pixels, width, height))
}

/// Express `target` relative to the directory holding `html`, when possible.
///
/// A path outside that tree has no relative spelling without `..`
/// gymnastics that break as soon as the page is served, so it is left
/// absolute — reported honestly by the caller rather than silently mangled.
fn relative_to_html(target: &Path, html: &Path) -> PathBuf {
    let Some(parent) = html.parent() else {
        return target.to_path_buf();
    };
    let (Ok(target_abs), Ok(parent_abs)) =
        (std::fs::canonicalize(target), std::fs::canonicalize(parent))
    else {
        return target.to_path_buf();
    };
    match target_abs.strip_prefix(&parent_abs) {
        Ok(relative) => relative.to_path_buf(),
        Err(_) => target_abs,
    }
}

/// Build a manifest describing an existing frame directory.
///
/// [`VideoManifest`] has no reader of its own, so `viewer` describes the
/// directory directly instead of round-tripping through a manifest file that
/// a plain `oxigaf render` never wrote.
fn manifest_from_directory(
    files: &[PathBuf],
    fps: u32,
    path_mode: PathMode,
    html: &Path,
) -> Result<(VideoManifest, usize)> {
    let mut frames = Vec::with_capacity(files.len());
    let mut dimensions: Option<(u32, u32)> = None;
    let mut absolute_fallbacks = 0usize;

    for (index, path) in files.iter().enumerate() {
        let (_pixels, width, height) = read_frame(path, dimensions)?;
        dimensions = Some((width, height));
        let size_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let file_path = match path_mode {
            PathMode::Absolute => path.clone(),
            PathMode::Relative => {
                let relative = relative_to_html(path, html);
                if relative.is_absolute() {
                    absolute_fallbacks += 1;
                }
                relative
            }
        };
        frames.push(FrameMetadata {
            frame_index: index,
            timestamp_ms: index as f64 * 1000.0 / f64::from(fps.max(1)),
            file_path,
            width,
            height,
            size_bytes,
        });
    }

    let (width, height) = dimensions.unwrap_or((0, 0));
    let duration_ms = if fps > 0 {
        frames.len() as f64 * 1000.0 / f64::from(fps)
    } else {
        0.0
    };

    Ok((
        VideoManifest {
            frames,
            fps,
            width,
            height,
            duration_ms,
            format: "directory".to_string(),
        },
        absolute_fallbacks,
    ))
}

// ---------------------------------------------------------------------------
// video build
// ---------------------------------------------------------------------------

fn cmd_build(args: BuildArgs, ctx: &CmdContext) -> Result<()> {
    if args.fps == 0 {
        anyhow::bail!("--fps must be at least 1");
    }
    if !(1..=100).contains(&args.quality) {
        anyhow::bail!("--quality must be within 1..=100 (got {})", args.quality);
    }
    // `generate_html_viewer` emits one `<img src>` per manifest frame.
    // `--format manifest` records no file at all (the path is empty) and
    // `--format gif` records the *same* combined file for every frame, so a
    // page built from either shows nothing or one still image. Refusing is
    // the honest answer; writing a broken page and calling it a viewer is
    // not.
    if args.html.is_some()
        && !matches!(args.format, VideoOutputFormat::Png | VideoOutputFormat::Jpg)
    {
        anyhow::bail!(
            "--html needs per-frame image files, so it only works with --format png or \
             --format jpg (got {:?}). Run `oxigaf video build` without --html, then \
             `oxigaf video viewer --frames <dir>`.",
            args.format
        );
    }

    let files = frame_paths(&args.frames)?;

    if ctx.dry_run {
        emit(
            ctx,
            "video build",
            json!({
                "dry_run": true,
                "frames": files.len(),
                "format": format!("{:?}", args.format).to_lowercase(),
                "would_create": [args.output.display().to_string()],
            }),
            &[],
            || {
                println!(
                    "Would encode {} frame(s) into {}",
                    files.len(),
                    args.output.display()
                );
            },
        );
        return Ok(());
    }

    // Establish the frame size before `FrameCollector::new` creates anything:
    // the collector's config carries the dimensions and rejects every frame
    // that disagrees with them.
    let (first_pixels, width, height) = read_frame(&files[0], None)?;

    let mut config = VideoExportConfig::new(
        args.format.to_video_format(),
        width,
        height,
        args.output.clone(),
    );
    config.fps = args.fps;
    config.filename_prefix = args.prefix.clone();
    config.loop_gif = !args.no_loop;
    config.quality = args.quality;

    let mut collector = FrameCollector::new(config)?;

    let progress = if ctx.human() && ctx.verbosity.show_progress() {
        Some(BatchProgress::new(files.len() as u64, "frames encoded"))
    } else {
        None
    };

    collector.add_frame(&first_pixels, 0)?;
    if let Some(ref bar) = progress {
        bar.increment();
    }
    for (index, path) in files.iter().enumerate().skip(1) {
        let (pixels, _, _) = read_frame(path, Some((width, height)))?;
        collector.add_frame(&pixels, index)?;
        if let Some(ref bar) = progress {
            bar.increment();
        }
    }
    if let Some(ref bar) = progress {
        bar.finish();
    }

    let result = collector.finalize()?;

    let mut html_written: Option<PathBuf> = None;
    if let Some(ref html_path) = args.html {
        if prepare_output(ctx, html_path, args.force)? {
            let manifest = VideoManifest::from_collector(&collector);
            generate_html_viewer(
                &manifest,
                html_path,
                &HtmlViewerConfig {
                    title: format!("OxiGAF — {}", args.prefix),
                    fps: args.fps,
                    loop_playback: !args.no_loop,
                    show_controls: true,
                },
            )?;
            html_written = Some(html_path.clone());
        }
    }

    let mut artifacts: Vec<(&str, &Path)> = Vec::new();
    if let Some(ref manifest_path) = result.manifest_path {
        artifacts.push(("manifest", manifest_path.as_path()));
    }
    if let Some(ref html_path) = html_written {
        artifacts.push(("viewer", html_path.as_path()));
    }

    emit(
        ctx,
        "video build",
        json!({
            "source": args.frames.display().to_string(),
            "format": result.format,
            "frame_count": result.frame_count,
            "width": result.width,
            "height": result.height,
            "fps": result.fps,
            "duration_ms": result.duration_ms,
            "total_size_bytes": result.total_size_bytes,
            "output_dir": result.output_dir.display().to_string(),
            "manifest": result
                .manifest_path
                .as_ref()
                .map(|p| p.display().to_string()),
            "viewer": html_written.as_ref().map(|p| p.display().to_string()),
        }),
        &artifacts,
        || {
            println!("{}", result.format_summary());
            if let Some(ref manifest_path) = result.manifest_path {
                println!("Manifest: {}", manifest_path.display());
            }
            if let Some(ref html_path) = html_written {
                println!("Viewer:   {}", html_path.display());
            }
        },
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// video viewer
// ---------------------------------------------------------------------------

fn cmd_viewer(args: ViewerArgs, ctx: &CmdContext) -> Result<()> {
    if args.fps == 0 {
        anyhow::bail!("--fps must be at least 1");
    }

    let files = frame_paths(&args.frames)?;

    if !prepare_output(ctx, &args.output, args.force)? {
        emit(
            ctx,
            "video viewer",
            json!({
                "dry_run": true,
                "frames": files.len(),
                "would_create": [args.output.display().to_string()],
            }),
            &[],
            || println!("Would write viewer: {}", args.output.display()),
        );
        return Ok(());
    }

    let (manifest, absolute_fallbacks) =
        manifest_from_directory(&files, args.fps, args.path_mode, &args.output)?;

    generate_html_viewer(
        &manifest,
        &args.output,
        &HtmlViewerConfig {
            title: args.title.clone(),
            fps: args.fps,
            loop_playback: !args.no_loop,
            show_controls: !args.no_controls,
        },
    )?;

    emit(
        ctx,
        "video viewer",
        json!({
            "source": args.frames.display().to_string(),
            "output": args.output.display().to_string(),
            "frame_count": manifest.frames.len(),
            "width": manifest.width,
            "height": manifest.height,
            "fps": manifest.fps,
            "duration_ms": manifest.duration_ms,
            "absolute_path_fallbacks": absolute_fallbacks,
        }),
        &[("viewer", args.output.as_path())],
        || {
            println!(
                "Wrote a {}-frame viewer to {}",
                manifest.frames.len(),
                args.output.display()
            );
            if absolute_fallbacks > 0 {
                println!(
                    "  NOTE: {absolute_fallbacks} frame(s) are outside the page's directory \
                     and were embedded as absolute paths; the page will not work elsewhere."
                );
            }
        },
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verbosity::Verbosity;

    fn ctx(dry_run: bool) -> CmdContext {
        CmdContext::new(Verbosity::Quiet, true, dry_run)
    }

    /// A directory of `count` solid-colour PNGs, all the same size.
    fn frame_dir(name: &str, count: usize, width: u32, height: u32) -> PathBuf {
        let dir = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp frame dir");
        for index in 0..count {
            let shade = (index * 20).min(255) as u8;
            let buffer: image::RgbaImage =
                image::ImageBuffer::from_pixel(width, height, image::Rgba([shade, 64, 32, 255]));
            buffer
                .save(dir.join(format!("view_{index:03}.png")))
                .expect("temp PNG write");
        }
        dir
    }

    fn build_args(frames: PathBuf, output: PathBuf) -> BuildArgs {
        BuildArgs {
            frames,
            output,
            format: VideoOutputFormat::Png,
            fps: 24,
            prefix: "frame".to_string(),
            quality: 90,
            no_loop: false,
            html: None,
            force: false,
        }
    }

    #[test]
    fn build_writes_frames_and_a_manifest() {
        let frames = frame_dir("oxigaf_video_build_src", 3, 8, 8);
        let output = std::env::temp_dir().join("oxigaf_video_build_out");
        let _ = std::fs::remove_dir_all(&output);

        assert!(cmd_build(build_args(frames.clone(), output.clone()), &ctx(false)).is_ok());
        assert!(output.join("frame_000000.png").is_file());
        assert!(output.join("frame_000002.png").is_file());
        assert!(output.join("frame_manifest.json").is_file());

        let _ = std::fs::remove_dir_all(&frames);
        let _ = std::fs::remove_dir_all(&output);
    }

    /// Regression: the global `--dry-run` must stop before
    /// `FrameCollector::new`, which creates the output directory as a side
    /// effect of construction.
    #[test]
    fn dry_run_build_creates_no_output_directory() {
        let frames = frame_dir("oxigaf_video_dry_src", 2, 8, 8);
        let output = std::env::temp_dir().join("oxigaf_video_dry_out");
        let _ = std::fs::remove_dir_all(&output);

        assert!(cmd_build(build_args(frames.clone(), output.clone()), &ctx(true)).is_ok());
        assert!(
            !output.exists(),
            "dry run must not create {}",
            output.display()
        );

        let _ = std::fs::remove_dir_all(&frames);
    }

    /// A mixed-resolution directory must name the offending frame *and both
    /// sizes*, not fail with an opaque buffer-length message from inside the
    /// encoder — and not with the bare "Invalid input file: <path>" that
    /// `CliError::InputInvalid`'s `Display` used to produce while its
    /// `reason` field went unprinted. `read_frame` used to compensate with a
    /// duplicate `anyhow` context layer carrying the same sentence; now that
    /// `Display` renders `reason` directly (`error.rs`) and that layer is
    /// gone, the resolution must show up exactly once in the full chain.
    #[test]
    fn mixed_resolutions_name_the_offending_frame() {
        let frames = frame_dir("oxigaf_video_mixed", 2, 8, 8);
        let odd = frames.join("view_009.png");
        let buffer: image::RgbaImage =
            image::ImageBuffer::from_pixel(4, 4, image::Rgba([1, 2, 3, 255]));
        buffer.save(&odd).expect("temp PNG write");

        let err = read_frame(&odd, Some((8, 8))).expect_err("a mismatched frame must not load");
        let message = format!("{err}");
        assert!(message.contains("view_009"), "message was: {message}");
        assert!(message.contains("4×4"), "message was: {message}");
        assert!(message.contains("8×8"), "message was: {message}");

        // What the process actually prints, and the status it exits with.
        let (cli_err, detail) = crate::commands::error_report::classify_error(err);
        assert!(detail.contains("4×4"), "rendered chain was: {detail}");
        assert_eq!(
            detail.matches("4×4").count(),
            1,
            "resolution must appear exactly once, not duplicated by a \
             leftover context layer: {detail}"
        );
        assert_eq!(cli_err.exit_code(), crate::error::EXIT_IO_ERROR);

        let _ = std::fs::remove_dir_all(&frames);
    }

    /// The same guard reached through `cmd_build`, so the whole path from a
    /// directory of frames to the error message stays covered.
    #[test]
    fn build_refuses_a_mixed_resolution_directory() {
        let frames = frame_dir("oxigaf_video_mixed_build", 2, 8, 8);
        let buffer: image::RgbaImage =
            image::ImageBuffer::from_pixel(4, 4, image::Rgba([1, 2, 3, 255]));
        buffer
            .save(frames.join("view_009.png"))
            .expect("temp PNG write");
        let output = std::env::temp_dir().join("oxigaf_video_mixed_build_out");
        let _ = std::fs::remove_dir_all(&output);

        let err = cmd_build(build_args(frames.clone(), output.clone()), &ctx(false))
            .expect_err("a mixed-resolution directory must not build");
        let message = format!("{err}");
        assert!(message.contains("view_009"), "message was: {message}");
        assert!(message.contains("4×4"), "message was: {message}");

        let _ = std::fs::remove_dir_all(&frames);
        let _ = std::fs::remove_dir_all(&output);
    }

    #[test]
    fn viewer_writes_relative_paths_next_to_the_page() {
        let frames = frame_dir("oxigaf_video_viewer_src", 2, 8, 8);
        let output = frames.join("viewer.html");

        let args = ViewerArgs {
            frames: frames.clone(),
            output: output.clone(),
            fps: 12,
            title: "Test".to_string(),
            no_loop: false,
            no_controls: false,
            path_mode: PathMode::Relative,
            force: true,
        };
        assert!(cmd_viewer(args, &ctx(false)).is_ok());

        let html = std::fs::read_to_string(&output).expect("viewer read");
        assert!(html.contains("view_000.png"), "viewer HTML: {html}");
        // A relative path must not carry the temp-dir prefix.
        assert!(
            !html.contains(&frames.display().to_string()),
            "frame paths should be relative to the page"
        );

        let _ = std::fs::remove_dir_all(&frames);
    }

    /// `--html` with a format that writes no per-frame files would produce a
    /// page whose every `<img>` points at nothing (manifest) or at the same
    /// combined file (gif). It has to be refused, not silently written.
    #[test]
    fn build_refuses_html_for_formats_without_per_frame_files() {
        let frames = frame_dir("oxigaf_video_html_guard", 2, 8, 8);
        let out = std::env::temp_dir().join("oxigaf_video_html_guard_out");
        let _ = std::fs::remove_dir_all(&out);

        for format in [VideoOutputFormat::Gif, VideoOutputFormat::Manifest] {
            let mut args = build_args(frames.clone(), out.clone());
            args.format = format;
            args.html = Some(out.join("viewer.html"));
            assert!(
                cmd_build(args, &ctx(false)).is_err(),
                "--html must be refused for {format:?}"
            );
        }
        assert!(
            !out.exists(),
            "the guard must fire before anything is written"
        );

        let _ = std::fs::remove_dir_all(&frames);
    }

    #[test]
    fn build_rejects_an_out_of_range_quality() {
        let frames = frame_dir("oxigaf_video_quality", 1, 8, 8);
        let mut args = build_args(
            frames.clone(),
            std::env::temp_dir().join("oxigaf_video_quality_out"),
        );
        args.quality = 0;
        assert!(cmd_build(args, &ctx(false)).is_err());
        let _ = std::fs::remove_dir_all(&frames);
    }
}
