//! Video export for OxiGAF rendered frames.
//!
//! Supports frame sequence output (PNG / JPEG), a single animated GIF
//! (assembled in pure Rust via the `image`/`gif` crates — no external tools
//! required), and JSON manifest generation.  An optional HTML viewer can be
//! generated from a manifest for browser-based playback.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxigaf_cli::video_export::{VideoExportConfig, FrameCollector};
//! use std::path::PathBuf;
//!
//! let config = VideoExportConfig::frame_sequence_png(64, 64, PathBuf::from("/tmp/frames"));
//! let mut collector = FrameCollector::new(config).expect("create collector");
//! let pixels = vec![0u8; 64 * 64 * 4]; // RGBA zeros
//! collector.add_frame(&pixels, 0).expect("add frame");
//! let result = collector.finalize().expect("finalize");
//! println!("{}", result.format_summary());
//! ```

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::error::CliError;

// ---------------------------------------------------------------------------
// VideoFormat
// ---------------------------------------------------------------------------

/// Output format for video export.
#[derive(Debug, Clone)]
pub enum VideoFormat {
    /// Save every frame as an image file.  `extension` must be `"png"` or
    /// `"jpg"`.
    FrameSequence { extension: String },
    /// Assemble every frame into a single animated `.gif` file. Frames are
    /// buffered in memory as they are added and encoded (with the pure-Rust
    /// `image`/`gif` codec stack) when [`FrameCollector::finalize`] is
    /// called. Frame delay derives from [`VideoExportConfig::fps`] and
    /// looping from [`VideoExportConfig::loop_gif`].
    Gif,
    /// Write only the JSON manifest, skip individual frame files.
    Manifest,
}

// ---------------------------------------------------------------------------
// VideoExportConfig
// ---------------------------------------------------------------------------

/// Configuration for a video export run.
#[derive(Debug, Clone)]
pub struct VideoExportConfig {
    /// Output format.
    pub format: VideoFormat,
    /// Frames per second (used for timestamp calculation and manifest).
    pub fps: u32,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Directory to write output files into.
    pub output_dir: PathBuf,
    /// Prefix for per-frame filenames, e.g. `"frame"` → `"frame_000000.png"`.
    pub filename_prefix: String,
    /// Whether to mark the GIF as looping (stored in manifest).
    pub loop_gif: bool,
    /// JPEG quality in `[1, 100]`; ignored for PNG.
    pub quality: u8,
}

impl VideoExportConfig {
    /// Create a new config with explicit format and sensible defaults.
    ///
    /// Defaults: fps=30, prefix=`"frame"`, loop_gif=true, quality=90.
    #[must_use]
    pub fn new(format: VideoFormat, width: u32, height: u32, output_dir: PathBuf) -> Self {
        Self {
            format,
            fps: 30,
            width,
            height,
            output_dir,
            filename_prefix: "frame".to_string(),
            loop_gif: true,
            quality: 90,
        }
    }

    /// Convenience constructor: PNG frame sequence.
    #[must_use]
    pub fn frame_sequence_png(width: u32, height: u32, output_dir: PathBuf) -> Self {
        Self::new(
            VideoFormat::FrameSequence {
                extension: "png".to_string(),
            },
            width,
            height,
            output_dir,
        )
    }

    /// Convenience constructor: single animated GIF.
    #[must_use]
    pub fn gif(width: u32, height: u32, output_dir: PathBuf) -> Self {
        Self::new(VideoFormat::Gif, width, height, output_dir)
    }

    /// Validate the configuration.
    ///
    /// Returns `Err(CliError::VideoExport)` if any value is out of range.
    pub fn validate(&self) -> Result<(), CliError> {
        if self.width == 0 {
            return Err(CliError::VideoExport(
                "VideoExportConfig: width must be > 0".to_string(),
            ));
        }
        if self.height == 0 {
            return Err(CliError::VideoExport(
                "VideoExportConfig: height must be > 0".to_string(),
            ));
        }
        if self.fps == 0 {
            return Err(CliError::VideoExport(
                "VideoExportConfig: fps must be > 0".to_string(),
            ));
        }
        if let VideoFormat::FrameSequence { extension } = &self.format {
            let ext = extension.to_lowercase();
            if ext != "png" && ext != "jpg" && ext != "jpeg" {
                return Err(CliError::VideoExport(format!(
                    "VideoExportConfig: unsupported extension '{extension}'. Use 'png' or 'jpg'."
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FrameMetadata
// ---------------------------------------------------------------------------

/// Metadata about a single exported frame.
#[derive(Debug, Clone)]
pub struct FrameMetadata {
    /// Zero-based frame index.
    pub frame_index: usize,
    /// Timestamp in milliseconds: `frame_index * 1000 / fps`.
    pub timestamp_ms: f64,
    /// Absolute path to the written image file (empty for `Manifest` format).
    pub file_path: PathBuf,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// File size in bytes (0 for `Manifest` format).
    pub size_bytes: u64,
}

// ---------------------------------------------------------------------------
// FrameCollector
// ---------------------------------------------------------------------------

/// Collects rendered frames and writes them to disk according to config.
pub struct FrameCollector {
    /// Export configuration.
    pub config: VideoExportConfig,
    frames: Vec<FrameMetadata>,
    total_frames_added: usize,
    /// Buffered RGBA pixel data for frames destined for a combined
    /// animated GIF. Only populated when `config.format` is
    /// [`VideoFormat::Gif`]; assembled into a single `.gif` file by
    /// [`FrameCollector::finalize`].
    gif_frame_buffers: Vec<Vec<u8>>,
}

impl FrameCollector {
    /// Create a new collector.
    ///
    /// Creates `config.output_dir` (and all parents) if it does not yet exist.
    pub fn new(config: VideoExportConfig) -> Result<Self, CliError> {
        config.validate()?;
        fs::create_dir_all(&config.output_dir).map_err(|e| {
            CliError::VideoExport(format!(
                "Failed to create output directory '{}': {e}",
                config.output_dir.display()
            ))
        })?;
        Ok(Self {
            config,
            frames: Vec::new(),
            total_frames_added: 0,
            gif_frame_buffers: Vec::new(),
        })
    }

    /// Add a rendered frame.
    ///
    /// `pixels` must be RGBA u8 data, length = `width * height * 4`.
    ///
    /// Returns [`FrameMetadata`] describing the saved frame.
    pub fn add_frame(
        &mut self,
        pixels: &[u8],
        frame_index: usize,
    ) -> Result<FrameMetadata, CliError> {
        let expected_len = (self.config.width as usize)
            .checked_mul(self.config.height as usize)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| CliError::VideoExport("Frame dimensions overflow usize".to_string()))?;

        if pixels.len() != expected_len {
            return Err(CliError::VideoExport(format!(
                "add_frame: pixel buffer length {} does not match expected {} ({}x{}x4)",
                pixels.len(),
                expected_len,
                self.config.width,
                self.config.height,
            )));
        }

        let timestamp_ms = frame_index as f64 * 1000.0 / self.config.fps as f64;

        let (file_path, size_bytes) = match &self.config.format {
            VideoFormat::Manifest => {
                // No file written; path is empty.
                (PathBuf::new(), 0u64)
            }
            VideoFormat::FrameSequence { extension } => {
                let ext = extension.clone();
                let filename =
                    format!("{}_{:06}.{}", self.config.filename_prefix, frame_index, ext);
                let path = self.config.output_dir.join(&filename);
                let sz = self.save_frame_image(pixels, &path, &ext)?;
                (path, sz)
            }
            VideoFormat::Gif => {
                // Buffered in memory; assembled into a single animated GIF
                // file at `finalize()` rather than written per frame. The
                // reported path is where that combined file will land; its
                // size is unknown until the GIF is actually encoded.
                self.gif_frame_buffers.push(pixels.to_vec());
                (self.gif_output_path(), 0u64)
            }
        };

        let meta = FrameMetadata {
            frame_index,
            timestamp_ms,
            file_path,
            width: self.config.width,
            height: self.config.height,
            size_bytes,
        };

        self.frames.push(meta.clone());
        self.total_frames_added += 1;
        Ok(meta)
    }

    /// Return the number of frames added so far.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.total_frames_added
    }

    /// Return the cumulative size of all written frame files, in bytes.
    #[must_use]
    pub fn total_size_bytes(&self) -> u64 {
        self.frames.iter().map(|m| m.size_bytes).sum()
    }

    /// Finalize the export: assemble the animated GIF (if applicable),
    /// write the JSON manifest, and return a summary result.
    pub fn finalize(&self) -> Result<VideoExportResult, CliError> {
        // For `VideoFormat::Gif`, the combined animated GIF is only
        // assembled now (rather than per-frame) since every frame's pixels
        // are needed to write a single multi-frame file.
        let gif_size_bytes = if matches!(self.config.format, VideoFormat::Gif)
            && !self.gif_frame_buffers.is_empty()
        {
            let gif_path = self.gif_output_path();
            self.write_gif(&gif_path)?;
            Some(fs::metadata(&gif_path).map(|m| m.len()).unwrap_or(0))
        } else {
            None
        };

        let manifest = VideoManifest::from_collector(self);

        let manifest_filename = format!("{}_manifest.json", self.config.filename_prefix);
        let manifest_path = self.config.output_dir.join(&manifest_filename);
        manifest.save(&manifest_path)?;

        let frame_count = self.frames.len();
        let total_size = gif_size_bytes.unwrap_or_else(|| self.total_size_bytes());
        let duration_ms = if self.config.fps > 0 && frame_count > 0 {
            frame_count as f64 * 1000.0 / self.config.fps as f64
        } else {
            0.0
        };
        let format_str = format_name(&self.config.format);

        Ok(VideoExportResult {
            format: format_str,
            frame_count,
            total_size_bytes: total_size,
            output_dir: self.config.output_dir.clone(),
            manifest_path: Some(manifest_path),
            duration_ms,
            fps: self.config.fps,
            width: self.config.width,
            height: self.config.height,
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Path of the single combined animated GIF file for
    /// [`VideoFormat::Gif`] exports.
    fn gif_output_path(&self) -> PathBuf {
        self.config
            .output_dir
            .join(format!("{}.gif", self.config.filename_prefix))
    }

    /// Assemble all buffered frames into a single animated GIF at `path`,
    /// using the pure-Rust `image`/`gif` codec stack (no external tools or
    /// C dependencies). Frame delay derives from `config.fps` and looping
    /// from `config.loop_gif`.
    fn write_gif(&self, path: &Path) -> Result<(), CliError> {
        use image::codecs::gif::{GifEncoder, Repeat};
        use image::Delay;
        use image::Frame as GifAnimationFrame;

        let file = fs::File::create(path).map_err(|e| {
            CliError::VideoExport(format!(
                "Failed to create GIF file '{}': {e}",
                path.display()
            ))
        })?;
        let mut encoder = GifEncoder::new(std::io::BufWriter::new(file));

        let repeat = if self.config.loop_gif {
            Repeat::Infinite
        } else {
            Repeat::Finite(0)
        };
        encoder
            .set_repeat(repeat)
            .map_err(|e| CliError::VideoExport(format!("Failed to set GIF repeat mode: {e}")))?;

        // fps=0 is rejected by `VideoExportConfig::validate`, but guard
        // against a directly-constructed config bypassing it.
        let delay = Delay::from_numer_denom_ms(1000, self.config.fps.max(1));

        for pixels in &self.gif_frame_buffers {
            let rgba_image =
                image::RgbaImage::from_raw(self.config.width, self.config.height, pixels.clone())
                    .ok_or_else(|| {
                    CliError::VideoExport(format!(
                    "Failed to construct RgbaImage ({}x{}) for GIF frame: buffer may be too small",
                    self.config.width, self.config.height
                ))
                })?;
            let frame = GifAnimationFrame::from_parts(rgba_image, 0, 0, delay);
            encoder.encode_frame(frame).map_err(|e| {
                CliError::VideoExport(format!(
                    "GIF frame encoding failed for '{}': {e}",
                    path.display()
                ))
            })?;
        }

        Ok(())
    }

    /// Save RGBA pixels to `path` as PNG or JPEG.
    ///
    /// Returns the number of bytes written.
    fn save_frame_image(
        &self,
        pixels: &[u8],
        path: &Path,
        extension: &str,
    ) -> Result<u64, CliError> {
        let img =
            image::RgbaImage::from_raw(self.config.width, self.config.height, pixels.to_vec())
                .ok_or_else(|| {
                    CliError::VideoExport(format!(
                        "Failed to construct RgbaImage ({}x{}): buffer may be too small",
                        self.config.width, self.config.height
                    ))
                })?;

        let ext = extension.to_lowercase();
        if ext == "jpg" || ext == "jpeg" {
            // Write JPEG with configured quality.
            let file = fs::File::create(path).map_err(|e| {
                CliError::VideoExport(format!(
                    "Failed to create frame file '{}': {e}",
                    path.display()
                ))
            })?;
            let mut buf_writer = std::io::BufWriter::new(file);
            let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
                &mut buf_writer,
                self.config.quality,
            );
            // JpegEncoder expects RGB; convert from RGBA by dropping alpha.
            let rgb = image::DynamicImage::ImageRgba8(img).to_rgb8();
            encoder.encode_image(&rgb).map_err(|e| {
                CliError::VideoExport(format!(
                    "JPEG encoding failed for '{}': {e}",
                    path.display()
                ))
            })?;
            buf_writer.flush().map_err(|e| {
                CliError::VideoExport(format!(
                    "Failed to flush frame file '{}': {e}",
                    path.display()
                ))
            })?;
        } else {
            // Default to PNG.
            img.save(path).map_err(|e| {
                CliError::VideoExport(format!("PNG save failed for '{}': {e}", path.display()))
            })?;
        }

        let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        Ok(size)
    }
}

// ---------------------------------------------------------------------------
// VideoManifest
// ---------------------------------------------------------------------------

/// JSON-serialisable manifest for all exported frames.
#[derive(Debug, Clone)]
pub struct VideoManifest {
    /// Metadata for every frame, in order.
    pub frames: Vec<FrameMetadata>,
    /// Frames per second.
    pub fps: u32,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Total duration in milliseconds.
    pub duration_ms: f64,
    /// Human-readable format string, e.g. `"png"`, `"gif"`, `"manifest"`.
    pub format: String,
}

impl VideoManifest {
    /// Construct a manifest from a finalised [`FrameCollector`].
    #[must_use]
    pub fn from_collector(collector: &FrameCollector) -> Self {
        let frame_count = collector.frames.len();
        let duration_ms = if collector.config.fps > 0 && frame_count > 0 {
            frame_count as f64 * 1000.0 / collector.config.fps as f64
        } else {
            0.0
        };
        Self {
            frames: collector.frames.clone(),
            fps: collector.config.fps,
            width: collector.config.width,
            height: collector.config.height,
            duration_ms,
            format: format_name(&collector.config.format),
        }
    }

    /// Serialise the manifest to a JSON string (no external serde dependency
    /// needed — built by hand to keep struct derives minimal).
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut s = String::with_capacity(512 + self.frames.len() * 128);
        s.push_str("{\n");
        s.push_str(&format!("  \"fps\": {},\n", self.fps));
        s.push_str(&format!("  \"width\": {},\n", self.width));
        s.push_str(&format!("  \"height\": {},\n", self.height));
        s.push_str(&format!("  \"duration_ms\": {},\n", self.duration_ms));
        s.push_str(&format!("  \"format\": {},\n", json_string(&self.format)));
        s.push_str("  \"frames\": [\n");
        for (i, frame) in self.frames.iter().enumerate() {
            let comma = if i + 1 < self.frames.len() { "," } else { "" };
            let path_str = frame.file_path.to_string_lossy();
            s.push_str("    {\n");
            s.push_str(&format!("      \"frame_index\": {},\n", frame.frame_index));
            s.push_str(&format!(
                "      \"timestamp_ms\": {},\n",
                frame.timestamp_ms
            ));
            s.push_str(&format!(
                "      \"file_path\": {},\n",
                json_string(&path_str)
            ));
            s.push_str(&format!("      \"width\": {},\n", frame.width));
            s.push_str(&format!("      \"height\": {},\n", frame.height));
            s.push_str(&format!("      \"size_bytes\": {}\n", frame.size_bytes));
            s.push_str(&format!("    }}{comma}\n"));
        }
        s.push_str("  ]\n");
        s.push('}');
        s
    }

    /// Write the manifest JSON to `path`.
    pub fn save(&self, path: &Path) -> Result<(), CliError> {
        let json = self.to_json();
        let mut file = fs::File::create(path).map_err(|e| {
            CliError::VideoExport(format!(
                "Failed to create manifest file '{}': {e}",
                path.display()
            ))
        })?;
        file.write_all(json.as_bytes()).map_err(|e| {
            CliError::VideoExport(format!(
                "Failed to write manifest file '{}': {e}",
                path.display()
            ))
        })?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// VideoExportResult
// ---------------------------------------------------------------------------

/// Summary of a completed video export operation.
#[derive(Debug, Clone)]
pub struct VideoExportResult {
    /// Human-readable format name.
    pub format: String,
    /// Number of frames exported.
    pub frame_count: usize,
    /// Total bytes written for all frame files.
    pub total_size_bytes: u64,
    /// Directory where frames and manifest were written.
    pub output_dir: PathBuf,
    /// Path to the written JSON manifest, if any.
    pub manifest_path: Option<PathBuf>,
    /// Total duration in milliseconds.
    pub duration_ms: f64,
    /// Frames per second.
    pub fps: u32,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

impl VideoExportResult {
    /// Return a human-readable single-line summary of the export result.
    #[must_use]
    pub fn format_summary(&self) -> String {
        let size_kb = self.total_size_bytes as f64 / 1024.0;
        format!(
            "Video export: {} frames | format={} | {}x{} | fps={} | \
             duration={:.1}ms | size={:.1}KB | dir={}",
            self.frame_count,
            self.format,
            self.width,
            self.height,
            self.fps,
            self.duration_ms,
            size_kb,
            self.output_dir.display(),
        )
    }
}

// ---------------------------------------------------------------------------
// HtmlViewerConfig
// ---------------------------------------------------------------------------

/// Configuration for the generated HTML viewer page.
#[derive(Debug, Clone)]
pub struct HtmlViewerConfig {
    /// Page `<title>`.
    pub title: String,
    /// Playback FPS for the `setInterval` timer.
    pub fps: u32,
    /// Whether JS should loop back to frame 0 after the last frame.
    pub loop_playback: bool,
    /// Whether to show play/pause/step controls below the image.
    pub show_controls: bool,
}

impl Default for HtmlViewerConfig {
    fn default() -> Self {
        Self {
            title: "OxiGAF Frame Viewer".to_string(),
            fps: 30,
            loop_playback: true,
            show_controls: true,
        }
    }
}

// ---------------------------------------------------------------------------
// generate_html_viewer
// ---------------------------------------------------------------------------

/// Generate a self-contained HTML page that cycles through the exported frames.
///
/// The page uses a JS `setInterval` to swap `<img>` `src` attributes at the
/// configured FPS.  Frame paths are embedded as a JSON array inside the page.
pub fn generate_html_viewer(
    manifest: &VideoManifest,
    output_path: &Path,
    config: &HtmlViewerConfig,
) -> Result<(), CliError> {
    // Build JS array of relative or absolute paths.
    let paths_js: String = {
        let mut arr = String::from("[");
        for (i, frame) in manifest.frames.iter().enumerate() {
            if i > 0 {
                arr.push(',');
            }
            arr.push_str(&json_string(&frame.file_path.to_string_lossy()));
        }
        arr.push(']');
        arr
    };

    let interval_ms = 1000_u32.checked_div(config.fps).unwrap_or(33);

    let loop_js = if config.loop_playback {
        "if (idx >= frames.length) idx = 0;"
    } else {
        "if (idx >= frames.length) { clearInterval(timer); return; }"
    };

    let controls_html = if config.show_controls {
        r#"  <div id="controls" style="margin-top:8px">
    <button onclick="playing=false;clearInterval(timer);">Pause</button>
    <button onclick="if(!playing){playing=true;startTimer();}">Play</button>
    <button onclick="if(idx>0){idx--;document.getElementById('frame').src=frames[idx];}">Prev</button>
    <button onclick="if(idx<frames.length-1){idx++;document.getElementById('frame').src=frames[idx];}">Next</button>
  </div>"#
    } else {
        ""
    };

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    body {{ font-family: sans-serif; background: #111; color: #eee; text-align: center; padding: 20px; }}
    img {{ max-width: 100%; border: 2px solid #444; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>{frame_count} frames &bull; {fps} fps &bull; {width}&times;{height} &bull; {duration:.1}ms</p>
  <img id="frame" src="{first_src}" alt="frame" width="{width}" height="{height}">
{controls_html}
  <script>
    var frames = {paths_js};
    var idx = 0;
    var playing = true;
    var timer;
    function startTimer() {{
      timer = setInterval(function() {{
        idx++;
        {loop_js}
        if (frames.length > 0) document.getElementById('frame').src = frames[idx] || frames[0];
      }}, {interval_ms});
    }}
    if (frames.length > 0) startTimer();
  </script>
</body>
</html>
"#,
        title = html_escape(&config.title),
        frame_count = manifest.frames.len(),
        fps = manifest.fps,
        width = manifest.width,
        height = manifest.height,
        duration = manifest.duration_ms,
        first_src = html_escape(
            &manifest
                .frames
                .first()
                .map(|f| f.file_path.to_string_lossy().into_owned())
                .unwrap_or_default()
        ),
        controls_html = controls_html,
        paths_js = paths_js,
        loop_js = loop_js,
        interval_ms = interval_ms,
    );

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| {
                CliError::VideoExport(format!(
                    "Failed to create HTML viewer directory '{}': {e}",
                    parent.display()
                ))
            })?;
        }
    }

    let mut file = fs::File::create(output_path).map_err(|e| {
        CliError::VideoExport(format!(
            "Failed to create HTML viewer file '{}': {e}",
            output_path.display()
        ))
    })?;
    file.write_all(html.as_bytes()).map_err(|e| {
        CliError::VideoExport(format!(
            "Failed to write HTML viewer file '{}': {e}",
            output_path.display()
        ))
    })?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Return a human-readable format name for a [`VideoFormat`].
fn format_name(format: &VideoFormat) -> String {
    match format {
        VideoFormat::FrameSequence { extension } => extension.clone(),
        VideoFormat::Gif => "gif".to_string(),
        VideoFormat::Manifest => "manifest".to_string(),
    }
}

/// Wrap a string in JSON double-quotes, escaping special characters.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str(r#"\""#),
            '\\' => out.push_str(r"\\"),
            '\n' => out.push_str(r"\n"),
            '\r' => out.push_str(r"\r"),
            '\t' => out.push_str(r"\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Minimal HTML-entity escaping for embedding strings in HTML attributes/text.
fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Unique temp directory for each call (uses pid + nanos to avoid collisions
    /// when tests run in parallel).
    fn unique_temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let pid = std::process::id();
        std::env::temp_dir().join(format!("oxigaf_ve_{label}_{pid}_{nanos}"))
    }

    /// 4-byte RGBA black frame.
    fn rgba_frame(width: u32, height: u32) -> Vec<u8> {
        vec![0u8; width as usize * height as usize * 4]
    }

    // ------------------------------------------------------------------
    // test_config_new
    // ------------------------------------------------------------------
    #[test]
    fn test_config_new() {
        let cfg = VideoExportConfig::new(
            VideoFormat::FrameSequence {
                extension: "png".to_string(),
            },
            320,
            240,
            PathBuf::from("/tmp/out"),
        );
        assert_eq!(cfg.fps, 30);
        assert_eq!(cfg.width, 320);
        assert_eq!(cfg.height, 240);
        assert_eq!(cfg.quality, 90);
        assert!(cfg.loop_gif);
    }

    // ------------------------------------------------------------------
    // test_config_frame_sequence_png
    // ------------------------------------------------------------------
    #[test]
    fn test_config_frame_sequence_png() {
        let cfg = VideoExportConfig::frame_sequence_png(128, 64, PathBuf::from("/tmp/x"));
        assert_eq!(cfg.width, 128);
        assert_eq!(cfg.height, 64);
        assert!(
            matches!(
                &cfg.format,
                VideoFormat::FrameSequence { extension } if extension == "png"
            ),
            "unexpected format: {:?}",
            cfg.format
        );
    }

    // ------------------------------------------------------------------
    // test_config_gif
    // ------------------------------------------------------------------
    #[test]
    fn test_config_gif() {
        let cfg = VideoExportConfig::gif(64, 64, PathBuf::from("/tmp/x"));
        assert!(matches!(cfg.format, VideoFormat::Gif));
    }

    // ------------------------------------------------------------------
    // test_config_validate_zero_width_error
    // ------------------------------------------------------------------
    #[test]
    fn test_config_validate_zero_width_error() {
        let cfg = VideoExportConfig::new(
            VideoFormat::FrameSequence {
                extension: "png".to_string(),
            },
            0,
            100,
            PathBuf::from("/tmp"),
        );
        let err = cfg.validate();
        assert!(err.is_err(), "validate should fail with zero width");
        let msg = err.err().unwrap().to_string();
        assert!(
            msg.contains("width"),
            "error message should mention 'width': {msg}"
        );
    }

    // ------------------------------------------------------------------
    // test_config_validate_zero_fps_error
    // ------------------------------------------------------------------
    #[test]
    fn test_config_validate_zero_fps_error() {
        let mut cfg = VideoExportConfig::frame_sequence_png(64, 64, PathBuf::from("/tmp"));
        cfg.fps = 0;
        let err = cfg.validate();
        assert!(err.is_err(), "validate should fail with fps=0");
        let msg = err.err().unwrap().to_string();
        assert!(
            msg.contains("fps"),
            "error message should mention 'fps': {msg}"
        );
    }

    // ------------------------------------------------------------------
    // test_frame_collector_new_creates_dir
    // ------------------------------------------------------------------
    #[test]
    fn test_frame_collector_new_creates_dir() {
        let dir = unique_temp_dir("new_creates_dir");
        assert!(!dir.exists(), "dir should not exist before test");
        let cfg = VideoExportConfig::frame_sequence_png(4, 4, dir.clone());
        let _collector = FrameCollector::new(cfg).expect("collector creation must succeed");
        assert!(
            dir.exists(),
            "output_dir must be created by FrameCollector::new"
        );
    }

    // ------------------------------------------------------------------
    // test_add_frame_creates_file
    // ------------------------------------------------------------------
    #[test]
    fn test_add_frame_creates_file() {
        let dir = unique_temp_dir("add_frame_creates_file");
        let cfg = VideoExportConfig::frame_sequence_png(4, 4, dir.clone());
        let mut collector = FrameCollector::new(cfg).expect("create collector");
        let pixels = rgba_frame(4, 4);
        let meta = collector.add_frame(&pixels, 0).expect("add frame");
        assert!(
            meta.file_path.exists(),
            "frame file must exist: {}",
            meta.file_path.display()
        );
    }

    // ------------------------------------------------------------------
    // test_add_frame_increments_count
    // ------------------------------------------------------------------
    #[test]
    fn test_add_frame_increments_count() {
        let dir = unique_temp_dir("add_frame_increments");
        let cfg = VideoExportConfig::frame_sequence_png(4, 4, dir);
        let mut collector = FrameCollector::new(cfg).expect("create collector");
        let pixels = rgba_frame(4, 4);
        assert_eq!(collector.frame_count(), 0);
        collector.add_frame(&pixels, 0).expect("add frame 0");
        assert_eq!(collector.frame_count(), 1);
        collector.add_frame(&pixels, 1).expect("add frame 1");
        assert_eq!(collector.frame_count(), 2);
    }

    // ------------------------------------------------------------------
    // test_frame_metadata_timestamp
    // ------------------------------------------------------------------
    #[test]
    fn test_frame_metadata_timestamp() {
        let dir = unique_temp_dir("frame_metadata_ts");
        let mut cfg = VideoExportConfig::frame_sequence_png(4, 4, dir);
        cfg.fps = 25;
        let mut collector = FrameCollector::new(cfg).expect("create collector");
        let pixels = rgba_frame(4, 4);
        // frame 5 at 25fps → 5 * 1000 / 25 = 200ms
        let meta = collector.add_frame(&pixels, 5).expect("add frame");
        let expected = 5.0 * 1000.0 / 25.0;
        assert!(
            (meta.timestamp_ms - expected).abs() < 1e-9,
            "timestamp_ms expected {expected}, got {}",
            meta.timestamp_ms
        );
    }

    // ------------------------------------------------------------------
    // test_frame_collector_total_size
    // ------------------------------------------------------------------
    #[test]
    fn test_frame_collector_total_size() {
        let dir = unique_temp_dir("total_size");
        let cfg = VideoExportConfig::frame_sequence_png(4, 4, dir);
        let mut collector = FrameCollector::new(cfg).expect("create collector");
        let pixels = rgba_frame(4, 4);
        collector.add_frame(&pixels, 0).expect("add frame 0");
        collector.add_frame(&pixels, 1).expect("add frame 1");
        assert!(
            collector.total_size_bytes() > 0,
            "total size must be > 0 after adding frames"
        );
    }

    // ------------------------------------------------------------------
    // test_finalize_creates_manifest
    // ------------------------------------------------------------------
    #[test]
    fn test_finalize_creates_manifest() {
        let dir = unique_temp_dir("finalize_manifest");
        let cfg = VideoExportConfig::frame_sequence_png(4, 4, dir.clone());
        let mut collector = FrameCollector::new(cfg).expect("create collector");
        let pixels = rgba_frame(4, 4);
        collector.add_frame(&pixels, 0).expect("add frame");
        let result = collector.finalize().expect("finalize");
        let manifest_path = result.manifest_path.expect("manifest path must be set");
        assert!(
            manifest_path.exists(),
            "manifest file must be created: {}",
            manifest_path.display()
        );
    }

    // ------------------------------------------------------------------
    // test_manifest_to_json_contains_fps
    // ------------------------------------------------------------------
    #[test]
    fn test_manifest_to_json_contains_fps() {
        let manifest = VideoManifest {
            frames: vec![],
            fps: 24,
            width: 100,
            height: 100,
            duration_ms: 0.0,
            format: "png".to_string(),
        };
        let json = manifest.to_json();
        assert!(
            json.contains("\"fps\": 24"),
            "JSON must contain fps field: {json}"
        );
    }

    // ------------------------------------------------------------------
    // test_manifest_to_json_contains_frames
    // ------------------------------------------------------------------
    #[test]
    fn test_manifest_to_json_contains_frames() {
        let dir = unique_temp_dir("json_frames");
        let cfg = VideoExportConfig::frame_sequence_png(4, 4, dir);
        let mut collector = FrameCollector::new(cfg).expect("create collector");
        let pixels = rgba_frame(4, 4);
        collector.add_frame(&pixels, 0).expect("add frame");
        let manifest = VideoManifest::from_collector(&collector);
        let json = manifest.to_json();
        assert!(
            json.contains("\"frames\""),
            "JSON must contain 'frames' key: {json}"
        );
        assert!(
            json.contains("\"frame_index\": 0"),
            "JSON must contain frame_index 0: {json}"
        );
    }

    // ------------------------------------------------------------------
    // test_manifest_save
    // ------------------------------------------------------------------
    #[test]
    fn test_manifest_save() {
        let dir = unique_temp_dir("manifest_save");
        fs::create_dir_all(&dir).expect("create temp dir");
        let manifest = VideoManifest {
            frames: vec![],
            fps: 30,
            width: 64,
            height: 64,
            duration_ms: 500.0,
            format: "png".to_string(),
        };
        let path = dir.join("test_manifest.json");
        manifest.save(&path).expect("save must succeed");
        assert!(path.exists(), "manifest file must exist after save");
        let contents = fs::read_to_string(&path).expect("read manifest");
        assert!(
            contents.contains("fps"),
            "manifest contents must include fps: {contents}"
        );
    }

    // ------------------------------------------------------------------
    // test_html_viewer_generation
    // ------------------------------------------------------------------
    #[test]
    fn test_html_viewer_generation() {
        let dir = unique_temp_dir("html_viewer");
        fs::create_dir_all(&dir).expect("create temp dir");
        let manifest = VideoManifest {
            frames: vec![FrameMetadata {
                frame_index: 0,
                timestamp_ms: 0.0,
                file_path: PathBuf::from("frame_000000.png"),
                width: 64,
                height: 64,
                size_bytes: 1024,
            }],
            fps: 30,
            width: 64,
            height: 64,
            duration_ms: 33.3,
            format: "png".to_string(),
        };
        let viewer_path = dir.join("viewer.html");
        let html_config = HtmlViewerConfig::default();
        generate_html_viewer(&manifest, &viewer_path, &html_config)
            .expect("HTML viewer generation must succeed");
        assert!(viewer_path.exists(), "HTML viewer file must be created");
        let html = fs::read_to_string(&viewer_path).expect("read HTML file");
        assert!(
            html.contains("setInterval"),
            "HTML must contain setInterval: {html}"
        );
        assert!(
            html.contains("frame_000000.png"),
            "HTML must contain frame path"
        );
    }

    // ------------------------------------------------------------------
    // test_result_format_summary
    // ------------------------------------------------------------------
    #[test]
    fn test_result_format_summary() {
        let result = VideoExportResult {
            format: "png".to_string(),
            frame_count: 10,
            total_size_bytes: 512 * 1024,
            output_dir: PathBuf::from("/tmp/frames"),
            manifest_path: None,
            duration_ms: 333.3,
            fps: 30,
            width: 320,
            height: 240,
        };
        let summary = result.format_summary();
        assert!(
            summary.contains("10 frames"),
            "summary must mention frame count: {summary}"
        );
        assert!(
            summary.contains("png"),
            "summary must mention format: {summary}"
        );
        assert!(
            summary.contains("320"),
            "summary must mention width: {summary}"
        );
    }

    // ------------------------------------------------------------------
    // test_encode_png_rgba_valid — verifies PNG signature through image crate
    // ------------------------------------------------------------------
    #[test]
    fn test_encode_png_rgba_valid() {
        let dir = unique_temp_dir("encode_png_valid");
        let cfg = VideoExportConfig::frame_sequence_png(2, 2, dir);
        let mut collector = FrameCollector::new(cfg).expect("create collector");
        // A tiny 2x2 RGBA image.
        let pixels: Vec<u8> = vec![
            255, 0, 0, 255, // red
            0, 255, 0, 255, // green
            0, 0, 255, 255, // blue
            255, 255, 0, 255, // yellow
        ];
        let meta = collector.add_frame(&pixels, 0).expect("add frame");
        // Read back and verify PNG signature: [137, 80, 78, 71, 13, 10, 26, 10]
        let bytes = fs::read(&meta.file_path).expect("read back PNG file");
        let png_sig = &[137u8, 80, 78, 71, 13, 10, 26, 10];
        assert!(
            bytes.starts_with(png_sig),
            "PNG file must start with PNG signature bytes"
        );
    }

    // ------------------------------------------------------------------
    // GIF assembly — regression tests for a real animated GIF encoder.
    //
    // `VideoFormat::Gif` used to only ever write individual PNG files
    // (identical to `FrameSequence`) and never actually produced a `.gif`
    // container. These tests lock in that a genuine, decodable, multi-frame
    // animated GIF is written by `finalize()`.
    // ------------------------------------------------------------------

    fn solid_rgba_frame(width: u32, height: u32, rgba: [u8; 4]) -> Vec<u8> {
        let mut px = vec![0u8; width as usize * height as usize * 4];
        for chunk in px.chunks_mut(4) {
            chunk.copy_from_slice(&rgba);
        }
        px
    }

    #[test]
    fn test_gif_export_creates_real_animated_gif() {
        let dir = unique_temp_dir("gif_export_real");
        let cfg = VideoExportConfig::gif(2, 2, dir.clone());
        let mut collector = FrameCollector::new(cfg).expect("create collector");

        let red = solid_rgba_frame(2, 2, [255, 0, 0, 255]);
        let blue = solid_rgba_frame(2, 2, [0, 0, 255, 255]);
        collector.add_frame(&red, 0).expect("add frame 0");
        collector.add_frame(&blue, 1).expect("add frame 1");

        let result = collector.finalize().expect("finalize");

        let gif_path = dir.join("frame.gif");
        assert!(
            gif_path.exists(),
            "combined animated GIF file must be created at {}",
            gif_path.display()
        );

        let bytes = fs::read(&gif_path).expect("read GIF file");
        assert!(
            bytes.starts_with(b"GIF89a") || bytes.starts_with(b"GIF87a"),
            "GIF file must start with a GIF header signature"
        );
        assert!(result.total_size_bytes > 0, "reported GIF size must be > 0");

        // Decode it back and verify it really contains 2 animation frames
        // (not just 2 loose still images under a misleading name).
        use image::codecs::gif::GifDecoder;
        use image::AnimationDecoder;
        let decoder = GifDecoder::new(std::io::Cursor::new(bytes)).expect("construct GIF decoder");
        let frames = decoder
            .into_frames()
            .collect_frames()
            .expect("decode GIF frames");
        assert_eq!(frames.len(), 2, "decoded GIF must contain 2 frames");
    }

    #[test]
    fn test_gif_export_no_per_frame_png_files() {
        // For `VideoFormat::Gif`, individual frames must not be written as
        // separate PNG files on disk (that was the pre-fix behavior); only
        // the combined `.gif` file and the manifest should appear.
        let dir = unique_temp_dir("gif_export_no_pngs");
        let cfg = VideoExportConfig::gif(2, 2, dir.clone());
        let mut collector = FrameCollector::new(cfg).expect("create collector");
        let pixels = solid_rgba_frame(2, 2, [10, 20, 30, 255]);
        collector.add_frame(&pixels, 0).expect("add frame 0");
        collector.add_frame(&pixels, 1).expect("add frame 1");
        collector.finalize().expect("finalize");

        assert!(
            !dir.join("frame_000000.png").exists(),
            "GIF export must not leave per-frame PNG files behind"
        );
        assert!(
            !dir.join("frame_000001.png").exists(),
            "GIF export must not leave per-frame PNG files behind"
        );
    }

    #[test]
    fn test_gif_export_empty_writes_no_gif_file() {
        // finalize() with zero frames added should not fabricate an empty
        // .gif file.
        let dir = unique_temp_dir("gif_export_empty");
        let cfg = VideoExportConfig::gif(2, 2, dir.clone());
        let collector = FrameCollector::new(cfg).expect("create collector");
        collector.finalize().expect("finalize");
        assert!(!dir.join("frame.gif").exists());
    }
}
