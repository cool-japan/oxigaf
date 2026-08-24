//! `oxigaf config-cmd` subcommand — manage OxiGAF configuration files.
//!
//! Subcommands:
//! - `init [--output <path>]` — write a default config TOML to stdout or to a file.
//! - `validate <path>` — parse a config TOML and report errors or "OK".
//! - `show <path>` — parse and pretty-print all configuration fields.

use std::path::Path;

use anyhow::{Context, Result};

use crate::cli::ConfigCmdSubcommand;
use crate::config::ProjectConfig;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run the `config-cmd` subcommand.
pub fn run_config_cmd(command: ConfigCmdSubcommand) -> Result<()> {
    match command {
        ConfigCmdSubcommand::Init {
            output,
            interactive,
        } => {
            if interactive {
                run_config_init_wizard(output.as_deref())
            } else {
                run_config_init(output.as_deref())
            }
        }
        ConfigCmdSubcommand::Validate { path } => run_config_validate(&path),
        ConfigCmdSubcommand::Show { path } => run_config_show(&path),
    }
}

// ---------------------------------------------------------------------------
// config init
// ---------------------------------------------------------------------------

/// Write the default `ProjectConfig` as TOML to stdout, or to `output` if given.
pub fn run_config_init(output: Option<&Path>) -> Result<()> {
    let default_config = ProjectConfig::default();

    let toml_string = toml::to_string_pretty(&default_config)
        .context("Failed to serialize default config to TOML")?;

    match output {
        None => {
            print!("{}", toml_string);
        }
        Some(path) => {
            if path.exists() {
                anyhow::bail!(
                    "Output file already exists: {}. Remove it first, or re-run with a \
                     different -o/--output path.",
                    path.display()
                );
            }
            std::fs::write(path, &toml_string)
                .with_context(|| format!("Failed to write config to: {}", path.display()))?;
            println!("Default config written to: {}", path.display());
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// config validate
// ---------------------------------------------------------------------------

/// Parse a config TOML file and validate its contents.
///
/// Prints "OK" on success, or a detailed error message on failure.
pub fn run_config_validate(path: &Path) -> Result<()> {
    if !path.exists() {
        anyhow::bail!("Config file not found: {}", path.display());
    }

    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;

    let config: ProjectConfig = toml::from_str(&content)
        .with_context(|| format!("Failed to parse TOML in: {}", path.display()))?;

    config
        .validate()
        .with_context(|| format!("Config validation failed for: {}", path.display()))?;

    println!("OK — {} is valid.", path.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// config show
// ---------------------------------------------------------------------------

/// Parse and pretty-print a config TOML file.
pub fn run_config_show(path: &Path) -> Result<()> {
    if !path.exists() {
        anyhow::bail!("Config file not found: {}", path.display());
    }

    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;

    let config: ProjectConfig = toml::from_str(&content)
        .with_context(|| format!("Failed to parse TOML in: {}", path.display()))?;

    println!("=== OxiGAF Configuration ===");
    println!("File: {}", path.display());
    println!();

    // [model]
    println!("[model]");
    println!(
        "  flame_model_path    = {:?}",
        config.model.flame_model_path
    );
    println!(
        "  diffusion_weights_dir = {:?}",
        config.model.diffusion_weights_dir
    );
    println!();

    // [device]
    println!("[device]");
    println!("  backend   = {:?}", config.device.backend);
    println!("  gpu_index = {}", config.device.gpu_index);
    println!();

    // [training]
    let t = &config.training;
    println!("[training]");
    println!("  total_iterations        = {}", t.total_iterations);
    println!("  views_per_step          = {}", t.views_per_step);
    println!("  image_size              = {}", t.image_size);
    println!("  guidance_scale_start    = {}", t.guidance_scale_start);
    println!("  guidance_scale_end      = {}", t.guidance_scale_end);
    println!("  guidance_anneal_steps   = {}", t.guidance_anneal_steps);
    println!("  num_inference_steps     = {}", t.num_inference_steps);
    println!("  opacity_reset_interval  = {}", t.opacity_reset_interval);
    println!();

    // [training.init]
    let init = &t.init;
    println!("[training.init]");
    println!("  num_rigid_gaussians     = {}", init.num_rigid_gaussians);
    println!(
        "  num_flexible_gaussians  = {}",
        init.num_flexible_gaussians
    );
    println!("  initial_scale           = {}", init.initial_scale);
    println!("  initial_opacity         = {}", init.initial_opacity);
    println!("  sh_degree               = {}", init.sh_degree);
    println!();

    // [training.optimizer]
    let opt = &t.optimizer;
    println!("[training.optimizer]");
    println!("  position_lr             = {:.2e}", opt.position_lr);
    println!("  position_lr_final       = {:.2e}", opt.position_lr_final);
    println!("  rotation_lr             = {:.2e}", opt.rotation_lr);
    println!("  scale_lr                = {:.2e}", opt.scale_lr);
    println!("  opacity_lr              = {:.2e}", opt.opacity_lr);
    println!("  sh_lr                   = {:.2e}", opt.sh_lr);
    println!("  offset_lr               = {:.2e}", opt.offset_lr);
    println!("  beta1                   = {}", opt.beta1);
    println!("  beta2                   = {}", opt.beta2);
    println!("  epsilon                 = {:.2e}", opt.epsilon);
    println!(
        "  position_lr_decay_steps = {}",
        opt.position_lr_decay_steps
    );
    println!();

    // [training.density_control]
    let dc = &t.density_control;
    println!("[training.density_control]");
    println!("  interval                = {}", dc.interval);
    println!("  start_iteration         = {}", dc.start_iteration);
    println!("  end_iteration           = {}", dc.end_iteration);
    println!("  grad_threshold          = {:.2e}", dc.grad_threshold);
    println!("  min_opacity             = {}", dc.min_opacity);
    println!("  max_screen_size         = {}", dc.max_screen_size);
    println!("  split_scale_threshold   = {}", dc.split_scale_threshold);
    println!("  max_gaussians           = {}", dc.max_gaussians);
    println!();

    // [training.loss]
    let loss = &t.loss;
    println!("[training.loss]");
    println!("  lambda_l1               = {}", loss.lambda_l1);
    println!("  lambda_ssim             = {}", loss.lambda_ssim);
    println!("  lambda_ms_ssim          = {}", loss.lambda_ms_ssim);
    println!("  lambda_lpips            = {}", loss.lambda_lpips);
    println!(
        "  lambda_position_reg     = {:.2e}",
        loss.lambda_position_reg
    );
    println!("  lambda_scale_reg        = {:.2e}", loss.lambda_scale_reg);
    println!(
        "  lambda_opacity_reg      = {:.2e}",
        loss.lambda_opacity_reg
    );
    println!("  lambda_normal           = {}", loss.lambda_normal);
    println!(
        "  lambda_gradient_penalty = {}",
        loss.lambda_gradient_penalty
    );
    println!(
        "  gradient_penalty_threshold = {}",
        loss.gradient_penalty_threshold
    );
    println!();

    // [output]
    println!("[output]");
    println!(
        "  checkpoint_interval = {}",
        config.output.checkpoint_interval
    );
    println!("  log_interval        = {}", config.output.log_interval);
    println!("  export_format       = {:?}", config.output.export_format);

    Ok(())
}

// ---------------------------------------------------------------------------
// config init --interactive (hardware-detection wizard)
// ---------------------------------------------------------------------------

/// Real-hardware signal used to pick VRAM-bound wizard defaults.
///
/// Populated from an actual `wgpu` adapter query in [`detect_gpu`]. Kept as
/// a plain struct (rather than querying wgpu inline in the wizard) so the
/// tier-selection logic in [`wizard_hardware_profile`] can be unit-tested
/// without touching real hardware.
#[derive(Debug, Clone, Copy)]
struct DetectedGpu {
    device_type: wgpu::DeviceType,
    max_buffer_size: u64,
    max_texture_dimension_2d: u32,
}

/// Query the first available `wgpu` adapter for a coarse hardware profile.
///
/// Returns `None` if no adapter can be reached at all (headless environment,
/// missing drivers, sandboxed CI, ...); callers must handle that case with a
/// conservative fallback rather than erroring, since `config-cmd init
/// --interactive` should still produce a usable config offline.
fn detect_gpu() -> Option<DetectedGpu> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;

    let info = adapter.get_info();
    let limits = adapter.limits();
    Some(DetectedGpu {
        device_type: info.device_type,
        max_buffer_size: limits.max_buffer_size,
        max_texture_dimension_2d: limits.max_texture_dimension_2d,
    })
}

/// Derive wizard defaults from a (possibly absent) detected GPU profile.
///
/// `sh_degree`, `views_per_step`, and `max_gaussians` are VRAM-bound, not
/// CPU-bound, in a 3D Gaussian Splatting trainer -- a 32-core workstation
/// with a 6 GB GPU cannot run degree-3 SH at 8 views/step, and a 2-core
/// machine with a 24 GB GPU is needlessly throttled by a CPU-only guess.
/// This derives all four from the GPU's reported capability instead:
/// `device_type` (discrete/integrated/software) sets `sh_degree`, and
/// `max_buffer_size` (the largest single allocation the adapter accepts --
/// not literal total VRAM, since wgpu has no portable API for that, but a
/// real signal that scales with GPU class) sets a VRAM budget used to pick
/// `views_per_step`, `image_size`, and `max_gaussians`.
///
/// Pure function: no I/O, so it can be exercised directly in tests.
fn wizard_hardware_profile(gpu: Option<DetectedGpu>) -> (u32, usize, u32, usize, String) {
    // (sh_degree, views_per_step, image_size, max_gaussians, description)
    match gpu {
        Some(g) => {
            let sh_degree = match g.device_type {
                wgpu::DeviceType::DiscreteGpu => 3,
                wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::VirtualGpu => 2,
                wgpu::DeviceType::Cpu | wgpu::DeviceType::Other => 1,
            };
            let image_size = (g.max_texture_dimension_2d.min(1024)).max(512);
            let (views_per_step, max_gaussians) = match g.max_buffer_size {
                b if b >= 2_000_000_000 => (8, 1_000_000),
                b if b >= 1_000_000_000 => (4, 500_000),
                b if b >= 400_000_000 => (2, 200_000),
                _ => (1, 100_000),
            };
            let desc = format!(
                "{:?} adapter, max_buffer_size={} MB",
                g.device_type,
                g.max_buffer_size / (1024 * 1024)
            );
            (sh_degree, views_per_step, image_size, max_gaussians, desc)
        }
        None => (
            1,
            1,
            512,
            100_000,
            "no GPU adapter detected; using conservative offline defaults".to_string(),
        ),
    }
}

/// Generate and write a configuration based on hardware detection.
///
/// Instead of interactive stdin prompts, this wizard:
/// 1. Detects the available GPU via a real `wgpu` adapter query.
/// 2. Selects VRAM-bound defaults for `sh_degree`, `views_per_step`,
///    `image_size`, and `max_gaussians` from that adapter's reported
///    capability (falling back to conservative values if no adapter is
///    reachable).
/// 3. Prints explanations for each choice.
/// 4. Writes the annotated TOML to `output_path` or stdout.
pub fn run_config_init_wizard(output_path: Option<&Path>) -> Result<()> {
    run_config_init_wizard_with(output_path, detect_gpu())
}

/// Implementation of [`run_config_init_wizard`] parameterised over the
/// detected GPU profile, so tests can exercise both the "real GPU found"
/// and "no adapter reachable" branches deterministically.
fn run_config_init_wizard_with(output_path: Option<&Path>, gpu: Option<DetectedGpu>) -> Result<()> {
    if let Some(path) = output_path {
        if path.exists() {
            anyhow::bail!(
                "Output file already exists: {}. Remove it first, or re-run with a \
                 different -o/--output path.",
                path.display()
            );
        }
    }

    let (sh_degree, views_per_step, image_size, max_gaussians, hw_desc) =
        wizard_hardware_profile(gpu);

    // Inform the user of the decisions being made
    println!("# OxiGAF Configuration Wizard");
    println!("# ============================");
    println!("# Hardware: {hw_desc}");
    println!(
        "# Using sh_degree = {sh_degree}, views_per_step = {views_per_step}, \
         image_size = {image_size}, max_gaussians = {max_gaussians}."
    );
    println!();

    // Build the base default config
    let mut config = ProjectConfig::default();
    config.training.init.sh_degree = sh_degree;
    config.training.views_per_step = views_per_step;
    config.training.image_size = image_size;
    config.training.density_control.max_gaussians = max_gaussians;

    // Serialize to TOML
    let toml_body =
        toml::to_string_pretty(&config).context("Failed to serialize wizard config to TOML")?;

    // Build a comment header that explains the choices
    let header = format!(
        "# OxiGAF Configuration — generated by hardware-detection wizard\n\
         # Hardware            : {hw_desc}\n\
         # sh_degree           : {sh_degree}\n\
         # views_per_step      : {views_per_step}\n\
         # image_size          : {image_size}\n\
         # max_gaussians       : {max_gaussians}\n\
         #\n\
         # Tip: increase sh_degree to 3 for best visual quality (needs more VRAM).\n\
         # Tip: lower views_per_step or max_gaussians if you run out of GPU memory.\n\n",
    );

    let full_output = format!("{header}{toml_body}");

    match output_path {
        None => {
            print!("{full_output}");
        }
        Some(path) => {
            std::fs::write(path, &full_output)
                .with_context(|| format!("Failed to write wizard config to: {}", path.display()))?;
            println!("Wizard config written to: {}", path.display());
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;

    #[test]
    fn test_config_init_to_stdout() -> Result<()> {
        // Should not error — writes to stdout, which is fine in tests.
        run_config_init(None)
    }

    #[test]
    fn test_config_init_to_file() -> Result<()> {
        let tmp_dir = env::temp_dir();
        let path = tmp_dir.join("oxigaf_test_config_init.toml");
        fs::remove_file(&path).ok(); // start from a clean slate: init now refuses to overwrite
        run_config_init(Some(&path))?;
        assert!(path.exists());
        let content = fs::read_to_string(&path)?;
        assert!(content.contains("total_iterations"));
        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_config_init_refuses_existing_file() -> Result<()> {
        let tmp_dir = env::temp_dir();
        let path = tmp_dir.join("oxigaf_test_config_init_no_overwrite.toml");
        fs::write(
            &path,
            "# pre-existing tuned config, must not be clobbered\n",
        )?;
        let result = run_config_init(Some(&path));
        assert!(
            result.is_err(),
            "init must refuse to overwrite an existing file"
        );
        let content = fs::read_to_string(&path)?;
        assert!(
            content.contains("must not be clobbered"),
            "existing file must be left untouched"
        );
        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_config_validate_default_config() -> Result<()> {
        let tmp_dir = env::temp_dir();
        let path = tmp_dir.join("oxigaf_test_validate.toml");
        fs::remove_file(&path).ok(); // start from a clean slate: init now refuses to overwrite
                                     // Write default config to file, then validate it.
        run_config_init(Some(&path))?;
        run_config_validate(&path)?;
        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_config_validate_invalid_toml() {
        let tmp_dir = env::temp_dir();
        let path = tmp_dir.join("oxigaf_test_invalid.toml");
        fs::write(&path, "this is !! not valid [[[ toml").ok();
        let result = run_config_validate(&path);
        assert!(result.is_err());
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_config_validate_missing_file() {
        let path = env::temp_dir().join("oxigaf_test_missing_config.toml");
        let result = run_config_validate(&path);
        assert!(result.is_err());
        let err_msg = result.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(err_msg.contains("not found"));
    }

    #[test]
    fn test_config_show_default_config() -> Result<()> {
        let tmp_dir = env::temp_dir();
        let path = tmp_dir.join("oxigaf_test_show.toml");
        fs::remove_file(&path).ok(); // start from a clean slate: init now refuses to overwrite
        run_config_init(Some(&path))?;
        // Should not error.
        run_config_show(&path)?;
        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_config_show_missing_file() {
        let path = env::temp_dir().join("oxigaf_test_show_missing.toml");
        let result = run_config_show(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_roundtrip_serialize_deserialize() -> Result<()> {
        let default_config = ProjectConfig::default();
        let toml_string = toml::to_string_pretty(&default_config).context("serialize error")?;
        let parsed: ProjectConfig = toml::from_str(&toml_string).context("deserialize error")?;
        // Spot-check a few fields.
        assert_eq!(
            parsed.training.total_iterations,
            default_config.training.total_iterations
        );
        assert_eq!(
            parsed.training.init.sh_degree,
            default_config.training.init.sh_degree
        );
        assert_eq!(
            parsed.output.export_format,
            default_config.output.export_format
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Wizard tests
    //
    // These exercise `run_config_init_wizard_with(.., None)` rather than the
    // public `run_config_init_wizard`, so they are deterministic and do not
    // spin up a real `wgpu` adapter (slow, and unpredictable on headless
    // CI). `wizard_hardware_profile`'s branch selection is covered directly
    // below; `detect_gpu`/`run_config_init_wizard` themselves are thin,
    // real-hardware wrappers around already-tested pure logic.
    // -----------------------------------------------------------------------

    #[test]
    fn test_wizard_to_stdout_no_error() -> Result<()> {
        // Should complete without error when writing to stdout.
        run_config_init_wizard_with(None, None)
    }

    #[test]
    fn test_wizard_creates_file() -> Result<()> {
        let path = env::temp_dir().join("oxigaf_wizard_test_create.toml");
        fs::remove_file(&path).ok();
        run_config_init_wizard_with(Some(&path), None)?;
        assert!(path.exists(), "Wizard should create the output file");
        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_wizard_refuses_existing_file() -> Result<()> {
        let path = env::temp_dir().join("oxigaf_wizard_test_no_overwrite.toml");
        fs::write(
            &path,
            "# pre-existing tuned config, must not be clobbered\n",
        )?;
        let result = run_config_init_wizard_with(Some(&path), None);
        assert!(
            result.is_err(),
            "wizard init must refuse to overwrite an existing file"
        );
        let content = fs::read_to_string(&path)?;
        assert!(
            content.contains("must not be clobbered"),
            "existing file must be left untouched"
        );
        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_wizard_output_is_valid_toml() -> Result<()> {
        let path = env::temp_dir().join("oxigaf_wizard_test_valid_toml.toml");
        fs::remove_file(&path).ok();
        run_config_init_wizard_with(Some(&path), None)?;
        let content = fs::read_to_string(&path).context("read wizard output")?;
        // Strip comment lines (TOML parsers do handle # comments but let's verify parse)
        let _parsed: toml::Value =
            toml::from_str(&content).context("wizard output is not valid TOML")?;
        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_wizard_output_contains_expected_keys() -> Result<()> {
        let path = env::temp_dir().join("oxigaf_wizard_test_keys.toml");
        fs::remove_file(&path).ok();
        run_config_init_wizard_with(Some(&path), None)?;
        let content = fs::read_to_string(&path).context("read wizard output")?;
        // Check that the key training fields are present
        assert!(
            content.contains("total_iterations"),
            "Missing total_iterations in wizard output"
        );
        assert!(
            content.contains("sh_degree"),
            "Missing sh_degree in wizard output"
        );
        assert!(
            content.contains("views_per_step"),
            "Missing views_per_step in wizard output"
        );
        fs::remove_file(&path).ok();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // wizard_hardware_profile: VRAM-bound tier selection (pure, hermetic)
    //
    // Regression coverage for: the wizard used to derive `sh_degree` and
    // `views_per_step` from `available_parallelism()` (CPU cores), which
    // are both VRAM-bound properties of a 3DGS trainer, not CPU-bound ones.
    // -----------------------------------------------------------------------

    #[test]
    fn wizard_profile_no_adapter_is_conservative() {
        let (sh_degree, views_per_step, image_size, max_gaussians, desc) =
            wizard_hardware_profile(None);
        assert_eq!(sh_degree, 1);
        assert_eq!(views_per_step, 1);
        assert_eq!(image_size, 512);
        assert_eq!(max_gaussians, 100_000);
        assert!(desc.contains("no GPU adapter"));
    }

    #[test]
    fn wizard_profile_discrete_high_vram_gpu_gets_top_tier() {
        let gpu = DetectedGpu {
            device_type: wgpu::DeviceType::DiscreteGpu,
            max_buffer_size: 4 * 1024 * 1024 * 1024, // 4 GiB
            max_texture_dimension_2d: 16384,
        };
        let (sh_degree, views_per_step, image_size, max_gaussians, _) =
            wizard_hardware_profile(Some(gpu));
        assert_eq!(sh_degree, 3);
        assert_eq!(views_per_step, 8);
        assert_eq!(image_size, 1024);
        assert_eq!(max_gaussians, 1_000_000);
    }

    #[test]
    fn wizard_profile_integrated_low_vram_gpu_gets_low_tier() {
        let gpu = DetectedGpu {
            device_type: wgpu::DeviceType::IntegratedGpu,
            max_buffer_size: 256 * 1024 * 1024, // 256 MiB
            max_texture_dimension_2d: 8192,
        };
        let (sh_degree, views_per_step, _, max_gaussians, _) = wizard_hardware_profile(Some(gpu));
        assert_eq!(sh_degree, 2); // integrated caps at 2, regardless of VRAM tier
        assert_eq!(views_per_step, 1);
        assert_eq!(max_gaussians, 100_000);
    }

    #[test]
    fn wizard_profile_cpu_software_adapter_gets_lowest_sh_degree() {
        let gpu = DetectedGpu {
            device_type: wgpu::DeviceType::Cpu,
            max_buffer_size: 4 * 1024 * 1024 * 1024,
            max_texture_dimension_2d: 8192,
        };
        let (sh_degree, ..) = wizard_hardware_profile(Some(gpu));
        assert_eq!(
            sh_degree, 1,
            "a CPU/software adapter must not select high SH degree"
        );
    }

    #[test]
    fn wizard_profile_image_size_is_clamped_to_512_1024() {
        let tiny_texture_gpu = DetectedGpu {
            device_type: wgpu::DeviceType::DiscreteGpu,
            max_buffer_size: 4 * 1024 * 1024 * 1024,
            max_texture_dimension_2d: 256, // below the 512 floor
        };
        let (_, _, image_size, ..) = wizard_hardware_profile(Some(tiny_texture_gpu));
        assert_eq!(image_size, 512);

        let huge_texture_gpu = DetectedGpu {
            device_type: wgpu::DeviceType::DiscreteGpu,
            max_buffer_size: 4 * 1024 * 1024 * 1024,
            max_texture_dimension_2d: 16384, // above the 1024 ceiling
        };
        let (_, _, image_size, ..) = wizard_hardware_profile(Some(huge_texture_gpu));
        assert_eq!(image_size, 1024);
    }
}
