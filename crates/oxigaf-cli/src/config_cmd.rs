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

/// Generate and write a configuration based on hardware detection.
///
/// Instead of interactive stdin prompts, this wizard:
/// 1. Detects available CPU cores via [`std::thread::available_parallelism`].
/// 2. Selects conservative defaults for `batch_size` and `sh_degree`.
/// 3. Prints explanations for each choice.
/// 4. Writes the annotated TOML to `output_path` or stdout.
pub fn run_config_init_wizard(output_path: Option<&Path>) -> Result<()> {
    // Hardware detection
    let cpu_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    // Determine sh_degree: use 3 (quality) when >=4 cores, else 1 (speed)
    let sh_degree: u32 = if cpu_cores >= 4 { 3 } else { 1 };

    // Batch size: conservative estimate based on core count
    let views_per_step: usize = cpu_cores.clamp(1, 8);

    // Inform the user of the decisions being made
    println!("# OxiGAF Configuration Wizard");
    println!("# ============================");
    println!("# Detected {cpu_cores} CPU core(s).");
    println!(
        "# Using sh_degree = {sh_degree} ({quality}).",
        quality = if sh_degree == 3 { "quality" } else { "speed" }
    );
    println!("# Using views_per_step = {views_per_step} (based on core count).");
    println!();

    // Build the base default config
    let mut config = ProjectConfig::default();
    config.training.init.sh_degree = sh_degree;
    config.training.views_per_step = views_per_step;

    // Serialize to TOML
    let toml_body =
        toml::to_string_pretty(&config).context("Failed to serialize wizard config to TOML")?;

    // Build a comment header that explains the choices
    let header = format!(
        "# OxiGAF Configuration — generated by hardware-detection wizard\n\
         # CPU cores detected : {cpu_cores}\n\
         # sh_degree          : {sh_degree} ({quality})\n\
         # views_per_step     : {views_per_step}\n\
         #\n\
         # Tip: increase sh_degree to 3 for best visual quality (needs more VRAM).\n\
         # Tip: lower views_per_step if you run out of GPU memory during training.\n\n",
        quality = if sh_degree == 3 {
            "quality mode"
        } else {
            "speed mode"
        },
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
        run_config_init(Some(&path))?;
        assert!(path.exists());
        let content = fs::read_to_string(&path)?;
        assert!(content.contains("total_iterations"));
        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_config_validate_default_config() -> Result<()> {
        let tmp_dir = env::temp_dir();
        let path = tmp_dir.join("oxigaf_test_validate.toml");
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
    // -----------------------------------------------------------------------

    #[test]
    fn test_wizard_to_stdout_no_error() -> Result<()> {
        // Should complete without error when writing to stdout.
        run_config_init_wizard(None)
    }

    #[test]
    fn test_wizard_creates_file() -> Result<()> {
        let path = env::temp_dir().join("oxigaf_wizard_test_create.toml");
        run_config_init_wizard(Some(&path))?;
        assert!(path.exists(), "Wizard should create the output file");
        fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn test_wizard_output_is_valid_toml() -> Result<()> {
        let path = env::temp_dir().join("oxigaf_wizard_test_valid_toml.toml");
        run_config_init_wizard(Some(&path))?;
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
        run_config_init_wizard(Some(&path))?;
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
}
