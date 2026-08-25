//! `oxigaf sweep` — hyper-parameter search planning and scoring.
//!
//! Wires [`crate::parameter_sweep`]. The family is deliberately split into a
//! *plan* step and a *report* step so the actual training runs happen outside
//! the CLI:
//!
//! 1. `sweep plan` enumerates the trials for a search space and writes them
//!    out. Trial generation is a pure function of the search space, the
//!    strategy and `--seed`.
//! 2. You run each trial and record its score.
//! 3. `sweep report` re-derives the *same* trials from the same flags,
//!    attaches your scores, and reports the best trial together with each
//!    parameter's influence.
//!
//! `sweep hyperband` prints the successive-halving bracket schedule, which is
//! independent of any search space.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use serde_json::json;

use crate::commands::{emit, prepare_output, CmdContext};
use crate::parameter_sweep::{
    format_sweep_summary, format_sweep_trial, hyperband_bracket, ParamSpec, ParamValue,
    ParameterSweep, SweepConfig, SweepError, SweepStrategy, SweepTrial,
};

/// `oxigaf sweep <command>`.
#[derive(Debug, Args)]
pub struct SweepArgs {
    #[command(subcommand)]
    pub command: SweepCommand,
}

/// Sweep subcommands.
#[derive(Debug, Subcommand)]
pub enum SweepCommand {
    /// Enumerate the trials for a search space.
    Plan(PlanArgs),

    /// Score a previously planned sweep and rank the trials.
    Report(ReportArgs),

    /// Print a Hyperband successive-halving bracket schedule.
    Hyperband(HyperbandArgs),
}

/// Run the `sweep` family.
///
/// # Errors
///
/// Propagates malformed search-space definitions, unreadable score files and
/// refused overwrites.
pub fn run(args: SweepArgs, ctx: CmdContext) -> Result<()> {
    match args.command {
        SweepCommand::Plan(plan_args) => cmd_plan(plan_args, &ctx),
        SweepCommand::Report(report_args) => cmd_report(report_args, &ctx),
        SweepCommand::Hyperband(hyperband_args) => cmd_hyperband(hyperband_args, &ctx),
    }
}

// ---------------------------------------------------------------------------
// Search-space definition
// ---------------------------------------------------------------------------

/// How trial parameter combinations are generated.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum Strategy {
    /// Cartesian product over the discrete and categorical dimensions.
    Grid,
    /// Uniform random sampling over the whole space.
    #[default]
    Random,
    /// Nearest-neighbour surrogate guided search.
    Surrogate,
}

impl From<Strategy> for SweepStrategy {
    fn from(value: Strategy) -> Self {
        match value {
            Strategy::Grid => SweepStrategy::Grid,
            Strategy::Random => SweepStrategy::Random,
            Strategy::Surrogate => SweepStrategy::Surrogate,
        }
    }
}

/// The flags that define a search space, shared by `plan` and `report`.
#[derive(Debug, Args)]
pub struct SpaceArgs {
    /// Continuous dimension as `name=low:high` or `name=low:high:log`.
    #[arg(long = "continuous", value_name = "SPEC")]
    pub continuous: Vec<String>,

    /// Discrete numeric dimension as `name=1,2,4`.
    #[arg(long = "discrete", value_name = "SPEC")]
    pub discrete: Vec<String>,

    /// Categorical dimension as `name=adam,sgd`.
    #[arg(long = "categorical", value_name = "SPEC")]
    pub categorical: Vec<String>,

    /// Trial generation strategy.
    #[arg(long, value_enum, default_value = "random")]
    pub strategy: Strategy,

    /// Maximum number of trials to generate.
    #[arg(long, default_value = "20")]
    pub trials: usize,

    /// Seed for the deterministic sampler.
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// Treat a higher score as better (default: lower is better).
    #[arg(long)]
    pub maximize: bool,
}

/// Split `name=rest` into its two halves.
fn split_spec(raw: &str, kind: &str) -> Result<(String, String)> {
    let Some((name, rest)) = raw.split_once('=') else {
        anyhow::bail!("--{kind} {raw:?} must be written as name=…");
    };
    let name = name.trim();
    if name.is_empty() {
        anyhow::bail!("--{kind} {raw:?} has an empty parameter name");
    }
    Ok((name.to_string(), rest.trim().to_string()))
}

fn parse_continuous(raw: &str) -> Result<ParamSpec> {
    let (name, rest) = split_spec(raw, "continuous")?;
    let parts: Vec<&str> = rest.split(':').map(str::trim).collect();
    if parts.len() < 2 || parts.len() > 3 {
        anyhow::bail!("--continuous {raw:?} must be name=low:high or name=low:high:log");
    }
    let low: f64 = parts[0]
        .parse()
        .with_context(|| format!("--continuous {raw:?}: {:?} is not a number", parts[0]))?;
    let high: f64 = parts[1]
        .parse()
        .with_context(|| format!("--continuous {raw:?}: {:?} is not a number", parts[1]))?;
    if !(low.is_finite() && high.is_finite()) || low >= high {
        anyhow::bail!("--continuous {raw:?}: need finite low < high");
    }
    let log_scale = match parts.get(2) {
        None => false,
        Some(&"log") => true,
        Some(&"linear") => false,
        Some(other) => anyhow::bail!("--continuous {raw:?}: {other:?} must be 'log' or 'linear'"),
    };
    if log_scale && low <= 0.0 {
        anyhow::bail!("--continuous {raw:?}: log scaling needs a positive lower bound");
    }
    Ok(ParamSpec::Continuous {
        name,
        low,
        high,
        log_scale,
    })
}

fn parse_discrete(raw: &str) -> Result<ParamSpec> {
    let (name, rest) = split_spec(raw, "discrete")?;
    let mut values = Vec::new();
    for text in rest.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let value: f64 = text
            .parse()
            .with_context(|| format!("--discrete {raw:?}: {text:?} is not a number"))?;
        if !value.is_finite() {
            anyhow::bail!("--discrete {raw:?}: {text:?} is not finite");
        }
        values.push(value);
    }
    if values.is_empty() {
        anyhow::bail!("--discrete {raw:?} lists no values");
    }
    Ok(ParamSpec::Discrete { name, values })
}

fn parse_categorical(raw: &str) -> Result<ParamSpec> {
    let (name, rest) = split_spec(raw, "categorical")?;
    let choices: Vec<String> = rest
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if choices.is_empty() {
        anyhow::bail!("--categorical {raw:?} lists no choices");
    }
    Ok(ParamSpec::Categorical { name, choices })
}

impl SpaceArgs {
    /// Build the library configuration from the flags.
    fn config(&self) -> Result<SweepConfig> {
        if self.trials == 0 {
            anyhow::bail!("--trials must be at least 1");
        }
        let mut specs = Vec::new();
        for raw in &self.continuous {
            specs.push(parse_continuous(raw)?);
        }
        for raw in &self.discrete {
            specs.push(parse_discrete(raw)?);
        }
        for raw in &self.categorical {
            specs.push(parse_categorical(raw)?);
        }
        if specs.is_empty() {
            anyhow::bail!(
                "No search space given. Pass at least one --continuous, --discrete or \
                 --categorical dimension."
            );
        }
        if matches!(self.strategy, Strategy::Grid)
            && specs
                .iter()
                .any(|spec| matches!(spec, ParamSpec::Continuous { .. }))
        {
            anyhow::bail!(
                "--strategy grid cannot enumerate a continuous dimension; use --discrete \
                 values, or --strategy random."
            );
        }
        Ok(SweepConfig {
            specs,
            strategy: self.strategy.into(),
            max_trials: self.trials,
            seed: self.seed,
            minimize: !self.maximize,
        })
    }
}

/// Generate the full trial list a configuration implies.
fn plan_trials(config: SweepConfig) -> Result<(ParameterSweep, Vec<SweepTrial>)> {
    let max_trials = config.max_trials;
    let mut sweep = ParameterSweep::new(config)?;
    let mut trials = Vec::with_capacity(max_trials);
    for _ in 0..max_trials {
        match sweep.suggest() {
            Ok(trial) => trials.push(trial),
            // A finite grid can run out before `max_trials`; that is a
            // complete plan, not a failure.
            Err(SweepError::MaxTrialsReached { .. }) => break,
            Err(other) => return Err(other.into()),
        }
    }
    Ok((sweep, trials))
}

fn param_value_json(value: &ParamValue) -> serde_json::Value {
    match value {
        ParamValue::Float(number) => json!(number),
        ParamValue::Int(number) => json!(number),
        ParamValue::Choice(text) => json!(text),
    }
}

fn trial_json(trial: &SweepTrial) -> serde_json::Value {
    json!({
        "id": trial.id,
        "params": trial
            .params
            .iter()
            .map(|(name, value)| json!({ "name": name, "value": param_value_json(value) }))
            .collect::<Vec<_>>(),
        "score": trial.score,
    })
}

// ---------------------------------------------------------------------------
// sweep plan
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf sweep plan`.
#[derive(Debug, Args)]
pub struct PlanArgs {
    #[command(flatten)]
    pub space: SpaceArgs,

    /// Write the trial list here as JSON.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Overwrite the output file if it exists.
    #[arg(long)]
    pub force: bool,
}

fn cmd_plan(args: PlanArgs, ctx: &CmdContext) -> Result<()> {
    let config = args.space.config()?;
    let strategy = config.strategy;
    let minimize = config.minimize;
    let (_sweep, trials) = plan_trials(config)?;

    let trial_values: Vec<serde_json::Value> = trials.iter().map(trial_json).collect();
    let mut payload = json!({
        "strategy": format!("{strategy:?}"),
        "seed": args.space.seed,
        "objective": if minimize { "minimize" } else { "maximize" },
        "trials": trial_values,
        "count": trials.len(),
    });

    let mut artifacts: Vec<(&str, &Path)> = Vec::new();
    let mut written = false;
    if let Some(ref output) = args.output {
        if prepare_output(ctx, output, args.force)? {
            let document = serde_json::to_string_pretty(&payload)
                .context("Failed to serialise the trial plan")?;
            std::fs::write(output, document)
                .with_context(|| format!("Failed to write {}", output.display()))?;
            artifacts.push(("sweep-plan", output.as_path()));
            written = true;
        }
        if let Some(map) = payload.as_object_mut() {
            map.insert("output".to_string(), json!(output.display().to_string()));
            map.insert("written".to_string(), json!(written));
        }
    }

    emit(ctx, "sweep plan", payload, &artifacts, || {
        println!("Planned {} trial(s) with {strategy:?}:", trials.len());
        for trial in &trials {
            println!("{}", format_sweep_trial(trial));
        }
        if written {
            if let Some(ref output) = args.output {
                println!("Wrote {}", output.display());
            }
        }
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// sweep report
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf sweep report`.
///
/// The search-space flags must match the `sweep plan` invocation that
/// produced the trials, `--seed` included: the trials are re-derived rather
/// than read back, so the ids in the score file line up by construction.
#[derive(Debug, Args)]
pub struct ReportArgs {
    #[command(flatten)]
    pub space: SpaceArgs,

    /// Scores as `{"0": 1.23, …}` or `[{"id": 0, "score": 1.23}, …]`.
    #[arg(long)]
    pub scores: PathBuf,

    /// How many of the best trials to list.
    #[arg(long, default_value = "5")]
    pub top: usize,
}

/// Read a score file in either of the two accepted shapes.
fn read_scores(path: &Path) -> Result<Vec<(usize, f64)>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let document: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("Failed to parse {} as JSON", path.display()))?;

    let mut scores = Vec::new();
    match document {
        serde_json::Value::Object(map) => {
            for (key, value) in map {
                let id: usize = key.parse().with_context(|| {
                    format!("{}: object key {key:?} is not a trial id", path.display())
                })?;
                let score = value.as_f64().ok_or_else(|| {
                    anyhow::anyhow!("{}: score for trial {id} is not a number", path.display())
                })?;
                scores.push((id, score));
            }
        }
        serde_json::Value::Array(entries) => {
            for (position, entry) in entries.iter().enumerate() {
                let id = entry
                    .get("id")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "{}: entry {position} has no numeric \"id\"",
                            path.display()
                        )
                    })? as usize;
                let score = entry
                    .get("score")
                    .and_then(serde_json::Value::as_f64)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "{}: entry {position} has no numeric \"score\"",
                            path.display()
                        )
                    })?;
                scores.push((id, score));
            }
        }
        _ => anyhow::bail!(
            "{}: expected a JSON object or array of scores",
            path.display()
        ),
    }
    scores.sort_by_key(|(id, _)| *id);
    Ok(scores)
}

fn cmd_report(args: ReportArgs, ctx: &CmdContext) -> Result<()> {
    if args.top == 0 {
        anyhow::bail!("--top must be at least 1");
    }
    // `report` re-derives the trials rather than reading them back, which is
    // exact for Grid and Random (both are pure functions of the space and
    // the seed) but *not* for Surrogate: that strategy consults the scores
    // of already-completed trials, and re-deriving with no history would
    // silently regenerate its random fallback instead. Refuse rather than
    // produce a plausible-looking wrong ranking.
    if matches!(args.space.strategy, Strategy::Surrogate) {
        anyhow::bail!(
            "`sweep report` cannot re-derive a --strategy surrogate sweep: surrogate trials \
             depend on the scores of earlier trials, so they cannot be reconstructed from the \
             search space and seed alone. Score a grid or random sweep instead."
        );
    }
    let config = args.space.config()?;
    let (mut sweep, trials) = plan_trials(config)?;
    let scores = read_scores(&args.scores)?;
    if scores.is_empty() {
        anyhow::bail!("{} contains no scores", args.scores.display());
    }

    let mut unknown = Vec::new();
    for (id, score) in &scores {
        if trials.iter().any(|trial| trial.id == *id) {
            sweep.report(*id, *score)?;
        } else {
            unknown.push(*id);
        }
    }
    if !unknown.is_empty() {
        anyhow::bail!(
            "{} scores trial id(s) {unknown:?} that this search space does not generate. \
             The --continuous/--discrete/--categorical/--strategy/--seed/--trials flags must \
             match the `sweep plan` run that produced them.",
            args.scores.display()
        );
    }

    let summary = sweep.summary();
    let best = sweep.best_trial().map(trial_json);
    let top: Vec<serde_json::Value> = sweep
        .top_k_trials(args.top)
        .into_iter()
        .map(trial_json)
        .collect();

    let payload = json!({
        "scores_file": args.scores.display().to_string(),
        "total_trials": summary.total_trials,
        "completed_trials": summary.completed_trials,
        "best_score": summary.best_score,
        "worst_score": summary.worst_score,
        "mean_score": summary.mean_score,
        "std_score": summary.std_score,
        "param_importances": summary
            .param_importances
            .iter()
            .map(|(name, importance)| json!({ "name": name, "importance": importance }))
            .collect::<Vec<_>>(),
        "best_trial": best,
        "top_trials": top,
    });

    let summary_text = format_sweep_summary(&summary);
    emit(ctx, "sweep report", payload, &[], || {
        println!("{summary_text}");
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// sweep hyperband
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf sweep hyperband`.
#[derive(Debug, Args)]
pub struct HyperbandArgs {
    /// Maximum resource (iterations) any single configuration may receive.
    #[arg(long, default_value = "81")]
    pub max_iter: usize,

    /// Downsampling rate between successive-halving rounds.
    #[arg(long, default_value = "3")]
    pub eta: usize,
}

fn cmd_hyperband(args: HyperbandArgs, ctx: &CmdContext) -> Result<()> {
    if args.max_iter == 0 {
        anyhow::bail!("--max-iter must be at least 1");
    }
    if args.eta < 2 {
        anyhow::bail!("--eta must be at least 2 (it is the downsampling rate)");
    }

    let bracket = hyperband_bracket(args.max_iter, args.eta);
    let total_budget: usize = bracket
        .iter()
        .map(|(configs, resource)| configs * resource)
        .sum();

    let payload = json!({
        "max_iter": args.max_iter,
        "eta": args.eta,
        "rounds": bracket
            .iter()
            .map(|(configs, resource)| json!({
                "configurations": configs,
                "resource_per_configuration": resource,
            }))
            .collect::<Vec<_>>(),
        "total_budget": total_budget,
    });

    emit(ctx, "sweep hyperband", payload, &[], || {
        println!(
            "Hyperband bracket (max_iter={}, eta={}):",
            args.max_iter, args.eta
        );
        for (round, (configs, resource)) in bracket.iter().enumerate() {
            println!("  round {round}: {configs} configuration(s) × {resource} iteration(s)");
        }
        println!("  total budget: {total_budget} iteration(s)");
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verbosity::Verbosity;

    fn quiet_ctx() -> CmdContext {
        CmdContext::new(Verbosity::Quiet, true, false)
    }

    fn space(strategy: Strategy) -> SpaceArgs {
        SpaceArgs {
            continuous: Vec::new(),
            discrete: vec!["batch=1,2,4".to_string()],
            categorical: Vec::new(),
            strategy,
            trials: 3,
            seed: 7,
            maximize: false,
        }
    }

    #[test]
    fn continuous_specs_parse_and_validate() {
        let spec = parse_continuous("lr=0.0001:0.01:log").expect("spec");
        assert!(matches!(
            spec,
            ParamSpec::Continuous {
                log_scale: true,
                ..
            }
        ));
        assert!(parse_continuous("lr=0.01:0.0001").is_err(), "low >= high");
        assert!(parse_continuous("lr=0:1:log").is_err(), "log needs low > 0");
        assert!(parse_continuous("lr0.1:1").is_err(), "missing '='");
        assert!(parse_continuous("=0.1:1").is_err(), "empty name");
        assert!(parse_continuous("lr=0.1:1:cubed").is_err());
    }

    #[test]
    fn discrete_and_categorical_specs_parse() {
        assert!(matches!(
            parse_discrete("batch=1,2,4").expect("spec"),
            ParamSpec::Discrete { .. }
        ));
        assert!(parse_discrete("batch=").is_err());
        assert!(matches!(
            parse_categorical("opt=adam,sgd").expect("spec"),
            ParamSpec::Categorical { .. }
        ));
        assert!(parse_categorical("opt=").is_err());
    }

    #[test]
    fn grid_rejects_a_continuous_dimension() {
        let mut args = space(Strategy::Grid);
        args.continuous = vec!["lr=0.001:0.01".to_string()];
        assert!(args.config().is_err());
    }

    #[test]
    fn an_empty_space_is_rejected() {
        let mut args = space(Strategy::Random);
        args.discrete = Vec::new();
        assert!(args.config().is_err());
    }

    #[test]
    fn planning_is_deterministic_for_a_seed() {
        let first = plan_trials(space(Strategy::Random).config().expect("config"))
            .expect("plan")
            .1;
        let second = plan_trials(space(Strategy::Random).config().expect("config"))
            .expect("plan")
            .1;
        assert_eq!(first.len(), second.len());
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(format_sweep_trial(a), format_sweep_trial(b));
        }
    }

    #[test]
    fn scores_parse_from_both_shapes() {
        let dir = std::env::temp_dir();
        let object = dir.join("oxigaf_sweep_scores_object.json");
        std::fs::write(&object, br#"{"1": 2.5, "0": 1.5}"#).expect("write");
        assert_eq!(
            read_scores(&object).expect("object scores"),
            vec![(0usize, 1.5), (1usize, 2.5)]
        );

        let array = dir.join("oxigaf_sweep_scores_array.json");
        std::fs::write(&array, br#"[{"id": 0, "score": 1.5}]"#).expect("write");
        assert_eq!(
            read_scores(&array).expect("array scores"),
            vec![(0usize, 1.5)]
        );

        let bad = dir.join("oxigaf_sweep_scores_bad.json");
        std::fs::write(&bad, b"[{\"id\": 0}]").expect("write");
        assert!(read_scores(&bad).is_err());

        let _ = std::fs::remove_file(&object);
        let _ = std::fs::remove_file(&array);
        let _ = std::fs::remove_file(&bad);
    }

    #[test]
    fn report_rejects_scores_for_unknown_trials() {
        let path = std::env::temp_dir().join("oxigaf_sweep_scores_unknown.json");
        std::fs::write(&path, br#"{"999": 1.0}"#).expect("write");
        let args = ReportArgs {
            space: space(Strategy::Random),
            scores: path.clone(),
            top: 3,
        };
        assert!(cmd_report(args, &quiet_ctx()).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn report_refuses_to_re_derive_a_surrogate_sweep() {
        // Surrogate trials depend on earlier scores, so re-deriving them
        // with no history would silently return the random fallback.
        let path = std::env::temp_dir().join("oxigaf_sweep_scores_surrogate.json");
        std::fs::write(&path, br#"{"0": 1.0}"#).expect("write");
        let args = ReportArgs {
            space: space(Strategy::Surrogate),
            scores: path.clone(),
            top: 3,
        };
        assert!(cmd_report(args, &quiet_ctx()).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn hyperband_rejects_a_degenerate_eta() {
        let args = HyperbandArgs {
            max_iter: 81,
            eta: 1,
        };
        assert!(cmd_hyperband(args, &quiet_ctx()).is_err());
    }

    #[test]
    fn hyperband_brackets_trade_configurations_for_budget() {
        // [`hyperband_bracket`] returns `(n_configs, budget)` ordered
        // *innermost first* — fewest configurations on the largest budget,
        // widening to the most configurations on the smallest budget. That
        // ordering is the function's documented contract and is pinned
        // against Li et al. 2016 by
        // `parameter_sweep::tests::test_hyperband_bracket_matches_published_algorithm_max_iter_81_eta_3`
        // (`5, 8, 15, 34, 81` configurations for `max_iter=81, eta=3`).
        //
        // An earlier version of this test asserted the configuration count
        // must never *grow*, which reads the list in the opposite direction
        // and can therefore never hold; it is the per-configuration budget
        // that shrinks along the list.
        let bracket = hyperband_bracket(81, 3);
        assert_eq!(
            bracket.len(),
            5,
            "max_iter=81, eta=3 gives s_max=4, hence 5 brackets: {bracket:?}"
        );
        for pair in bracket.windows(2) {
            let (configs_inner, budget_inner) = pair[0];
            let (configs_outer, budget_outer) = pair[1];
            assert!(
                configs_outer >= configs_inner,
                "configuration count must not shrink towards the outermost bracket: {bracket:?}"
            );
            assert!(
                budget_outer <= budget_inner,
                "per-configuration budget must not grow towards the outermost bracket: \
                 {bracket:?}"
            );
        }
        for &(configs, budget) in &bracket {
            assert!(
                configs >= 1,
                "every bracket needs a configuration: {bracket:?}"
            );
            assert!(budget >= 1, "every bracket needs a budget: {bracket:?}");
        }
    }

    #[test]
    fn hyperband_reports_the_summed_bracket_budget() {
        // The `total_budget` `cmd_hyperband` reports is the sum over
        // brackets of `configurations × resource_per_configuration`; keep it
        // pinned so a change to the bracket shape cannot silently
        // misreport the work a sweep implies.
        let bracket = hyperband_bracket(81, 3);
        let total: usize = bracket
            .iter()
            .map(|(configs, budget)| configs * budget)
            .sum();
        assert_eq!(total, 5 * 81 + 8 * 27 + 15 * 9 + 34 * 3 + 81, "{bracket:?}");
        assert!(cmd_hyperband(
            HyperbandArgs {
                max_iter: 81,
                eta: 3
            },
            &quiet_ctx()
        )
        .is_ok());
    }
}
