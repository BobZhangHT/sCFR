"""
Simulation runner for sCFR study.

Modes
-----
--simulate [--demo|--full] [--reset] [--main-only] [--no-refresh]
    One-stop driver. Runs, in order:
      1. the main Monte Carlo grid over all 12 scenarios,
      2. the four auxiliary experiments (knot Table 3, prior Table 4,
         misspec Table 5, runtime Figure 5),
      3. run_analysis() to build the figures and the beta-MAE table, and
      4. refresh_docs.py to sync docs/figs_tables.
    demo : 5 reps/scenario for the main grid, 10 for each auxiliary.
    full : config.NUM_MONTE_CARLO_RUNS (500) reps/scenario for the main grid;
           config.AUX_NUM_REPLICATIONS (500) reps for knot/prior/misspec and 100
           reps for runtime; all cores. All experiments run in parallel (joblib);
           the runtime timing pins each fit to a dedicated core so parallelism does
           not corrupt the wall-clock measurement.
    Break-point support: completed main runs and cached auxiliary replicates are
    skipped, so re-invoking `--simulate --full` resumes and runs only what is
    missing (e.g. promoting 10-rep demo auxiliaries to the full count).
    --main-only  skip the auxiliary experiments (run the main grid only).
    --no-refresh skip the automatic refresh_docs.py sync at the end.

    Typical full-scale server command (single line):
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
            python simulation.py --simulate --full

--analyze
    Re-run analysis/plotting on existing simulation_outputs/.

The individual modes below (--runtime/--knot/--prior/--misspec) remain available
for running one auxiliary experiment in isolation.

--runtime [--demo]
    Section 3.5 timing/scalability experiment. Parallel, but each timed fit is
    pinned to a dedicated core (single-threaded) so the wall-clock stays valid.
    Grid: T in {100,200,400,800,1200}, K in {1,2,4,8}.
    demo : 10 reps/setting;  full : 100 reps/setting.

--knot [--demo]
    R2-2 knot-sensitivity experiment on scenario S09 (sinusoidal, K=2).
    J grid: {5,10,15,20,30}. Fits run in parallel (joblib).
    demo : 10 reps;  full : config.AUX_NUM_REPLICATIONS (500) reps.

--prior [--demo]
    R2-3 prior-sensitivity experiment on scenario S09.
    13 prior configurations (scale + family, one-at-a-time). Parallel (joblib).
    demo : 10 reps;  full : 500 reps.

--misspec [--demo]
    R2-6 misspecified-DGP experiment (5 data-generating processes). Parallel (joblib).
    demo : 10 reps;  full : 500 reps.

Server full-scale commands (run from repo root, set BLAS to single-thread):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python simulation.py --simulate --full
    python simulation.py --runtime
    python simulation.py --knot
    python simulation.py --prior
    python simulation.py --misspec
"""

import argparse
import os
import shutil
import json
import time
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# Force single-threaded numerical libraries BEFORE importing JAX, so each fit uses
# exactly one core. This lets the parallel runs (main grid, knot/prior/misspec, and
# the core-pinned runtime timing) scale cleanly across cores, and keeps per-fit
# wall-clock measurements valid. setdefault respects any value the user exports.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS",
                      "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import multiprocessing

import config
import data_generation
import evaluation as ev
from fsCFR_python import fsCFR_model_wrapper

# JAX/NumPyro (and methods, which imports them) power the model fitting. Import them
# optionally so that analysis, plotting, and cached-result regeneration still work
# when JAX is unavailable, e.g. blocked by a Windows application-control policy.
# Fitting requires JAX; those code paths raise a clear error if called without it.
try:
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    import methods
    _JAX_AVAILABLE = True
except Exception as _jax_import_error:  # noqa: BLE001
    jax = jnp = numpyro = dist = MCMC = NUTS = methods = None
    _JAX_AVAILABLE = False
    print(f"[simulation] WARNING: JAX/NumPyro unavailable ({_jax_import_error}). "
          f"Model fitting is disabled; analysis, plotting, and cached-result "
          f"regeneration still work.")


# =============================================================================
# Helper Functions
# =============================================================================

def log_error_to_file(scenario_id, run_idx, error_type, error_message, error_traceback, context=None):
    """Log detailed error information to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{error_type}_error_{scenario_id}_run_{run_idx}_{timestamp}.log"
    log_filepath = os.path.join(config.OUTPUT_DIR_LOGS, log_filename)
    
    log_content = {
        "timestamp": datetime.now().isoformat(),
        "error_type": error_type,
        "scenario_id": scenario_id,
        "run_idx": run_idx,
        "error_message": str(error_message),
        "error_traceback": error_traceback,
        "context": context or {}
    }

    try:
        with open(log_filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"ERROR LOG: {error_type.upper()}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {log_content['timestamp']}\n")
            f.write(f"Scenario ID: {scenario_id}\n")
            f.write(f"Run Index: {run_idx}\n")
            f.write("\n" + "-" * 80 + "\n")
            f.write("ERROR MESSAGE:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{error_message}\n")
            f.write("\n" + "-" * 80 + "\n")
            f.write("FULL TRACEBACK:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{error_traceback}\n")
            if context:
                f.write("\n" + "-" * 80 + "\n")
                f.write("ADDITIONAL CONTEXT:\n")
                f.write("-" * 80 + "\n")
                for key, value in context.items():
                    f.write(f"{key}: {value}\n")
            f.write("=" * 80 + "\n")
        
        json_log_filepath = log_filepath.replace('.log', '.json')
        with open(json_log_filepath, 'w') as f:
            json.dump(log_content, f, indent=2)
    except Exception as e:
        print(f"[Warning] Failed to write error log file: {e}")


def ensure_directories():
    """Ensure all output directories exist."""
    dirs = [
        config.OUTPUT_DIR_BASE,
        config.OUTPUT_DIR_PLOTS,
        config.OUTPUT_DIR_TABLES,
        config.OUTPUT_DIR_RESULTS_CSV,
        config.OUTPUT_DIR_POSTERIOR_SAMPLES,
        config.OUTPUT_DIR_BENCHMARK_RESULTS,
        config.OUTPUT_DIR_POSTERIOR_SUMMARIES,
        config.OUTPUT_DIR_RUN_METRICS_JSON,
        config.OUTPUT_DIR_LOGS
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"[System] Output directories ready: {config.OUTPUT_DIR_BASE}")


def clear_directories():
    """Clear all output directories."""
    if os.path.exists(config.OUTPUT_DIR_BASE):
        print(f"[System] Cleaning directory: {config.OUTPUT_DIR_BASE} ...")
        shutil.rmtree(config.OUTPUT_DIR_BASE)
    ensure_directories()


def save_results(scenario_id, run_idx, posterior_samples, benchmarks, metrics, elapsed_time):
    """Save results for a single run."""
    metrics_file = os.path.join(config.OUTPUT_DIR_RUN_METRICS_JSON, f"{scenario_id}_run_{run_idx}_metrics.json")
    
    serializable_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer, int)):
            serializable_metrics[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            serializable_metrics[k] = float(v)
        elif isinstance(v, (np.ndarray, list)):
            serializable_metrics[k] = np.array(v).tolist()
        else:
            serializable_metrics[k] = v
    
    serializable_metrics['elapsed_time_seconds'] = elapsed_time
    
    with open(metrics_file, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    
    # Save posterior samples based on configuration
    if config.SAVE_RAW_POSTERIOR_SAMPLES:
        post_file = os.path.join(config.OUTPUT_DIR_POSTERIOR_SAMPLES, f"{scenario_id}_run_{run_idx}_posterior.npz")
        if config.SAVE_ONLY_KEY_PARAMETERS:
            # Only save key parameters, exclude time series to save space
            # Key parameters: beta_abs, beta_slope_abs, sigma_delta, tau_alpha, lambda
            key_params = {}
            key_param_names = ['beta_abs', 'beta_slope_abs', 'sigma_delta', 'tau_alpha', 'lambda']
            for param_name in key_param_names:
                if param_name in posterior_samples:
                    key_params[param_name] = posterior_samples[param_name]
            if key_params:
                np.savez_compressed(post_file, **key_params)
        else:
            # Save all posterior samples
            np.savez_compressed(post_file, **posterior_samples)
    # If SAVE_RAW_POSTERIOR_SAMPLES is False, skip saving (summaries are already saved)
    
    bench_file = os.path.join(config.OUTPUT_DIR_BENCHMARK_RESULTS, f"{scenario_id}_run_{run_idx}_benchmarks.npz")
    bench_arrays = {k: v for k, v in benchmarks.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(bench_file, **bench_arrays)


# =============================================================================
# Analysis Functions
# =============================================================================

def sanitize_metrics_dataframe(df):
    """Clean DataFrame by converting list-like values to scalars."""
    for col in df.columns:
        if df[col].dtype == 'object':
            is_list_like = df[col].notna().any() and isinstance(df[col].dropna().iloc[0], list)
            if is_list_like:
                df[col] = df[col].apply(
                    lambda x: x[0] if isinstance(x, list) and len(x) == 1 else (np.nan if isinstance(x, list) else x)
                ).astype(float, errors='ignore')
    return df


def prepare_aggregated_plot_data(results_df_all):
    """Aggregate time-series results for summary plots."""
    aggregated_plot_data_list = []
    study_global_seed = config.GLOBAL_BASE_SEED
    
    for scenario_idx, scenario_config in enumerate(tqdm(config.SCENARIOS, desc="Aggregating Plot Data")):
        scenario_id = scenario_config["id"]
        scenario_base_seed = study_global_seed + (scenario_idx * config.NUM_MONTE_CARLO_RUNS * 1000)
        
        sim_data_true = data_generation.simulate_scenario_data(scenario_config, run_seed=scenario_base_seed)
        T_analyze = config.T_ANALYSIS_LENGTH
        
        scenario_mask = results_df_all["scenario_id"] == scenario_id
        if 'error' in results_df_all.columns:
            error_mask = results_df_all["error"].isin([None, "None"])
            scen_df_valid = results_df_all[scenario_mask & error_mask]
        else:
            scen_df_valid = results_df_all[scenario_mask]
        if scen_df_valid.empty:
            continue
        
        series_data = {key: [] for key in ['sCFR_mean', 'sCFR_lower', 'sCFR_upper',
                                           'sCFR_cf_mean', 'sCFR_cf_lower', 'sCFR_cf_upper',
                                           'cCFR_mean', 'cCFR_lower', 'cCFR_upper',
                                           'aCFR_mean', 'aCFR_lower', 'aCFR_upper',
                                           'fsCFR_factual_mean', 'fsCFR_factual_lower', 'fsCFR_factual_upper',
                                           'fsCFR_cf_mean', 'fsCFR_cf_lower', 'fsCFR_cf_upper']}
        
        for mc_run_idx in scen_df_valid["mc_run"].astype(int) - 1:
            post_file = os.path.join(config.OUTPUT_DIR_POSTERIOR_SUMMARIES, f"{scenario_id}_run_{mc_run_idx}_posterior_summary.json")
            if os.path.exists(post_file):
                with open(post_file, 'r') as f:
                    posterior_summary = json.load(f)
                
                series_data['sCFR_mean'].append(posterior_summary.get("p_mean", []))
                series_data['sCFR_lower'].append(posterior_summary.get("p_q025", []))
                series_data['sCFR_upper'].append(posterior_summary.get("p_q975", []))
                series_data['sCFR_cf_mean'].append(posterior_summary.get("p_cf_mean", []))
                series_data['sCFR_cf_lower'].append(posterior_summary.get("p_cf_q025", []))
                series_data['sCFR_cf_upper'].append(posterior_summary.get("p_cf_q975", []))
            
            bench_file = os.path.join(config.OUTPUT_DIR_BENCHMARK_RESULTS, f"{scenario_id}_run_{mc_run_idx}_benchmarks.npz")
            if os.path.exists(bench_file):
                bench_data = np.load(bench_file)
                series_data['cCFR_mean'].append(bench_data.get("cCFR_model", []))
                series_data['cCFR_lower'].append(bench_data.get("cCFR_model_lower", []))
                series_data['cCFR_upper'].append(bench_data.get("cCFR_model_upper", []))
                series_data['aCFR_mean'].append(bench_data.get("aCFR_model", []))
                series_data['aCFR_lower'].append(bench_data.get("aCFR_model_lower", []))
                series_data['aCFR_upper'].append(bench_data.get("aCFR_model_upper", []))
                series_data['fsCFR_factual_mean'].append(bench_data.get("fsCFR_factual_mean", []))
                series_data['fsCFR_factual_lower'].append(bench_data.get("fsCFR_factual_lower", []))
                series_data['fsCFR_factual_upper'].append(bench_data.get("fsCFR_factual_upper", []))
                series_data['fsCFR_cf_mean'].append(bench_data.get("fsCFR_counterfactual_mean", []))
                series_data['fsCFR_cf_lower'].append(bench_data.get("fsCFR_counterfactual_lower", []))
                series_data['fsCFR_cf_upper'].append(bench_data.get("fsCFR_counterfactual_upper", []))
        
        def safe_mean_and_slice(data_list, max_len):
            default_empty = np.full(max_len, np.nan)
            if not data_list:
                return default_empty
            valid_data = [np.array(s) for s in data_list if len(s) > 0]
            if not valid_data:
                return default_empty
            try:
                mean_result = np.mean(valid_data, axis=0)
                if np.isscalar(mean_result):
                    result = np.full(max_len, mean_result)
                else:
                    result = mean_result[:max_len] if len(mean_result) > max_len else mean_result
                    if len(result) < max_len:
                        padded = np.full(max_len, np.nan)
                        padded[:len(result)] = result
                        result = padded
                return result
            except (ValueError, IndexError):
                return default_empty
        
        agg_plot_dict = {
            "scenario_id": scenario_id,
            "true_r_t": sim_data_true["true_r_0_t"][:T_analyze],
            "true_rcf_0_t": sim_data_true["true_rcf_0_t"][:T_analyze],
            "true_intervention_times_0_abs": sim_data_true["true_intervention_times_0_abs"],
            "true_zeta_0_t": sim_data_true["true_zeta_0_t"][:T_analyze],
            "true_eta_0_t": sim_data_true["true_eta_0_t"][:T_analyze],
            "estimated_r_t_dict": {
                "sCFR": {k.replace('sCFR_', ''): safe_mean_and_slice(series_data[k], T_analyze) for k in series_data if 'sCFR' in k},
                "cCFR_model": {k.replace('cCFR_', ''): safe_mean_and_slice(series_data[k], T_analyze) for k in series_data if 'cCFR' in k},
                "aCFR_model": {k.replace('aCFR_', ''): safe_mean_and_slice(series_data[k], T_analyze) for k in series_data if 'aCFR' in k},
                "fsCFR_model": {k.replace('fsCFR_', ''): safe_mean_and_slice(series_data[k], T_analyze) for k in series_data if 'fsCFR' in k}
            }
        }
        
        aggregated_plot_data_list.append(agg_plot_dict)
    
    return aggregated_plot_data_list


def generate_posterior_summary(posterior_samples, scenario_id, run_idx):
    """Generate summary statistics for posterior samples."""
    summary = {}
    
    p_key = "r_t" if "r_t" in posterior_samples else "p"
    p_cf_key = "r_cf" if "r_cf" in posterior_samples else "p_cf"
    if p_key in posterior_samples:
        p_samples = posterior_samples[p_key]
        summary['p_mean'] = np.mean(p_samples, axis=0).tolist()
        summary['p_q025'] = np.percentile(p_samples, 2.5, axis=0).tolist()
        summary['p_q975'] = np.percentile(p_samples, 97.5, axis=0).tolist()
        
        if p_cf_key in posterior_samples:
            p_cf_samples = posterior_samples[p_cf_key]
            summary['p_cf_mean'] = np.mean(p_cf_samples, axis=0).tolist()
            summary['p_cf_q025'] = np.percentile(p_cf_samples, 2.5, axis=0).tolist()
            summary['p_cf_q975'] = np.percentile(p_cf_samples, 97.5, axis=0).tolist()
    
    summary_file = os.path.join(config.OUTPUT_DIR_POSTERIOR_SUMMARIES, f"{scenario_id}_run_{run_idx}_posterior_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    return summary


def run_analysis():
    """Run analysis on existing simulation results."""
    print("=" * 60)
    print("ANALYSIS MODE")
    print("=" * 60)
    
    analysis_start_time = time.time()
    all_loaded_metrics = []
    analysis_errors = []
    
    print("Starting analysis of existing simulation results...")
    
    for dir_path in [config.OUTPUT_DIR_PLOTS, config.OUTPUT_DIR_TABLES, config.OUTPUT_DIR_RESULTS_CSV, config.OUTPUT_DIR_LOGS]:
        os.makedirs(dir_path, exist_ok=True)
    
    for scenario in tqdm(config.SCENARIOS, desc="Loading All Metrics"):
        for mc_run in range(config.NUM_MONTE_CARLO_RUNS):
            metrics_file = os.path.join(config.OUTPUT_DIR_RUN_METRICS_JSON, f"{scenario['id']}_run_{mc_run}_metrics.json")
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        run_metrics = json.load(f)
                    all_loaded_metrics.append(run_metrics)
                except Exception as e:
                    error_msg = f"Failed to load metrics file {metrics_file}: {str(e)}"
                    print(f"[Warning] {error_msg}")
                    analysis_errors.append({"file": metrics_file, "error": error_msg, "step": "load_metrics"})
    
    if not all_loaded_metrics:
        print("No metrics files found. Cannot generate plots or tables.")
        return
        
    results_df_all = pd.DataFrame(all_loaded_metrics)
    results_df_valid = sanitize_metrics_dataframe(results_df_all)
    
    if 'error' in results_df_valid.columns:
        results_df_valid = results_df_valid[results_df_valid['error'].isin([None, "None"])].copy()
    
    if results_df_valid.empty:
        print("No valid simulation runs found. Analysis cannot proceed.")
        return
    
    cover_cols = [col for col in results_df_valid.columns if 'cover' in col]
    for col in cover_cols: 
        # Convert to float first, then round and convert to Int64 to avoid unsafe cast error
        # This handles float values like 0.0, 1.0 that need to be converted to integers
        results_df_valid[col] = pd.to_numeric(results_df_valid[col], errors='coerce').round().astype('Int64')
    
    summary_mean = results_df_valid.groupby("scenario_id").mean(numeric_only=True).add_suffix('_mean').reset_index().rename(columns={'scenario_id_mean':'scenario_id'})
    summary_std = results_df_valid.groupby("scenario_id").std(numeric_only=True).add_suffix('_std').reset_index().rename(columns={'scenario_id_std':'scenario_id'})
    results_df_summary = pd.merge(summary_mean, summary_std, on="scenario_id", how="left")
    
    analysis_csv_path = os.path.join(config.OUTPUT_DIR_RESULTS_CSV, "all_scenarios_metrics_aggregated.csv")
    results_df_summary.to_csv(analysis_csv_path, index=False)
    print(f"\nAggregated summary metrics saved to {analysis_csv_path}")
    
    try:
        print("\nPreparing aggregated data for summary plots...")
        aggregated_plot_data = prepare_aggregated_plot_data(results_df_all)
    except Exception as e:
        error_msg = f"Failed to prepare aggregated plot data: {str(e)}"
        print(f"[Error] {error_msg}")
        traceback.print_exc()
        log_error_to_file("ALL", -1, "analysis", error_msg, traceback.format_exc(), {"step": "prepare_aggregated_plot_data"})
        analysis_errors.append({"step": "prepare_aggregated_plot_data", "error": error_msg})
        aggregated_plot_data = []
    
    plot_functions = [
        ("Generating aggregated factual summary plot...", ev.plot_aggregated_factual_summary, "plot_aggregated_factual_summary"),
        ("Generating aggregated counterfactual summary plot...", ev.plot_aggregated_counterfactual_summary, "plot_aggregated_counterfactual_summary"),
        ("Generating effectiveness summary plot...", ev.plot_effectiveness_summary, "plot_effectiveness_summary"),
        ("Generating summary boxplots...", ev.plot_metric_summary_boxplots, "plot_metric_summary_boxplots"),
        ("Generating combined metrics summary...", ev.plot_combined_metrics_summary, "plot_combined_metrics_summary"),
    ]
    
    for msg, func, step_name in plot_functions:
        try:
            print(msg)
            if "aggregated" in step_name or "effectiveness" in step_name:
                func(aggregated_plot_data, config.OUTPUT_DIR_PLOTS)
            else:
                func(results_df_valid, config.OUTPUT_DIR_PLOTS)
        except Exception as e:
            error_msg = f"Failed to generate {step_name}: {str(e)}"
            print(f"[Error] {error_msg}")
            traceback.print_exc()
            log_error_to_file("ALL", -1, "analysis", error_msg, traceback.format_exc(), {"step": step_name})
            analysis_errors.append({"step": step_name, "error": error_msg})
    
    if analysis_errors:
        print(f"\n[Warning] Analysis completed with {len(analysis_errors)} error(s).")
        print("Error details saved to log files in:", config.OUTPUT_DIR_LOGS)
    else:
        print("\nAnalysis complete.")
    
    analysis_summary = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_time_seconds": time.time() - analysis_start_time,
        "total_metrics_loaded": len(all_loaded_metrics),
        "valid_runs": len(results_df_valid),
        "errors_count": len(analysis_errors),
        "errors": analysis_errors
    }
    summary_file = os.path.join(config.OUTPUT_DIR_LOGS, "analysis_summary.json")
    try:
        with open(summary_file, 'w') as f:
            json.dump(analysis_summary, f, indent=2)
    except Exception as e:
        print(f"[Warning] Failed to save analysis summary: {e}")


# =============================================================================
# Computational Experiment (Section 3.5: cost, scalability, online updating)
# =============================================================================
#
# This is a TIMING experiment. To keep the wall-clock measurement valid while still
# running in parallel, each worker is pinned to a single dedicated core (via CPU
# affinity) and every numerical library runs single-threaded, so a timed fit has a
# core to itself and does not contend with the other workers for CPU time. The
# experiment is one-click, shows a progress bar, and is fully checkpointed/resumable
# (one cache file per replicate). Compilation cost is removed by a single throwaway
# fit per (T, K) setting in each worker, plus median/IQR aggregation that is robust
# to any stray compile-inflated replicate. Replicate seeds vary, so each replicate
# is a distinct data realization (Plan A). Residual shared-resource effects (turbo
# clocking, memory bandwidth) are second-order and consistent across settings, so
# the scaling trends and the sCFR/fsCFR comparison are preserved; for the cleanest
# absolute times, use fewer workers via --jobs.

RUNTIME_CACHE_DIR = os.path.join(config.OUTPUT_DIR_TABLES, "runtime_cache")
RUNTIME_T_FOR_K = 200          # series length held fixed while K is varied
RUNTIME_T_GRID_FULL = [100, 200, 400, 800, 1200]
RUNTIME_K_GRID_FULL = [1, 2, 4, 8]
RUNTIME_T_GRID_DEMO = [100, 200, 400, 800, 1200]   # same grid as full; demo differs only in replications
RUNTIME_K_GRID_DEMO = [1, 2, 4, 8]


def _runtime_reps_for_T(T, demo=False):
    """Replicate datasets per setting: 10 in demo, 100 at full scale (run on the server)."""
    return 10 if demo else 100


def _runtime_build_data(T, K, seed):
    """Build one synthetic dataset and the model design objects at length T,
    with K interventions. Seed-varying so replicates are distinct realizations.
    """
    T_sim = T + config.T_SIMULATION_BUFFER
    rng = np.random.default_rng(seed)

    c_t_full = np.maximum(
        config.C_T_VSHAPE_MAX_CASES
        - config.C_T_VSHAPE_SLOPE * np.abs(np.arange(T_sim) - config.C_T_VSHAPE_PEAK_TIME_FACTOR * T),
        config.MIN_DRAWN_CASES,
    ).astype(float)

    f_s = data_generation.generate_delay_distribution(T_sim, config.F_MEAN, config.F_SHAPE)
    Q_full = data_generation.construct_Q_matrix(c_t_full, f_s, T_sim)
    Bm_full = data_generation.generate_bspline_basis(T_sim, config.N_SPLINE_KNOTS_J, config.SPLINE_ORDER)
    zeta_full = data_generation.generate_baseline_cfr_zeta(T_sim, T, "C1", {"cfr_const": 0.02})

    if K > 0:
        int_times = np.linspace(0.2 * T, 0.8 * T, K)
        Z_full = data_generation.generate_intervention_input_matrix(
            np.arange(T_sim, dtype=float), int_times, K)
    else:
        int_times = np.array([], dtype=float)
        Z_full = np.empty((T_sim, 0))

    r0_full = 1.0 / (1.0 + np.exp(-zeta_full))
    mu0 = np.maximum(Q_full @ r0_full, 1e-9)
    d_t_full = rng.poisson(mu0)

    return dict(
        d_t=d_t_full[:T], c_t=c_t_full[:T], Q=Q_full[:T, :T],
        Bm=Bm_full[:T, :], Z=Z_full[:T, :], f_s=f_s[:T],
        int_times=int_times, T=T, K=K,
        beta_signs=(np.array([-1.0] * K) if K > 0 else None),
    )


def _runtime_time_scfr(data, warmup, samples, seed):
    """Time a single sCFR NUTS fit (seconds)."""
    model_data = dict(
        dt=jnp.array(data["d_t"].astype(float)),
        fc_mat=jnp.array(data["Q"].astype(float)),
        Bm=jnp.array(data["Bm"].astype(float)),
        Z=jnp.array(data["Z"].astype(float)),
        beta_signs=(jnp.array(data["beta_signs"]) if data["beta_signs"] is not None else None),
    )
    K = data["K"]
    init_vals = methods.get_ols_initial_values(
        data["d_t"], data["Q"], data["Z"], data["Bm"],
        beta_signs=(None if K == 0 else np.ones(K)), c_t=data["c_t"])
    if not methods._validate_init_params(init_vals):
        init_vals = None
    t0 = time.perf_counter()
    methods.run_numpyro_sampler(
        model_data=model_data, rng_key=jax.random.PRNGKey(seed),
        num_warmup=warmup, num_samples=samples, num_chains=1, init_params=init_vals)
    return time.perf_counter() - t0


def _runtime_time_fscfr(data):
    """Time a single fsCFR (EM + L-BFGS-B) fit (seconds)."""
    K = data["K"]
    int_times = list(data["int_times"].astype(int)) if K > 0 else []
    signs = ([-1] * K) if K > 0 else []
    t0 = time.perf_counter()
    fsCFR_model_wrapper(
        d_t=data["d_t"], c_t=data["c_t"], f_s=data["f_s"], Bm=data["Bm"],
        intervention_times_abs=int_times, intervention_signs=signs, verbose=False)
    return time.perf_counter() - t0


# Process-local state for the runtime workers: the set of (T,K) already JIT-warmed
# in this worker, and the single core this worker is pinned to.
_RT_WARMED = set()
_RT_CORE = None


def _runtime_cache(kind, T, K, rep):
    return os.path.join(RUNTIME_CACHE_DIR, f"{kind}_T{T}_K{K}_rep{rep}.json")


def _pin_to_dedicated_core(ctr, lock, n_cores):
    """Pin this worker process to a single, distinct core (once). A timed fit then
    runs on a dedicated core with no scheduling contention from the other parallel
    workers, so its wall-clock measurement stays valid. No-op where CPU affinity is
    unavailable (e.g. Windows); there the local demo timing is approximate."""
    global _RT_CORE
    if _RT_CORE is not None or not hasattr(os, "sched_setaffinity"):
        return
    with lock:
        _RT_CORE = ctr.value % max(int(n_cores), 1)
        ctr.value += 1
    try:
        os.sched_setaffinity(0, {_RT_CORE})
    except OSError:
        pass


def _runtime_one(kind, T, K, rep, warmup, samples, ctr, lock, n_cores):
    """One timed (sCFR, fsCFR) fit for the runtime experiment (cache-or-compute).
    The worker pins itself to a dedicated core and runs single-threaded, and a
    throwaway warmup fit per (T,K) absorbs JIT compilation before timing."""
    cache = _runtime_cache(kind, T, K, rep)
    if os.path.exists(cache):
        return
    _pin_to_dedicated_core(ctr, lock, n_cores)
    seed = 1_000_000 + T * 1000 + K * 100 + rep
    data = _runtime_build_data(T, K, seed)
    if (T, K) not in _RT_WARMED:
        try:
            _runtime_time_scfr(data, min(warmup, 50), min(samples, 50), seed)
            _runtime_time_fscfr(data)
        except Exception as e:
            print(f"[Runtime] warmup ({T},{K}) failed: {e}")
        _RT_WARMED.add((T, K))
    try:
        scfr_t = _runtime_time_scfr(data, warmup, samples, seed)
        fscfr_t = _runtime_time_fscfr(data)
    except Exception as e:
        print(f"[Runtime] fit ({kind},T={T},K={K},rep={rep}) failed: {e}")
        return
    with open(cache, 'w') as fh:
        json.dump(dict(kind=kind, T=T, K=K, rep=rep,
                       scfr_time=scfr_t, fscfr_time=fscfr_t), fh)


def run_runtime_experiment(demo=False, warmup=None, samples=None, n_jobs=None):
    """Run the Section 3.5 computational experiment, parallel and cached. Each timed
    fit runs single-threaded on a dedicated, pinned core, so parallelism removes
    scheduling contention and the per-fit wall-clock measurement stays valid."""
    os.makedirs(RUNTIME_CACHE_DIR, exist_ok=True)
    warmup = warmup if warmup is not None else config.NUM_WARMUP
    samples = samples if samples is not None else config.NUM_SAMPLES
    n_jobs = n_jobs if n_jobs is not None else config.NUM_CORES_TO_USE
    T_grid = RUNTIME_T_GRID_DEMO if demo else RUNTIME_T_GRID_FULL
    K_grid = RUNTIME_K_GRID_DEMO if demo else RUNTIME_K_GRID_FULL

    # (kind, T, K): scale_T sweeps T at K=1; scale_K sweeps K at T=RUNTIME_T_FOR_K.
    settings = [("scale_T", T, 1) for T in T_grid]
    settings += [("scale_K", RUNTIME_T_FOR_K, K) for K in K_grid if K != 1]
    tasks = [(kind, T, K, rep) for kind, T, K in settings
             for rep in range(_runtime_reps_for_T(T, demo))]
    todo = [t for t in tasks if not os.path.exists(_runtime_cache(*t))]

    n_cores = (os.cpu_count() or 1) if n_jobs in (None, -1) else max(int(n_jobs), 1)
    print(f"[Runtime] {len(tasks)} timed fits (NUTS {warmup}+{samples}); "
          f"{len(tasks) - len(todo)} cached, {len(todo)} to run. Parallel with each fit "
          f"pinned to a dedicated core (single-threaded) for valid wall-clock timing.")
    if todo:
        mgr = multiprocessing.Manager()
        ctr = mgr.Value('i', 0)
        lock = mgr.Lock()
        Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_runtime_one)(kind, T, K, rep, warmup, samples, ctr, lock, n_cores)
            for kind, T, K, rep in tqdm(todo, desc="Runtime experiment", unit="fit"))

    aggregate_and_plot_runtime(demo=demo)


def _runtime_load_cache():
    """Load all cached replicate timings into a DataFrame."""
    rows = []
    if not os.path.isdir(RUNTIME_CACHE_DIR):
        return pd.DataFrame(rows)
    for fn in os.listdir(RUNTIME_CACHE_DIR):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(RUNTIME_CACHE_DIR, fn)) as fh:
                    rows.append(json.load(fh))
            except Exception:
                pass
    return pd.DataFrame(rows)


def aggregate_and_plot_runtime(demo=False):
    """Aggregate cached timings (median, IQR) and draw the two-panel figure."""
    df = _runtime_load_cache()
    if df.empty:
        print("[Runtime] no cached timings found; run with --runtime first.")
        return

    def agg(sub):
        out = {}
        for m in ("scfr_time", "fscfr_time"):
            v = sub[m].to_numpy(dtype=float)
            out[m + "_med"] = float(np.median(v))
            out[m + "_lo"] = float(np.percentile(v, 25))
            out[m + "_hi"] = float(np.percentile(v, 75))
        out["n"] = len(sub)
        return out

    # Panel A: scale in T (K=1). Panel B: scale in K (T fixed); K=1 reused from A.
    A = (df[df["kind"] == "scale_T"].groupby("T").apply(agg, include_groups=False).to_dict())
    A_T = sorted(A.keys())
    base_K1 = df[(df["kind"] == "scale_T") & (df["T"] == RUNTIME_T_FOR_K)]
    dfK = df[df["kind"] == "scale_K"].copy()
    if not base_K1.empty:
        b = base_K1.copy(); b["kind"] = "scale_K"; b["K"] = 1
        dfK = pd.concat([dfK, b], ignore_index=True)
    B = (dfK.groupby("K").apply(agg, include_groups=False).to_dict())
    B_K = sorted(B.keys())

    # write a small summary CSV (transparency; not a paper table)
    summ = []
    for T in A_T:
        a = A[T]
        summ.append(dict(panel="T", T=T, K=1, n=a["n"],
                         scfr_med=a["scfr_time_med"], fscfr_med=a["fscfr_time_med"]))
    for K in B_K:
        b = B[K]
        summ.append(dict(panel="K", T=RUNTIME_T_FOR_K, K=K, n=b["n"],
                         scfr_med=b["scfr_time_med"], fscfr_med=b["fscfr_time_med"]))
    os.makedirs(config.OUTPUT_DIR_TABLES, exist_ok=True)
    pd.DataFrame(summ).to_csv(
        os.path.join(config.OUTPUT_DIR_TABLES, "runtime_summary.csv"), index=False)

    _plot_runtime_figure(A, A_T, B, B_K)


def _plot_runtime_figure(A, A_T, B, B_K):
    """Two-panel runtime figure styled to match the other simulation figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c_s = ev.METHOD_COLORS["sCFR"]
    c_f = ev.METHOD_COLORS["fsCFR"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    def err(meds, los, his):
        meds = np.array(meds); los = np.array(los); his = np.array(his)
        return meds, np.vstack([meds - los, his - meds])

    # Panel A: time vs T (K=1)
    sm, se = err([A[t]["scfr_time_med"] for t in A_T],
                 [A[t]["scfr_time_lo"] for t in A_T],
                 [A[t]["scfr_time_hi"] for t in A_T])
    fm, fe = err([A[t]["fscfr_time_med"] for t in A_T],
                 [A[t]["fscfr_time_lo"] for t in A_T],
                 [A[t]["fscfr_time_hi"] for t in A_T])
    ax = axes[0]
    ax.errorbar(A_T, sm, yerr=se, fmt="o-", color=c_s, capsize=4, lw=2, label="sCFR")
    ax.errorbar(A_T, fm, yerr=fe, fmt="s-", color=c_f, capsize=4, lw=2, label="fsCFR")
    ax.set_xlabel("Series length $T$ (days)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Time per fit (s)", fontsize=16, fontweight="bold")
    ax.set_title("(A) Varying $T$ (at $K=1$)", fontsize=18, fontweight="bold")
    ax.legend(loc="best", fontsize=14, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel B: time vs K (T fixed)
    sm, se = err([B[k]["scfr_time_med"] for k in B_K],
                 [B[k]["scfr_time_lo"] for k in B_K],
                 [B[k]["scfr_time_hi"] for k in B_K])
    fm, fe = err([B[k]["fscfr_time_med"] for k in B_K],
                 [B[k]["fscfr_time_lo"] for k in B_K],
                 [B[k]["fscfr_time_hi"] for k in B_K])
    ax = axes[1]
    ax.errorbar(B_K, sm, yerr=se, fmt="o-", color=c_s, capsize=4, lw=2, label="sCFR")
    ax.errorbar(B_K, fm, yerr=fe, fmt="s-", color=c_f, capsize=4, lw=2, label="fsCFR")
    ax.set_xlabel("Number of interventions $K$", fontsize=16, fontweight="bold")
    ax.set_ylabel("Time per fit (s)", fontsize=16, fontweight="bold")
    ax.set_title(f"(B) Varying $K$ (at $T={RUNTIME_T_FOR_K}$)", fontsize=18, fontweight="bold")
    ax.set_xticks(B_K)
    ax.legend(loc="best", fontsize=14, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(config.OUTPUT_DIR_PLOTS, exist_ok=True)
    png = os.path.join(config.OUTPUT_DIR_PLOTS, "runtime_scaling.png")
    pdf = os.path.join(config.OUTPUT_DIR_PLOTS, "runtime_scaling.pdf")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    # Mirror into docs/figs_tables/ so refresh_docs.py (and a plain pdflatex) can
    # pick it up without a separate copy step.
    figs_dir = os.path.join("docs", "figs_tables")
    try:
        os.makedirs(figs_dir, exist_ok=True)
        shutil.copy(pdf, os.path.join(figs_dir, "runtime_scaling.pdf"))
    except Exception:
        pass
    print(f"[Runtime] saved {pdf} (+ png, + docs/figs_tables/runtime_scaling.pdf)")


# =============================================================================
# Sensitivity experiments on a simulation scenario (R2-2 knots, R2-3 priors)
# =============================================================================
#
# Both run on simulation data, where the true intervention magnitudes are known,
# so the tables show stability across knots / priors AND closeness to the truth.
# One sCFR (and, for knots, fsCFR) fit per configuration, sequential, checkpointed
# (one cache file per configuration), with a tqdm bar. Output: a LaTeX table under
# simulation_outputs/tables/, ready for refresh_docs.py.

SENS_SCENARIO_ID = "S09"          # sinusoidal baseline, K=2, sigma_u=0.10
# Knot grid brackets the values actually used (sim J=10, real-data J=20); going far
# beyond this (e.g. 1 knot per ~10 days) over-knots the T~260 series and lets the
# spline leak with the intervention near a change-point.
SENS_J_GRID = [5, 10, 15, 20, 30]
SENS_SEED = config.GLOBAL_BASE_SEED + 777
SENS_CACHE_DIR = os.path.join(config.OUTPUT_DIR_TABLES, "sensitivity_cache")

# Prior configurations: (label, |beta| log-scale, sigma_u HalfCauchy scale, tau Gamma conc, tau Gamma rate)
def _prior_base():
    """Baseline prior configuration: beta=(family,param), sigma=(family,param),
    tau=(Gamma conc, rate), alpha_sd (Normal sd on spline coefficients)."""
    return dict(group="Baseline", beta=("lognormal", 0.5), sigma=("halfcauchy", 0.1),
                tau=(0.01, 0.01), alpha_sd=5.0)


def _build_prior_configs():
    """Comprehensive prior-sensitivity grid (R2-3): vary each prior one at a time,
    over both scale and distributional family, around the baseline."""
    cfgs = [("Baseline", _prior_base())]

    def mk(label, group, **ov):
        c = _prior_base(); c["group"] = group; c.update(ov); cfgs.append((label, c))

    # (A) intervention-magnitude prior |beta|
    mk("Tighter: $\\mathcal{LN}(\\log0.5,0.25)$", "|beta| prior", beta=("lognormal", 0.25))
    mk("Diffuse: $\\mathcal{LN}(\\log0.5,1.0)$", "|beta| prior", beta=("lognormal", 1.0))
    mk("Half-normal: $\\mathcal{HN}(1.0)$", "|beta| prior", beta=("halfnormal", 1.0))
    # (B) random-effect scale prior sigma_u
    mk("Tighter: $C^+(0,0.05)$", "sigma_u prior", sigma=("halfcauchy", 0.05))
    mk("Wider: $C^+(0,0.5)$", "sigma_u prior", sigma=("halfcauchy", 0.5))
    mk("Half-normal: $\\mathcal{HN}(0.1)$", "sigma_u prior", sigma=("halfnormal", 0.1))
    mk("Exponential: $\\mathrm{Exp}(10)$", "sigma_u prior", sigma=("exponential", 10.0))
    # (C) spline-precision prior tau_alpha
    mk("Stronger: $\\Gamma(1,0.01)$", "tau_alpha prior", tau=(1.0, 0.01))
    mk("Vaguer: $\\Gamma(0.001,0.001)$", "tau_alpha prior", tau=(0.001, 0.001))
    mk("$\\Gamma(0.1,0.1)$", "tau_alpha prior", tau=(0.1, 0.1))
    # (D) spline-coefficient prior alpha
    mk("Tighter: $\\mathcal{N}(0,2^2)$", "alpha prior", alpha_sd=2.0)
    mk("Wider: $\\mathcal{N}(0,10^2)$", "alpha prior", alpha_sd=10.0)
    return cfgs


SENS_PRIOR_CONFIGS = _build_prior_configs()


def _prior_sig(cfg):
    """Order-independent cache id derived from the prior configuration content, so
    adding or removing a configuration does not invalidate the others' caches."""
    b, s, t = cfg["beta"], cfg["sigma"], cfg["tau"]
    return f"b{b[0]}{b[1]}-s{s[0]}{s[1]}-t{t[0]}{t[1]}-a{cfg['alpha_sd']}".replace(" ", "")


def _sens_scenario():
    for s in config.SCENARIOS:
        if s["id"] == SENS_SCENARIO_ID:
            return s
    raise ValueError(f"scenario {SENS_SCENARIO_ID} not found")


def _sens_data_for_sampler(sim_data, Bm):
    """Build the NUTS data dict and OLS init for a given B-spline basis Bm."""
    K = sim_data["num_interventions_true_K"]
    Z_input = sim_data["Z_input_true"] if K > 0 else np.empty((len(sim_data["d_t"]), 0))
    bsigns = jnp.array(sim_data["beta_signs_true"]) if (K > 0 and "beta_signs_true" in sim_data) else None
    data = dict(dt=jnp.array(sim_data["d_t"]), fc_mat=jnp.array(sim_data["Q_true"]),
                Bm=jnp.array(Bm), Z=jnp.array(Z_input), beta_signs=bsigns)
    try:
        init = methods.get_ols_initial_values(
            sim_data["d_t"], sim_data["Q_true"], Z_input, Bm,
            beta_signs=sim_data.get("beta_signs_true"), c_t=sim_data.get("c_t"))
        if not methods._validate_init_params(init):
            init = None
    except Exception:
        init = None
    return data, init


def _beta_dist(family, param, K):
    if family == "lognormal":
        return dist.LogNormal(jnp.log(0.5), param).expand([K]).to_event(1)
    if family == "halfnormal":
        return dist.HalfNormal(param).expand([K]).to_event(1)
    raise ValueError(family)


def _sigma_dist(family, param):
    if family == "halfcauchy":
        return dist.HalfCauchy(param)
    if family == "halfnormal":
        return dist.HalfNormal(param)
    if family == "exponential":
        return dist.Exponential(param)
    raise ValueError(family)


def _sens_prior_model(cfg):
    """sCFR model with a fully overridable prior configuration cfg (for R2-3):
    cfg['beta']=(family,param), cfg['sigma']=(family,param), cfg['tau']=(conc,rate),
    cfg['alpha_sd']=Normal sd on the spline coefficients."""
    bfam, bpar = cfg["beta"]; sfam, spar = cfg["sigma"]
    tconc, trate = cfg["tau"]; asd = cfg["alpha_sd"]

    def model(data):
        dt, fc_mat, Bm, Z = data['dt'], data['fc_mat'], data['Bm'], data['Z']
        beta_signs = data.get('beta_signs', None)
        T, J = dt.shape[0], Bm.shape[1]
        alpha = numpyro.sample("alpha", dist.Normal(0.0, asd).expand([J]).to_event(1))
        tau_alpha = numpyro.sample("tau_alpha", dist.Gamma(tconc, trate))
        if J >= 3:
            d2 = alpha[2:] - 2.0 * alpha[1:-1] + alpha[:-2]
            numpyro.factor("rw2_alpha_penalty", -0.5 * tau_alpha * jnp.sum(d2 ** 2))
        main_logit = jnp.dot(Bm, alpha)
        sigma_delta = numpyro.sample("sigma_delta", _sigma_dist(sfam, spar))
        delta_raw = numpyro.sample("delta_raw", dist.Normal(0.0, 1.0).expand([T]).to_event(1))
        delta = sigma_delta * delta_raw
        delta = delta - jnp.mean(delta)
        intervention_logit = 0.0
        if Z.shape[1] > 0 and beta_signs is not None:
            beta_abs = numpyro.sample("beta_abs", _beta_dist(bfam, bpar, Z.shape[1]))
            beta_slope_abs = numpyro.sample("beta_slope_abs", _beta_dist(bfam, bpar, Z.shape[1]))
            denom = jnp.maximum(T - 1, 1)
            Z_hinge = (jnp.cumsum(Z, axis=0) - Z) / denom
            intervention_logit = jnp.dot(Z, beta_abs * beta_signs) + jnp.dot(Z_hinge, beta_slope_abs * beta_signs)
        mu = jnp.maximum(jnp.dot(fc_mat, jax.nn.sigmoid(main_logit + delta + intervention_logit)), 1e-9)
        numpyro.sample("obs_deaths", dist.Poisson(mu), obs=dt)
    return model


def _sens_fit_scfr(data, init, key, warmup, samples, model_fn=None):
    if model_fn is None:
        mcmc = methods.run_numpyro_sampler(data, key, num_warmup=warmup, num_samples=samples,
                                           num_chains=1, init_params=init)
    else:
        # Heavy-tailed prior configs (diffuse |beta|, half-Cauchy sigma) can defeat
        # init_to_median; pin the OLS warm start when available (site names match
        # across families), else fall back to a robust median start.
        if init is not None:
            strat = numpyro.infer.init_to_value(values=init)
        else:
            strat = numpyro.infer.init_to_median
        kernel = NUTS(model_fn, target_accept_prob=0.9, init_strategy=strat)
        mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=1, progress_bar=False)
        mcmc.run(key, data)
    return mcmc.get_samples()


def _ms(vals):
    a = np.asarray(vals, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(a)), float(np.std(a))


def _fmt_ms(ms):
    m, s = ms
    return f"{m:.3f} $\\pm$ {s:.3f}"


def _sens_true(scen):
    return (float(scen["true_beta_abs_0"][0]), float(scen["true_beta_slope_abs_0"][0]),
            float(scen.get("sigma_delta_true", 0.0)))


def _leakage_diag(B, Z):
    """Baseline/intervention leakage diagnostics (R2-2), design-only.
    Returns (L_{Z|B}, max_j L_{j|B}, lambda_min(Z^T M_B Z), kappa(Z^T M_B Z)).
    """
    B = np.asarray(B, float); Z = np.asarray(Z, float)
    P_B = B @ np.linalg.pinv(B.T @ B) @ B.T
    M_B = np.eye(B.shape[0]) - P_B
    PBZ = P_B @ Z
    L = float((np.linalg.norm(PBZ, "fro") ** 2) / (np.linalg.norm(Z, "fro") ** 2))
    Lj = (np.linalg.norm(PBZ, axis=0) ** 2) / np.maximum(np.linalg.norm(Z, axis=0) ** 2, 1e-12)
    G = Z.T @ M_B @ Z
    ev = np.linalg.eigvalsh((G + G.T) / 2.0)
    lmin = float(np.min(ev))
    cond = float(np.max(ev) / lmin) if lmin > 1e-12 else float("inf")
    return L, float(np.max(Lj)), lmin, cond


# Per-task cache paths (one JSON file per fit), shared by the workers and the
# break-point pre-filter so completed fits are skipped on resume.
def _knot_cache(J, rep):
    return os.path.join(SENS_CACHE_DIR, f"knot_{SENS_SCENARIO_ID}_J{J}_rep{rep}.json")


def _prior_cache(i, rep):
    _, cfg = SENS_PRIOR_CONFIGS[i]
    return os.path.join(SENS_CACHE_DIR, f"prior_{SENS_SCENARIO_ID}_{_prior_sig(cfg)}_rep{rep}.json")


def _misspec_cache(kind, rep):
    return os.path.join(MISSPEC_CACHE_DIR, f"ms_{kind}_rep{rep}.json")


def _knot_one(J, rep, warmup, samples):
    """One knot-sensitivity fit (cache-or-compute), safe to run in a joblib worker."""
    cache = _knot_cache(J, rep)
    if os.path.exists(cache):
        return json.load(open(cache))
    scen = _sens_scenario()
    sim = data_generation.simulate_scenario_data(scen, SENS_SEED + 1000 * rep)
    T = len(sim["d_t"])
    Bm = data_generation.generate_bspline_basis(T, J, config.SPLINE_ORDER)
    data, init = _sens_data_for_sampler(sim, Bm)
    s = _sens_fit_scfr(data, init, jax.random.PRNGKey(SENS_SEED + J + rep), warmup, samples)
    fs = fsCFR_model_wrapper(d_t=sim["d_t"], c_t=sim["c_t"], f_s=sim["f_s_true"], Bm=Bm,
                             intervention_times_abs=list(np.asarray(sim["true_intervention_times_0_abs"]).astype(int)),
                             intervention_signs=list(np.asarray(sim["beta_signs_true"]).astype(int)), verbose=False)
    r = dict(J=J, rep=rep,
             scL=float(np.mean(s["beta_abs"][:, 0])), scS=float(np.mean(s["beta_slope_abs"][:, 0])),
             fsL=float(fs["fsCFR_beta_abs_est"][0]), fsS=float(fs["fsCFR_beta_slope_abs_est"][0]))
    json.dump(r, open(cache, "w"))
    return r


def _prior_one(i, rep, warmup, samples):
    """One prior-sensitivity fit (cache-or-compute), safe to run in a joblib worker."""
    _, cfg = SENS_PRIOR_CONFIGS[i]
    cache = _prior_cache(i, rep)
    if os.path.exists(cache):
        return json.load(open(cache))
    scen = _sens_scenario()
    sim = data_generation.simulate_scenario_data(scen, SENS_SEED + 1000 * rep)
    Bm = data_generation.generate_bspline_basis(len(sim["d_t"]), config.N_SPLINE_KNOTS_J, config.SPLINE_ORDER)
    data, init = _sens_data_for_sampler(sim, Bm)
    s = _sens_fit_scfr(data, init, jax.random.PRNGKey(SENS_SEED + 13 * (i + 1) + rep),
                       warmup, samples, model_fn=_sens_prior_model(cfg))
    bL, bS = np.asarray(s["beta_abs"][:, 0]), np.asarray(s["beta_slope_abs"][:, 0])
    r = dict(i=i, rep=rep, L=float(bL.mean()), S=float(bS.mean()),
             su=float(np.mean(s["sigma_delta"])), Lsd=float(bL.std()), Ssd=float(bS.std()))
    json.dump(r, open(cache, "w"))
    return r


def run_knot_sensitivity(demo=False, warmup=None, samples=None, reps=None, n_jobs=None):
    """R2-2: vary spline knots J on a simulation scenario; estimates averaged over
    independent replicate datasets, reported as mean +/- SD against the known truth.
    Fits run in parallel across replicate datasets (joblib)."""
    os.makedirs(SENS_CACHE_DIR, exist_ok=True)
    warmup = warmup if warmup is not None else config.NUM_WARMUP
    samples = samples if samples is not None else config.NUM_SAMPLES
    reps = reps if reps is not None else (10 if demo else config.AUX_NUM_REPLICATIONS)
    n_jobs = n_jobs if n_jobs is not None else config.NUM_CORES_TO_USE
    scen = _sens_scenario()
    true_L, true_S, _ = _sens_true(scen)

    tasks = [(J, rep) for J in SENS_J_GRID for rep in range(reps)]
    # Break-point pre-filter: dispatch only fits without a cache, compute them in
    # parallel, then load every result (cached + new) from cache for aggregation.
    todo = [t for t in tasks if not os.path.exists(_knot_cache(*t))]
    print(f"[knot ] {len(tasks)} fits; {len(tasks) - len(todo)} cached, {len(todo)} to run (n_jobs={n_jobs}).")
    if todo:
        Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_knot_one)(J, rep, warmup, samples)
            for J, rep in tqdm(todo, desc="Knot sensitivity", unit="fit"))
    per_J = {J: {"scL": [], "scS": [], "fsL": [], "fsS": []} for J in SENS_J_GRID}
    for (J, rep) in tasks:
        r = json.load(open(_knot_cache(J, rep)))
        for k in ("scL", "scS", "fsL", "fsS"):
            per_J[J][k].append(r[k])

    # design-only leakage diagnostics per J (R2-2): L_{Z|B}, lambda_min/kappa of Z^T M_B Z
    sim0 = data_generation.simulate_scenario_data(scen, SENS_SEED)
    T0 = len(sim0["d_t"])
    Zfull = np.concatenate([sim0["Z_input_true"], sim0["Z_hinge_true"]], axis=1)
    leak = {J: _leakage_diag(data_generation.generate_bspline_basis(T0, J, config.SPLINE_ORDER), Zfull)
            for J in SENS_J_GRID}
    pd.DataFrame([dict(scenario=SENS_SCENARIO_ID, J=J, L_Z_given_B=leak[J][0],
                       max_L_j_given_B=leak[J][1], lambda_min_ZMBZ=leak[J][2], cond_ZMBZ=leak[J][3])
                  for J in SENS_J_GRID]).to_csv(
        os.path.join(config.OUTPUT_DIR_TABLES, "leakage_diagnostics.csv"), index=False)

    rows = [dict(J=J, scL=_ms(per_J[J]["scL"]), scS=_ms(per_J[J]["scS"]),
                 fsL=_ms(per_J[J]["fsL"]), fsS=_ms(per_J[J]["fsS"]),
                 LZB=leak[J][0], lmin=leak[J][2]) for J in SENS_J_GRID]
    _write_knot_table(rows, true_L, true_S, reps)


def run_prior_sensitivity(demo=False, warmup=None, samples=None, reps=None, n_jobs=None):
    """R2-3: vary priors on a simulation scenario; sCFR posterior averaged over
    independent replicate datasets, reported as mean +/- SD against the known truth.
    Fits run in parallel across the prior-configuration x replicate grid (joblib).
    This is the heaviest auxiliary experiment (many prior configurations), so the
    parallelization gives the largest speed-up."""
    os.makedirs(SENS_CACHE_DIR, exist_ok=True)
    warmup = warmup if warmup is not None else config.NUM_WARMUP
    samples = samples if samples is not None else config.NUM_SAMPLES
    reps = reps if reps is not None else (10 if demo else config.AUX_NUM_REPLICATIONS)
    n_jobs = n_jobs if n_jobs is not None else config.NUM_CORES_TO_USE
    scen = _sens_scenario()
    true_L, true_S, true_su = _sens_true(scen)

    n_cfg = len(SENS_PRIOR_CONFIGS)
    tasks = [(i, rep) for i in range(n_cfg) for rep in range(reps)]
    # Break-point pre-filter: dispatch only uncached fits in parallel, then load all.
    todo = [t for t in tasks if not os.path.exists(_prior_cache(*t))]
    print(f"[prior] {len(tasks)} fits; {len(tasks) - len(todo)} cached, {len(todo)} to run (n_jobs={n_jobs}).")
    if todo:
        Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_prior_one)(i, rep, warmup, samples)
            for i, rep in tqdm(todo, desc="Prior sensitivity", unit="fit"))
    # per config: lists over reps of posterior MEAN and posterior SD of beta_L, beta_S, sigma_u
    per_cfg = {i: {k: [] for k in ("L", "S", "su", "Lsd", "Ssd")} for i in range(n_cfg)}
    for (i, rep) in tasks:
        r = json.load(open(_prior_cache(i, rep)))
        for k in ("L", "S", "su", "Lsd", "Ssd"):
            per_cfg[i][k].append(r[k])

    # baseline (config 0) posterior SD = scale for the prior-sensitivity index
    baseL = float(np.mean(per_cfg[0]["L"])); baseS = float(np.mean(per_cfg[0]["S"]))
    sdL = max(float(np.mean(per_cfg[0]["Lsd"])), 1e-6)
    sdS = max(float(np.mean(per_cfg[0]["Ssd"])), 1e-6)

    rows = []
    for i in range(n_cfg):
        L = _ms(per_cfg[i]["L"]); S = _ms(per_cfg[i]["S"]); su = _ms(per_cfg[i]["su"])
        # prior-sensitivity index: posterior shift relative to baseline posterior SD
        sidx = max(abs(L[0] - baseL) / sdL, abs(S[0] - baseS) / sdS)
        rows.append(dict(label=SENS_PRIOR_CONFIGS[i][0], group=SENS_PRIOR_CONFIGS[i][1]["group"],
                         L=L, S=S, su=su, sidx=float(sidx)))
    # transparency CSV
    pd.DataFrame([dict(config=r["label"], group=r["group"], beta_L_mean=r["L"][0], beta_L_sd=r["L"][1],
                       beta_S_mean=r["S"][0], beta_S_sd=r["S"][1], sigma_u_mean=r["su"][0],
                       sensitivity_index=r["sidx"]) for r in rows]).to_csv(
        os.path.join(config.OUTPUT_DIR_TABLES, "prior_sensitivity.csv"), index=False)
    _write_prior_table(rows, true_L, true_S, true_su, reps)


def _write_knot_table(rows, true_L, true_S, reps):
    body = ""
    for r in rows:
        body += (f"\\multirow{{2}}{{*}}{{{r['J']}}} & sCFR & {_fmt_ms(r['scL'])} & {_fmt_ms(r['scS'])} "
                 f"& \\multirow{{2}}{{*}}{{{r['LZB']:.3f}}} & \\multirow{{2}}{{*}}{{{r['lmin']:.2e}}} \\\\\n"
                 f" & fsCFR & {_fmt_ms(r['fsL'])} & {_fmt_ms(r['fsS'])} & & \\\\\n\\addlinespace\n")
    tex = (
        r"\begin{table}[htbp]" "\n\\centering\n"
        r"\caption{Knot sensitivity and leakage (scenario " + SENS_SCENARIO_ID +
        r"; truth: level and slope $=0.60$). Level $\beta^{(L)}_{\text{abs},1}$ and slope $\beta^{(S)}_{\text{abs},1}$ estimates (sCFR posterior mean, fsCFR point; mean\,$\pm$\,SD) across knot counts $J$. The design-only $\mathcal{L}_{Z\mid B}$ and $\lambda_{\min}(\bm{Z}^\top\bm{M}_B\bm{Z})$ gauge baseline/intervention separation, which is weaker for larger $\mathcal{L}_{Z\mid B}$ or smaller $\lambda_{\min}$.}" "\n"
        r"\label{tab:knot_sensitivity}" "\n"
        r"\resizebox{\textwidth}{!}{%" "\n"
        r"\begin{tabular}{cccccc}" "\n\\toprule\n"
        r"$J$ & Method & $\beta^{(L)}_{\text{abs},1}$ & $\beta^{(S)}_{\text{abs},1}$ & $\mathcal{L}_{Z\mid B}$ & $\lambda_{\min}(\bm{Z}^\top\bm{M}_B\bm{Z})$ \\" "\n\\midrule\n"
        + body +
        r"\bottomrule" "\n\\end{tabular}\n}\n\\end{table}\n"
    )
    out = os.path.join(config.OUTPUT_DIR_TABLES, "knot_sensitivity.tex")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print(f"[knot ] wrote {out}")


def _write_prior_table(rows, true_L, true_S, true_su, reps):
    gdisp = {"|beta| prior": r"the intervention-magnitude prior $|\bm{\beta}|$",
             "sigma_u prior": r"the random-effect scale prior $\sigma_u$",
             "tau_alpha prior": r"the spline-precision prior $\tau_\alpha$",
             "alpha prior": r"the spline-coefficient prior $\bm{\alpha}$"}
    body = f"True & {true_L:.3f} & {true_S:.3f} & {true_su:.3f} & -- \\\\\n\\midrule\n"
    last_group = None
    for r in rows:
        if r["group"] != "Baseline" and r["group"] != last_group:
            body += f"\\addlinespace\n\\multicolumn{{5}}{{l}}{{\\textit{{Varying {gdisp.get(r['group'], r['group'])}}}}}\\\\\n"
            last_group = r["group"]
        sidx = "--" if r["group"] == "Baseline" else f"{r['sidx']:.2f}"
        body += (f"\\quad {r['label']} & {_fmt_ms(r['L'])} & {_fmt_ms(r['S'])} "
                 f"& {_fmt_ms(r['su'])} & {sidx} \\\\\n")
    tex = (
        r"\begin{table}[htbp]" "\n\\centering\n"
        r"\caption{Prior sensitivity (scenario " + SENS_SCENARIO_ID +
        r"; truth: level $0.60$, slope $0.60$, $\sigma_u=0.10$). sCFR posterior-mean level $\beta^{(L)}_{\text{abs},1}$, slope $\beta^{(S)}_{\text{abs},1}$, and $\sigma_u$ (mean\,$\pm$\,SD) under priors varied one at a time, in scale and family, around the baseline ($\mathcal{LN}(\log0.5,0.5)$ on $|\bm{\beta}|$, $C^+(0,0.1)$ on $\sigma_u$, $\Gamma(0.01,0.01)$ on $\tau_\alpha$, $\mathcal{N}(0,5^2)$ on $\bm{\alpha}$). The index $S=\max_{j\in\{L,S\}}|\widehat\beta_j-\widehat\beta_j^{\text{base}}|/\widehat{\mathrm{sd}}(\beta_j^{\text{base}})$ is the largest shift from the baseline estimate $\widehat\beta_j^{\text{base}}$ in baseline posterior SDs; $S<1$ is within posterior uncertainty.}" "\n"
        r"\label{tab:prior_sensitivity}" "\n"
        r"\resizebox{\textwidth}{!}{%" "\n"
        r"\begin{tabular}{p{0.34\textwidth}cccc}" "\n\\toprule\n"
        r"Prior configuration & $\beta^{(L)}_{\text{abs},1}$ & $\beta^{(S)}_{\text{abs},1}$ & $\sigma_u$ & $S$ \\" "\n\\midrule\n"
        + body +
        r"\bottomrule" "\n\\end{tabular}\n}\n\\end{table}\n"
    )
    out = os.path.join(config.OUTPUT_DIR_TABLES, "prior_sensitivity.tex")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print(f"[prior] wrote {out}")


# =============================================================================
# Misspecified-scenario experiment (R2-6)
# =============================================================================
MISSPEC_CACHE_DIR = os.path.join(config.OUTPUT_DIR_TABLES, "misspec_cache")
MISSPEC_SEED = 909
MISSPEC_NB_SIZE = 100.0   # negative-binomial size: moderate overdispersion (variance ~1.9x mean)
# (display label, kind, departure described in the caption)
MISSPEC_KINDS = [
    ("Well-specified", "well", "i.i.d.\\ effects, hinge intervention, Poisson, correct delay"),
    ("AR(1) day-level effects", "ar1", "$u_t=0.7\\,u_{t-1}+\\varepsilon_t$ (serial correlation)"),
    ("Saturating intervention", "saturating", "gradual exponential onset, not level/slope hinge"),
    ("Negative-binomial deaths", "negbin", "overdispersed deaths, not Poisson"),
    ("Misspecified delay", "delay", "data delay $1.5\\times$ the fitted onset-to-death mean"),
]


def _scfr_traj_logit(samples, Bm, Z_step, Z_hinge, signs):
    """Reconstruct per-draw factual and counterfactual logit-CFR from sCFR samples."""
    A = np.asarray(samples["alpha"])                       # S x J
    main = A @ Bm.T                                        # S x T
    dr = np.asarray(samples["delta_raw"]); sd = np.asarray(samples["sigma_delta"])[:, None]
    u = sd * (dr - dr.mean(axis=1, keepdims=True))         # S x T, centered
    signs = np.asarray(signs, float)
    ba = np.asarray(samples["beta_abs"]); bs = np.asarray(samples["beta_slope_abs"])
    interv = (ba * signs) @ Z_step.T + (bs * signs) @ Z_hinge.T
    return main + u + interv, main + u                     # logit_F, logit_CF


def _logit_mae(r_hat, true_logit):
    rc = np.clip(np.asarray(r_hat, float), 1e-6, 1 - 1e-6)
    return float(np.mean(np.abs(np.log(rc / (1 - rc)) - true_logit)))


def _logit_mae_p(true_p, est_p):
    """Logit-scale MAE between two probability vectors (matches evaluation.calculate_logit_mae)."""
    eps = 1e-6
    tc = np.clip(np.asarray(true_p, float), eps, 1 - eps)
    ec = np.clip(np.asarray(est_p, float), eps, 1 - eps)
    return float(np.mean(np.abs(np.log(ec / (1 - ec)) - np.log(tc / (1 - tc)))))


def _misspec_one(kind, rep, kind_idx_val, warmup, samples):
    """One misspecified-scenario fit (cache-or-compute), safe in a joblib worker."""
    cache = _misspec_cache(kind, rep)
    if os.path.exists(cache):
        return json.load(open(cache))
    sim = data_generation.simulate_misspecified_data(MISSPEC_SEED + 1000 * rep, kind,
                                                     nb_size=MISSPEC_NB_SIZE)
    Bm = sim["Bm_true"]
    Z_step, Z_hinge = sim["Z_input_true"], sim["Z_hinge_true"]
    signs = np.asarray(sim["beta_signs_true"], float)
    # Mirror the main simulation evaluation exactly (evaluation.py): noisy factual and
    # counterfactual CFR targets on the analysis window [0:T_an), estimates with the
    # day-level effect retained, logit MAE via clip-then-logit.
    T_an = config.T_ANALYSIS_LENGTH
    true_rF = np.asarray(sim["true_r_0_t"])[:T_an]
    true_rCF = np.asarray(sim["true_rcf_0_t"])[:T_an]
    data, init = _sens_data_for_sampler(sim, Bm)
    s = _sens_fit_scfr(data, init, jax.random.PRNGKey(MISSPEC_SEED + 7 * rep + 101 * kind_idx_val),
                       warmup, samples)
    lF, lCF = _scfr_traj_logit(s, Bm, Z_step, Z_hinge, signs)   # include u
    rF_draws = 1.0 / (1.0 + np.exp(-lF)); rCF_draws = 1.0 / (1.0 + np.exp(-lCF))
    scF = _logit_mae_p(true_rF, rF_draws.mean(0)[:T_an])
    scCF = _logit_mae_p(true_rCF, rCF_draws.mean(0)[:T_an])
    lo, hi = np.percentile(rF_draws[:, :T_an], [2.5, 97.5], axis=0)
    scCov = float(np.mean((true_rF >= lo) & (true_rF <= hi)))
    fs = fsCFR_model_wrapper(d_t=sim["d_t"], c_t=sim["c_t"], f_s=sim["f_s_true"], Bm=Bm,
                             intervention_times_abs=[int(sim["true_intervention_times_0_abs"][0])],
                             intervention_signs=list(signs.astype(int)), verbose=False)
    fsF = _logit_mae_p(true_rF, np.asarray(fs["fsCFR_factual_mean"])[:T_an])
    fsCF = _logit_mae_p(true_rCF, np.asarray(fs["fsCFR_counterfactual_mean"])[:T_an])
    r = dict(kind=kind, rep=rep, scF=scF, scCF=scCF, scCov=scCov, fsF=fsF, fsCF=fsCF)
    json.dump(r, open(cache, "w"))
    return r


def run_misspecification(demo=False, warmup=None, samples=None, reps=None, n_jobs=None):
    """R2-6: generate data under misspecified DGPs (departing one assumption at a
    time) and compare sCFR and fsCFR recovery of the factual and counterfactual CFR
    against the known truth, averaged over replicate datasets. Fits run in parallel
    across the DGP x replicate grid (joblib)."""
    os.makedirs(MISSPEC_CACHE_DIR, exist_ok=True)
    warmup = warmup if warmup is not None else config.NUM_WARMUP
    samples = samples if samples is not None else config.NUM_SAMPLES
    reps = reps if reps is not None else (10 if demo else config.AUX_NUM_REPLICATIONS)
    n_jobs = n_jobs if n_jobs is not None else config.NUM_CORES_TO_USE

    kinds = [k for _, k, _ in MISSPEC_KINDS]
    kind_idx = {k: i for i, k in enumerate(kinds)}
    tasks = [(k, rep) for k in kinds for rep in range(reps)]
    # Break-point pre-filter: dispatch only uncached fits in parallel, then load all.
    todo = [t for t in tasks if not os.path.exists(_misspec_cache(*t))]
    print(f"[misspec] {len(tasks)} fits; {len(tasks) - len(todo)} cached, {len(todo)} to run (n_jobs={n_jobs}).")
    if todo:
        Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_misspec_one)(k, rep, kind_idx[k], warmup, samples)
            for k, rep in tqdm(todo, desc="Misspecification", unit="fit"))
    acc = {k: {m: [] for m in ("scF", "scCF", "scCov", "fsF", "fsCF")} for k in kinds}
    for (k, rep) in tasks:
        r = json.load(open(_misspec_cache(k, rep)))
        for m in ("scF", "scCF", "scCov", "fsF", "fsCF"):
            acc[k][m].append(r[m])

    rows = []
    for label, kind, _ in MISSPEC_KINDS:
        a = acc[kind]
        rows.append(dict(label=label, scF=_ms(a["scF"]), scCF=_ms(a["scCF"]),
                         scCov=_ms(a["scCov"]), fsF=_ms(a["fsF"]), fsCF=_ms(a["fsCF"])))
    pd.DataFrame([dict(scenario=r["label"], scfr_factual_mae=r["scF"][0],
                       scfr_cf_mae=r["scCF"][0], scfr_cov95=r["scCov"][0],
                       fscfr_factual_mae=r["fsF"][0], fscfr_cf_mae=r["fsCF"][0]) for r in rows]).to_csv(
        os.path.join(config.OUTPUT_DIR_TABLES, "misspecification.csv"), index=False)
    _write_misspec_table(rows, reps)


def _write_misspec_table(rows, reps):
    body = ""
    for r in rows:
        body += (f"\\multirow{{2}}{{*}}{{{r['label']}}} & sCFR & {_fmt_ms(r['scF'])} & {_fmt_ms(r['scCF'])} "
                 f"& {_fmt_ms(r['scCov'])} \\\\\n"
                 f" & fsCFR & {_fmt_ms(r['fsF'])} & {_fmt_ms(r['fsCF'])} & -- \\\\\n\\addlinespace\n")
    tex = (
        r"\begin{table}[htbp]" "\n\\centering\n"
        r"\caption{Misspecified scenarios. Each row draws data from a process that departs"
        r" from the sCFR estimator along one axis (relative to a single-intervention"
        r" sinusoidal-baseline reference), while both methods fit their standard form."
        r" Columns: mean absolute error of the factual and counterfactual CFR on the logit"
        r" scale over the analysis window (mean\,$\pm$\,SD), and the empirical coverage of the sCFR"
        r" $95\%$ credible interval for the factual CFR. Lower MAE is better; coverage near"
        r" $0.95$ is calibrated.}" "\n"
        r"\label{tab:misspecification}" "\n"
        r"\resizebox{\textwidth}{!}{%" "\n"
        r"\begin{tabular}{llccc}" "\n\\toprule\n"
        r"Data-generating process & Method & Factual MAE & Counterfactual MAE & sCFR 95\% coverage \\" "\n\\midrule\n"
        + body +
        r"\bottomrule" "\n\\end{tabular}\n}\n\\end{table}\n"
    )
    out = os.path.join(config.OUTPUT_DIR_TABLES, "misspecification.tex")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print(f"[misspec] wrote {out}")


# =============================================================================
# Core Simulation Logic
# =============================================================================

def run_single_simulation_task(scenario, run_idx, seed):
    """Execute a single simulation task."""
    scenario_id = scenario['id']
    
    metrics_file = os.path.join(config.OUTPUT_DIR_RUN_METRICS_JSON, f"{scenario_id}_run_{run_idx}_metrics.json")
    posterior_file = os.path.join(config.OUTPUT_DIR_POSTERIOR_SAMPLES, f"{scenario_id}_run_{run_idx}_posterior.npz")
    benchmark_file = os.path.join(config.OUTPUT_DIR_BENCHMARK_RESULTS, f"{scenario_id}_run_{run_idx}_benchmarks.npz")
    
    # Check required files for skipping (posterior_file only if saving is enabled)
    required_files = [metrics_file, benchmark_file]
    if config.SAVE_RAW_POSTERIOR_SAMPLES:
        required_files.append(posterior_file)
    
    if all(os.path.exists(f) for f in required_files):
        try:
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            if metrics_data.get('error') in [None, "None"]:
                return None
        except (json.JSONDecodeError, KeyError, IOError):
            pass
    
    start_time = time.time()
    
    try:
        sim_data = data_generation.simulate_scenario_data(scenario, seed)
        benchmark_results = methods.run_all_benchmarks(sim_data)
        
        rng_key = jax.random.PRNGKey(seed)
        posterior_samples, _ = methods.fit_proposed_model(sim_data, rng_key)
        
        posterior_summary = generate_posterior_summary(posterior_samples, scenario_id, run_idx)
        
        evaluator = ev.get_default_evaluator()
        benchmarks_r_t = {
            "cCFR_model": benchmark_results.get("cCFR_model", np.array([])),
            "aCFR_model": benchmark_results.get("aCFR_model", np.array([])),
        }
        its_results = {k: v for k, v in benchmark_results.items() if k.startswith("fsCFR_")}
        evaluation_result = evaluator.evaluate_single_run(
            sim_data=sim_data,
            posterior_scfr=posterior_samples,
            benchmarks_r_t=benchmarks_r_t,
            benchmark_cis={},
            its_results=its_results
        )

        run_metrics = {
            "scenario_id": scenario_id,
            "mc_run": run_idx,
            "seed": seed,
            "true_beta_abs": scenario["true_beta_abs_0"],
            "error": None,
        }
        for model_name, model_result in evaluation_result.model_results.items():
            for metric_name, metric_value in model_result.metrics.items():
                run_metrics[f"{metric_name}_{model_name}"] = metric_value

        if "fsCFR_beta_abs_est" in benchmark_results:
            run_metrics["fsCFR_beta_abs_est"] = benchmark_results["fsCFR_beta_abs_est"]
        
        elapsed = time.time() - start_time
        save_results(scenario_id, run_idx, posterior_samples, benchmark_results, run_metrics, elapsed)
        
        return f"{scenario_id}-{run_idx}: Success"
        
    except Exception as e:
        error_msg = str(e)
        error_tb = traceback.format_exc()
        
        print(f"\n[Error] Scenario {scenario_id} Run {run_idx} failed: {error_msg}")
        traceback.print_exc()
        
        context = {
            "seed": seed,
            "scenario_config": {
                "cfr_type_code": scenario.get("cfr_type_code", "unknown"),
                "intervention_type_code": scenario.get("intervention_type_code", "unknown"),
                "num_interventions": scenario.get("num_interventions_K_true", "unknown")
            },
            "elapsed_time_seconds": time.time() - start_time,
        }
        
        log_error_to_file(scenario_id, run_idx, "simulation", error_msg, error_tb, context)
        
        error_metrics = {
            "scenario_id": scenario_id,
            "mc_run": run_idx,
            "seed": seed,
            "error": error_msg,
            "elapsed_time_seconds": context["elapsed_time_seconds"],
        }
        error_metrics_file = os.path.join(config.OUTPUT_DIR_RUN_METRICS_JSON, f"{scenario_id}_run_{run_idx}_metrics.json")
        try:
            with open(error_metrics_file, 'w') as f:
                json.dump(error_metrics, f, indent=4)
        except Exception as save_err:
            print(f"[Warning] Failed to save error metrics: {save_err}")
        
        return f"{scenario_id}-{run_idx}: Failed"


# =============================================================================
# Main Program
# =============================================================================

def _main_task_complete(scenario_id, run_idx):
    """A main run is complete (cacheable break-point) when its metrics and benchmark
    files exist with no recorded error. Mirrors the skip logic in
    run_single_simulation_task so the task list can be pre-filtered."""
    metrics_file = os.path.join(config.OUTPUT_DIR_RUN_METRICS_JSON, f"{scenario_id}_run_{run_idx}_metrics.json")
    benchmark_file = os.path.join(config.OUTPUT_DIR_BENCHMARK_RESULTS, f"{scenario_id}_run_{run_idx}_benchmarks.npz")
    required = [metrics_file, benchmark_file]
    if config.SAVE_RAW_POSTERIOR_SAMPLES:
        required.append(os.path.join(config.OUTPUT_DIR_POSTERIOR_SAMPLES, f"{scenario_id}_run_{run_idx}_posterior.npz"))
    if not all(os.path.exists(f) for f in required):
        return False
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f).get('error') in (None, "None")
    except (json.JSONDecodeError, KeyError, IOError):
        return False


def run_refresh_docs():
    """Sync docs/figs_tables with the latest results by invoking refresh_docs.main().
    Wrapped so a refresh failure does not mask a successful simulation run."""
    print("\n" + "=" * 60)
    print("REFRESHING docs/figs_tables FROM LATEST RESULTS...")
    print("=" * 60)
    try:
        import refresh_docs
        refresh_docs.main()
    except Exception as e:
        print(f"[Warning] refresh_docs failed: {e}")
        traceback.print_exc()


def run_auxiliary_experiments(demo=False, n_jobs=None):
    """Run the four auxiliary experiments at the same scale as the main grid:
    knot sensitivity (Table 3), prior sensitivity (Table 4), misspecification
    (Table 5), and runtime scaling (Figure 5). Each is independently checkpointed
    and wrapped so that one failure does not abort the others. Full scale uses
    config.AUX_NUM_REPLICATIONS replicates for knot/prior/misspec and 100 for
    runtime; demo uses 10 for each. All four are parallelized over n_jobs; the
    runtime timing pins each fit to a dedicated core so parallelism does not
    corrupt the wall-clock measurement."""
    steps = [
        ("knot sensitivity (Table 3)", lambda: run_knot_sensitivity(demo=demo, n_jobs=n_jobs)),
        ("prior sensitivity (Table 4)", lambda: run_prior_sensitivity(demo=demo, n_jobs=n_jobs)),
        ("misspecification (Table 5)", lambda: run_misspecification(demo=demo, n_jobs=n_jobs)),
        ("runtime scaling (Figure 5)", lambda: run_runtime_experiment(demo=demo, n_jobs=n_jobs)),
    ]
    for name, fn in steps:
        print("\n" + "=" * 60)
        print(f"AUXILIARY EXPERIMENT: {name}")
        print("=" * 60)
        try:
            fn()
        except Exception as e:
            print(f"[Warning] Auxiliary experiment '{name}' failed: {e}")
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="sCFR Simulation Runner and Analyzer")
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--simulate', action='store_true', help="Run simulation")
    mode_group.add_argument('--analyze', action='store_true', help="Analyze existing results")
    mode_group.add_argument('--runtime', action='store_true', help="Run the Section 3.5 computational (timing/scalability) experiment")
    mode_group.add_argument('--knot', action='store_true', help="Run the knot-sensitivity experiment (R2-2) on a simulation scenario")
    mode_group.add_argument('--prior', action='store_true', help="Run the prior-sensitivity experiment (R2-3) on a simulation scenario")
    mode_group.add_argument('--misspec', action='store_true', help="Run the misspecified-scenario experiment (R2-6)")

    parser.add_argument('--demo', action='store_true', help="Demo mode (5 runs per scenario)")
    parser.add_argument('--full', action='store_true', help="Full mode (all configured runs)")
    parser.add_argument('--reset', action='store_true', help="Clear all outputs and restart")
    parser.add_argument('--jobs', type=int, default=config.NUM_CORES_TO_USE, help=f"Number of parallel jobs (default: {config.NUM_CORES_TO_USE})")
    parser.add_argument('--main-only', action='store_true', help="With --simulate, run only the main 12-scenario grid and skip the auxiliary experiments (knot/prior/misspec/runtime)")
    parser.add_argument('--no-refresh', action='store_true', help="With --simulate, skip the automatic refresh_docs.py sync of docs/figs_tables at the end")
    
    args = parser.parse_args()
    
    if args.demo:
        n_runs = 5
        n_jobs = 5
        print(">>> Run mode: DEMO (5 repetitions per scenario, 5 cores)")
    else:
        n_runs = config.NUM_MONTE_CARLO_RUNS
        n_jobs = args.jobs
        print(f">>> Run mode: FULL ({n_runs} repetitions per scenario, {n_jobs} cores)")
    
    if args.reset:
        confirm = input(f"!!! WARNING !!! You are about to delete all data under {config.OUTPUT_DIR_BASE}.\nPlease enter 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            clear_directories()
        else:
            print("Operation cancelled.")
            return
    else:
        ensure_directories()
    
    if args.runtime:
        # Section 3.5 timing experiment. Demo uses the full-scale grid and the
        # production NUTS length, differing from full only in the replicate count (10).
        run_runtime_experiment(demo=args.demo, n_jobs=args.jobs)
        return

    # The added simulation experiments (knot R2-2, prior R2-3, misspec R2-6) use the
    # SAME NUTS length and settings in demo and full; demo differs only by using
    # fewer replicate datasets (10). All other methods/parameters are identical.
    if args.knot:
        run_knot_sensitivity(demo=args.demo)
        return

    if args.prior:
        run_prior_sensitivity(demo=args.demo)
        return

    if args.misspec:
        run_misspecification(demo=args.demo)
        return

    if args.analyze:
        run_analysis()
        try:
            aggregate_and_plot_runtime(demo=args.demo)
        except Exception as e:
            print(f"[Warning] Runtime figure regeneration skipped: {e}")
    else:
        base_seed = config.GLOBAL_BASE_SEED

        print("\n[System] Building task list...")
        all_tasks = []
        for scenario in config.SCENARIOS:
            for i in range(n_runs):
                scen_idx = int(scenario['id'][1:])
                current_seed = base_seed + scen_idx * 10000 + i
                all_tasks.append((scenario, i, current_seed))

        # Break-point / cache support: skip main runs already completed (metrics and
        # benchmark files present with no recorded error). Re-invoking
        # `--simulate --full` therefore resumes and runs only the missing runs.
        tasks = [t for t in all_tasks if not _main_task_complete(t[0]['id'], t[1])]
        precached = len(all_tasks) - len(tasks)
        print(f"[System] {len(all_tasks)} total runs; {precached} already cached, "
              f"{len(tasks)} to run (Scenarios: {len(config.SCENARIOS)} x Repetitions: {n_runs}).")

        success_count = 0
        failed_count = 0
        if tasks:
            print(f"[System] Starting parallel simulation (Cores: {n_jobs})...")
            results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(run_single_simulation_task)(scen, idx, seed)
                for scen, idx, seed in tqdm(tasks, desc="Simulation Progress", unit="run")
            )
            failed_count = sum(1 for r in results if r and "Failed" in r)
            success_count = sum(1 for r in results if r and "Success" in r)
        else:
            print("[System] All main runs already cached; nothing new to run.")

        total_complete = precached + success_count
        print("\n" + "=" * 30)
        print("       SIMULATION SUMMARY       ")
        print("=" * 30)
        print(f"Total runs  : {len(all_tasks)}")
        print(f"Cached      : {precached}")
        print(f"New success : {success_count}")
        print(f"Failed      : {failed_count}")
        print(f"Completed   : {total_complete}")
        print("=" * 30)
        print(f"Results saved in: {config.OUTPUT_DIR_BASE}")

        # Always (re)build the analysis and plots from all completed results, even
        # when nothing new ran, so the figures and the beta-MAE table reflect the cache.
        if total_complete > 0:
            print("\n" + "=" * 60)
            print("AUTOMATICALLY STARTING ANALYSIS...")
            print("=" * 60)
            try:
                run_analysis()
            except Exception as e:
                print(f"\n[Warning] Analysis failed: {e}")
                print("You can run 'python simulation.py --analyze' manually.")
        else:
            print("\n[Warning] No completed simulations. Skipping analysis.")

        # After the main 12-scenario grid, run the four auxiliary experiments at the
        # same scale (knot Table 3, prior Table 4, misspec Table 5, runtime Figure 5),
        # so that `--simulate --full` produces every paper-consumed result in one go.
        # Each is independently checkpointed, so this resumes and only fills gaps.
        # Use --main-only to skip them.
        if not args.main_only:
            print("\n" + "=" * 60)
            print("RUNNING AUXILIARY EXPERIMENTS (knot, prior, misspec, runtime)...")
            print("=" * 60)
            run_auxiliary_experiments(demo=args.demo, n_jobs=n_jobs)

        # Finally, sync docs/figs_tables with the latest results (figures plus the
        # data-driven tables), unless --no-refresh is given.
        if not args.no_refresh:
            run_refresh_docs()


if __name__ == "__main__":
    main()
