"""
Simulation runner for sCFR study.

This script provides:
- Monte Carlo simulation with checkpoint support
- Automatic analysis and visualization
- Parallel execution using joblib

Usage:
    python simulation.py --simulate --demo    # Quick test (5 runs)
    python simulation.py --simulate --full    # Full simulation
    python simulation.py --analyze            # Analysis only
    python simulation.py --simulate --reset   # Clear and restart
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
import jax

import config
import methods
import data_generation
import evaluation as ev


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
    
    post_file = os.path.join(config.OUTPUT_DIR_POSTERIOR_SAMPLES, f"{scenario_id}_run_{run_idx}_posterior.npz")
    np.savez_compressed(post_file, **posterior_samples)
    
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
        results_df_valid[col] = results_df_valid[col].astype('Int64')
    
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
        ("Generating summary boxplots...", ev.plot_metric_summary_boxplots, "plot_metric_summary_boxplots"),
        ("Generating combined metrics summary...", ev.plot_combined_metrics_summary, "plot_combined_metrics_summary"),
    ]
    
    for msg, func, step_name in plot_functions:
        try:
            print(msg)
            if "aggregated" in step_name:
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
# Core Simulation Logic
# =============================================================================

def run_single_simulation_task(scenario, run_idx, seed):
    """Execute a single simulation task."""
    scenario_id = scenario['id']
    
    metrics_file = os.path.join(config.OUTPUT_DIR_RUN_METRICS_JSON, f"{scenario_id}_run_{run_idx}_metrics.json")
    posterior_file = os.path.join(config.OUTPUT_DIR_POSTERIOR_SAMPLES, f"{scenario_id}_run_{run_idx}_posterior.npz")
    benchmark_file = os.path.join(config.OUTPUT_DIR_BENCHMARK_RESULTS, f"{scenario_id}_run_{run_idx}_benchmarks.npz")
    
    if all(os.path.exists(f) for f in [metrics_file, posterior_file, benchmark_file]):
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

def main():
    parser = argparse.ArgumentParser(description="sCFR Simulation Runner and Analyzer")
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--simulate', action='store_true', help="Run simulation")
    mode_group.add_argument('--analyze', action='store_true', help="Analyze existing results")
    
    parser.add_argument('--demo', action='store_true', help="Demo mode (5 runs per scenario)")
    parser.add_argument('--full', action='store_true', help="Full mode (all configured runs)")
    parser.add_argument('--reset', action='store_true', help="Clear all outputs and restart")
    parser.add_argument('--jobs', type=int, default=config.NUM_CORES_TO_USE, help=f"Number of parallel jobs (default: {config.NUM_CORES_TO_USE})")
    
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
    
    if args.analyze:
        run_analysis()
    else:
        tasks = []
        base_seed = config.GLOBAL_BASE_SEED
        
        print("\n[System] Building task list...")
        for scenario in config.SCENARIOS:
            for i in range(n_runs):
                scen_idx = int(scenario['id'][1:])
                current_seed = base_seed + scen_idx * 10000 + i
                tasks.append((scenario, i, current_seed))
        
        total_tasks = len(tasks)
        print(f"[System] Total tasks: {total_tasks} (Scenarios: {len(config.SCENARIOS)} x Repetitions: {n_runs})")
        
        print(f"[System] Starting parallel simulation (Cores: {n_jobs})...")
        
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(run_single_simulation_task)(scen, idx, seed)
            for scen, idx, seed in tqdm(tasks, desc="Simulation Progress", unit="run")
        )
        
        skipped_count = results.count(None)
        failed_count = sum(1 for r in results if r and "Failed" in r)
        success_count = sum(1 for r in results if r and "Success" in r)
        
        print("\n" + "=" * 30)
        print("       SIMULATION SUMMARY       ")
        print("=" * 30)
        print(f"Total Tasks : {total_tasks}")
        print(f"Success     : {success_count}")
        print(f"Skipped     : {skipped_count} (Checkpoints found)")
        print(f"Failed      : {failed_count}")
        print("=" * 30)
        print(f"Results saved in: {config.OUTPUT_DIR_BASE}")
        
        if success_count > 0:
            print("\n" + "=" * 60)
            print("AUTOMATICALLY STARTING ANALYSIS...")
            print("=" * 60)
            try:
                run_analysis()
            except Exception as e:
                print(f"\n[Warning] Analysis failed: {e}")
                print("You can run 'python simulation.py --analyze' manually.")
        else:
            print("\n[Warning] No successful simulations. Skipping analysis.")


if __name__ == "__main__":
    main()
