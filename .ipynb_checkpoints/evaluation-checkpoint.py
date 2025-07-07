import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
import config

def get_posterior_estimates(posterior_samples, key_name, percentiles=(2.5, 97.5)):
    """Helper to get posterior mean, lower, and upper credible interval bounds for a parameter."""
    if not posterior_samples or key_name not in posterior_samples:
        shape_ref = config.T_SERIES_LENGTH_SIM
        nan_array = np.full(shape_ref, np.nan) if key_name in ['p', 'p_cf'] else np.array([np.nan])
        return nan_array, nan_array, nan_array
    samples = posterior_samples[key_name]
    mean_est, lower_est, upper_est = np.mean(samples, axis=0), np.percentile(samples, percentiles[0], axis=0), np.percentile(samples, percentiles[1], axis=0)
    return mean_est, lower_est, upper_est

def calculate_mae_rt(true_r_t, estimated_r_t_mean):
    return np.mean(np.abs(estimated_r_t_mean - true_r_t))

def calculate_mciw_rt(r_t_lower, r_t_upper):
    return np.mean(r_t_upper - r_t_lower)

def calculate_mcic_rt(true_r_t, r_t_lower, r_t_upper):
    return np.mean((true_r_t >= r_t_lower) & (true_r_t <= r_t_upper))

def calculate_param_bias(true_param_val, est_param_mean):
    return est_param_mean - true_param_val

def calculate_param_cri_width(param_lower_ci, param_upper_ci):
    return param_upper_ci - param_lower_ci

def calculate_param_cri_coverage(true_param_val, param_lower_ci, param_upper_ci):
    return (true_param_val >= param_lower_ci) & (true_param_val <= param_upper_ci)

def collect_all_metrics(sim_data, posterior_scfr, benchmarks_r_t, benchmark_cis, its_results):
    """
    Collects all specified metrics for the constrained sCFR model and all benchmarks.
    """
    metrics = {"scenario_id": sim_data["scenario_id"]}
    T_analyze = config.T_ANALYSIS_LENGTH
    true_r_t_analysis, true_rcf_0_t_analysis = sim_data["true_r_0_t"][:T_analyze], sim_data["true_rcf_0_t"][:T_analyze]
    
    # --- sCFR Metrics ---
    if posterior_scfr:
        r_t_mean, r_t_lower, r_t_upper = get_posterior_estimates(posterior_scfr, "p")
        rcf_t_mean, rcf_t_lower, rcf_t_upper = get_posterior_estimates(posterior_scfr, "p_cf")
        
        metrics["mae_rt_sCFR"] = float(calculate_mae_rt(true_r_t_analysis, r_t_mean[:T_analyze]))
        metrics["mciw_rt_sCFR"] = float(calculate_mciw_rt(r_t_lower[:T_analyze], r_t_upper[:T_analyze]))
        metrics["mcic_rt_sCFR"] = float(calculate_mcic_rt(true_r_t_analysis, r_t_lower[:T_analyze], r_t_upper[:T_analyze]))
        
        metrics["mae_rcf_sCFR"] = float(calculate_mae_rt(true_rcf_0_t_analysis, rcf_t_mean[:T_analyze]))
        metrics["mciw_rcf_sCFR"] = float(calculate_mciw_rt(rcf_t_lower[:T_analyze], rcf_t_upper[:T_analyze]))
        metrics["mcic_rcf_sCFR"] = float(calculate_mcic_rt(true_rcf_0_t_analysis, rcf_t_lower[:T_analyze], rcf_t_upper[:T_analyze]))

    # --- Parameter Metrics for sCFR (gamma, lambda) ---
    num_interventions = sim_data["num_interventions_true_K"]
    if num_interventions > 0:
        true_beta_abs = sim_data["true_beta_abs_0"]
        true_lambdas = sim_data["true_lambda_0"]
        
        # --- sCFR Parameter Metrics (for beta_abs and lambda) ---
        if posterior_scfr:
            est_beta_abs_m, est_beta_abs_l, est_beta_abs_u = map(np.atleast_1d, get_posterior_estimates(posterior_scfr, "beta_abs"))
            est_lambda_m, est_lambda_l, est_lambda_u = map(np.atleast_1d, get_posterior_estimates(posterior_scfr, "lambda"))
            
            for k in range(num_interventions):
                if k < len(est_beta_abs_m):
                    metrics[f"bias_beta_abs_{k+1}_sCFR"] = float(calculate_param_bias(true_beta_abs[k], est_beta_abs_m[k]))
                    metrics[f"width_beta_abs_{k+1}_sCFR"] = float(calculate_param_cri_width(est_beta_abs_l[k], est_beta_abs_u[k]))
                    metrics[f"cover_beta_abs_{k+1}_sCFR"] = bool(calculate_param_cri_coverage(true_beta_abs[k], est_beta_abs_l[k], est_beta_abs_u[k]))
                if k < len(est_lambda_m):
                    metrics[f"bias_lambda_{k+1}_sCFR"] = float(calculate_param_bias(true_lambdas[k], est_lambda_m[k]))
                    metrics[f"width_lambda_{k+1}_sCFR"] = float(calculate_param_cri_width(est_lambda_l[k], est_lambda_u[k]))
                    metrics[f"cover_lambda_{k+1}_sCFR"] = bool(calculate_param_cri_coverage(true_lambdas[k], est_lambda_l[k], est_lambda_u[k]))
        
    # --- Benchmark Metrics (cCFR, aCFR, ITS) ---
    if benchmarks_r_t and benchmark_cis:
        for bench_name, r_t_est_bench in benchmarks_r_t.items():
            metrics[f"mae_rt_{bench_name}"] = float(calculate_mae_rt(true_r_t_analysis, r_t_est_bench[:T_analyze]))
            lower_ci, upper_ci = benchmark_cis.get(f"{bench_name}_lower", []), benchmark_cis.get(f"{bench_name}_upper", [])
            metrics[f"mciw_rt_{bench_name}"] = float(calculate_mciw_rt(lower_ci[:T_analyze], upper_ci[:T_analyze]))
            metrics[f"mcic_rt_{bench_name}"] = float(calculate_mcic_rt(true_r_t_analysis, lower_ci[:T_analyze], upper_ci[:T_analyze]))

    if its_results:
        # ITS Curve Fit Metrics
        metrics["mae_rt_its"] = float(calculate_mae_rt(true_r_t_analysis, its_results["its_factual_mean"][:T_analyze]))
        metrics["mciw_rt_its"] = float(calculate_mciw_rt(its_results["its_factual_lower"][:T_analyze], its_results["its_factual_upper"][:T_analyze]))
        metrics[f"mcic_rt_its"] = float(calculate_mcic_rt(true_r_t_analysis, its_results["its_factual_lower"][:T_analyze], its_results["its_factual_upper"][:T_analyze]))
        metrics[f"mae_rcf_its"] = float(calculate_mae_rt(true_rcf_0_t_analysis, its_results["its_counterfactual_mean"][:T_analyze]))
        metrics[f"mciw_rcf_its"] = float(calculate_mciw_rt(its_results["its_counterfactual_lower"][:T_analyze], its_results["its_counterfactual_upper"][:T_analyze]))
        metrics[f"mcic_rcf_its"] = float(calculate_mcic_rt(true_rcf_0_t_analysis, its_results["its_counterfactual_lower"][:T_analyze], its_results["its_counterfactual_upper"][:T_analyze]))

        # --- NEW: ITS Parameter Metrics for beta_abs and lambda ---
        if num_interventions > 0:
            true_beta_abs = sim_data["true_beta_abs_0"]
            true_lambdas = sim_data["true_lambda_0"]
            for k in range(num_interventions):
                if k < len(its_results["its_beta_abs_est"]):
                    metrics[f"bias_beta_abs_{k+1}_its"] = float(calculate_param_bias(true_beta_abs[k], its_results["its_beta_abs_est"][k]))
                    metrics[f"width_beta_abs_{k+1}_its"] = float(calculate_param_cri_width(its_results["its_beta_abs_lower"][k], its_results["its_beta_abs_upper"][k]))
                    metrics[f"cover_beta_abs_{k+1}_its"] = bool(calculate_param_cri_coverage(true_beta_abs[k], its_results["its_beta_abs_lower"][k], its_results["its_beta_abs_upper"][k]))
                
                if k < len(its_results["its_lambda_est"]):
                    metrics[f"bias_lambda_{k+1}_its"] = float(calculate_param_bias(true_lambdas[k], its_results["its_lambda_est"][k]))
                    metrics[f"width_lambda_{k+1}_its"] = float(calculate_param_cri_width(its_results["its_lambda_lower"][k], its_results["its_lambda_upper"][k]))
                    metrics[f"cover_lambda_{k+1}_its"] = bool(calculate_param_cri_coverage(true_lambdas[k], its_results["its_lambda_lower"][k], its_results["its_lambda_upper"][k]))
            
    return metrics
