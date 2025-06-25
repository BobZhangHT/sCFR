# File: evaluation.py
# Description: Functions for calculating all specified evaluation metrics.
# This version includes explicit type casting to prevent serialization and aggregation errors.

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid

import config

def get_posterior_estimates(posterior_samples, key_name, percentiles=(2.5, 97.5)):
    """
    Helper to get posterior mean, lower, and upper credible interval bounds for a parameter.

    Args:
        posterior_samples (dict): The dictionary of MCMC samples from NumPyro.
        key_name (str): The name of the parameter/quantity to estimate (e.g., "p", "beta_abs").
        percentiles (tuple): The percentiles for the credible interval.

    Returns:
        tuple: (mean_estimate, lower_bound, upper_bound).
    """
    if key_name not in posterior_samples:
        # Determine expected shape for NaNs from a known variable like 'M'
        T_shape = posterior_samples.get("M", np.array([[np.nan]])).shape[1]
        nan_array = np.full(T_shape, np.nan) if key_name in ['p', 'p_cf', 'eps'] else np.array([np.nan])
        return nan_array, nan_array, nan_array

    samples = posterior_samples[key_name]
    mean_est = np.mean(samples, axis=0)
    lower_est = np.percentile(samples, percentiles[0], axis=0)
    upper_est = np.percentile(samples, percentiles[1], axis=0)
    return mean_est, lower_est, upper_est

def calculate_mae_rt(true_r_t, estimated_r_t_mean):
    """Calculates Mean Absolute Error for the time-series r_t."""
    return np.mean(np.abs(estimated_r_t_mean - true_r_t))

def calculate_mciw_rt(r_t_lower, r_t_upper):
    """Calculates Mean Credible Interval Width for the time-series r_t."""
    return np.mean(r_t_upper - r_t_lower)

def calculate_mcic_rt(true_r_t, r_t_lower, r_t_upper):
    """Calculates Mean Credible Interval Coverage for the time-series r_t."""
    coverage_flags = (true_r_t >= r_t_lower) & (true_r_t <= r_t_upper)
    return np.mean(coverage_flags)

def calculate_param_bias(true_param_val, est_param_mean):
    """Calculates Bias for a single parameter estimate."""
    return est_param_mean - true_param_val

def calculate_param_cri_width(param_lower_ci, param_upper_ci):
    """Calculates Credible Interval Width for a single parameter estimate."""
    return param_upper_ci - param_lower_ci

def calculate_param_cri_coverage(true_param_val, param_lower_ci, param_upper_ci):
    """Calculates Credible Interval Coverage for a single parameter estimate."""
    return (true_param_val >= param_lower_ci) & (true_param_val <= param_upper_ci)

def transform_alpha_post_hoc(posterior_samples, sim_data):
    """
    Transforms the posterior samples of the B-spline coefficients (alpha) to correct
    for the bias introduced by the orthogonalization procedure.

    Args:
        posterior_samples (dict): The raw dictionary of MCMC samples.
        sim_data (dict): The dictionary of simulated data, containing Bm and Z_input_true.

    Returns:
        dict: The posterior_samples dictionary, now updated with "alpha_transformed".
    """
    if "alpha" not in posterior_samples or "beta" not in posterior_samples:
        return posterior_samples # Nothing to transform

    alpha_star_samples = posterior_samples['alpha'] # These are samples of alpha*
    beta_samples = posterior_samples['beta']

    Bm = sim_data["Bm_true"]
    Z_orig = sim_data["Z_input_true"]
    
    if Z_orig.shape[1] == 0: # No interventions, no transformation needed
        posterior_samples['alpha_transformed'] = alpha_star_samples
        return posterior_samples

    try:
        # Pre-calculate the transformation matrix: (B.T @ B)^-1 @ B.T @ Z
        B_inv_B_T = np.linalg.pinv(Bm) @ Z_orig

        # Apply transformation for each posterior sample via vectorized operations
        # Correction term shape: (num_samples, K_alpha)
        alpha_correction = beta_samples @ B_inv_B_T.T
        
        # Transformed alpha samples
        alpha_transformed_samples = alpha_star_samples - alpha_correction
        
        posterior_samples['alpha_transformed'] = alpha_transformed_samples
    except np.linalg.LinAlgError:
        print("Warning: Could not perform post-hoc transformation. Using original alpha samples.")
        posterior_samples['alpha_transformed'] = alpha_star_samples

    return posterior_samples

def collect_all_metrics(sim_data, posterior_samples, benchmarks_r_t, benchmark_cis):
    """
    Collects all specified metrics for a single Monte Carlo run.

    Args:
        sim_data (dict): Dictionary of true data.
        posterior_samples (dict): Dictionary of raw MCMC samples.
        benchmarks_r_t (dict): Dictionary of r_t point estimates from benchmarks.
        benchmark_cis (dict): Dictionary of lower/upper CI bounds for benchmarks.

    Returns:
        dict: A dictionary of all calculated scalar metrics for this run.
    """
    # metrics = {"scenario_id": sim_data["scenario_id"]}
    # T_analyze = config.T_ANALYSIS_LENGTH
    
    posterior_samples = transform_alpha_post_hoc(posterior_samples, sim_data)
    metrics = {"scenario_id": sim_data["scenario_id"]}
    T_analyze = config.T_ANALYSIS_LENGTH

    true_r_0_t_analysis = sim_data["true_r_0_t"][:T_analyze]
    
    # --- sCFR (Proposed Model) Metrics ---
    r_t_mean_sCFR_full, r_t_lower_sCFR_full, r_t_upper_sCFR_full = get_posterior_estimates(posterior_samples, "p")
    metrics["mae_rt_sCFR"] = float(calculate_mae_rt(true_r_0_t_analysis, r_t_mean_sCFR_full[:T_analyze]))
    metrics["mciw_rt_sCFR"] = float(calculate_mciw_rt(r_t_lower_sCFR_full[:T_analyze], r_t_upper_sCFR_full[:T_analyze]))
    metrics["mcic_rt_sCFR"] = float(calculate_mcic_rt(true_r_0_t_analysis, r_t_lower_sCFR_full[:T_analyze], r_t_upper_sCFR_full[:T_analyze]))

    # --- Counterfactual r_t Metrics (sCFR) ---
    # rcf_t_mean_sCFR_full, rcf_t_lower_sCFR_full, rcf_t_upper_sCFR_full = get_posterior_estimates(posterior_samples, "p_cf")
    if 'alpha_transformed' in posterior_samples:
        M_transformed_samples = sim_data["Bm_true"] @ posterior_samples['alpha_transformed'].T
        eps_samples = posterior_samples.get('eps', np.zeros_like(M_transformed_samples.T)).T
        logit_p_cf_samples = M_transformed_samples + eps_samples
        p_cf_samples = sigmoid(logit_p_cf_samples).T # Shape: (num_samples, T_sim)
        
        # Now get estimates from these corrected samples
        rcf_t_mean_sCFR_full = np.mean(p_cf_samples, axis=0)
        rcf_t_lower_sCFR_full = np.percentile(p_cf_samples, 2.5, axis=0)
        rcf_t_upper_sCFR_full = np.percentile(p_cf_samples, 97.5, axis=0)
    else: # Fallback if transformation fails
        rcf_t_mean_sCFR_full, rcf_t_lower_sCFR_full, rcf_t_upper_sCFR_full = get_posterior_estimates(posterior_samples, "p_cf")
        
    true_rcf_0_t_analysis = sim_data["true_rcf_0_t"][:T_analyze]
    metrics["mae_rcf_sCFR"] = float(calculate_mae_rt(true_rcf_0_t_analysis, rcf_t_mean_sCFR_full[:T_analyze]))
    metrics["mciw_rcf_sCFR"] = float(calculate_mciw_rt(rcf_t_lower_sCFR_full[:T_analyze], rcf_t_upper_sCFR_full[:T_analyze]))
    metrics["mcic_rcf_sCFR"] = float(calculate_mcic_rt(true_rcf_0_t_analysis, rcf_t_lower_sCFR_full[:T_analyze], rcf_t_upper_sCFR_full[:T_analyze]))

    # --- Benchmark Metrics (Now with CIs) ---
    for bench_name, r_t_est_bench_full in benchmarks_r_t.items():
        metrics[f"mae_rt_{bench_name}"] = float(calculate_mae_rt(true_r_0_t_analysis, r_t_est_bench_full[:T_analyze]))
        
        lower_ci_key = f"{bench_name}_lower"
        upper_ci_key = f"{bench_name}_upper"
        
        if lower_ci_key in benchmark_cis and upper_ci_key in benchmark_cis:
            lower_bound_analysis = benchmark_cis[lower_ci_key][:T_analyze]
            upper_bound_analysis = benchmark_cis[upper_ci_key][:T_analyze]
            metrics[f"mciw_rt_{bench_name}"] = float(calculate_mciw_rt(lower_bound_analysis, upper_bound_analysis))
            metrics[f"mcic_rt_{bench_name}"] = float(calculate_mcic_rt(true_r_0_t_analysis, lower_bound_analysis, upper_bound_analysis))
        else:
            metrics[f"mciw_rt_{bench_name}"] = np.nan 
            metrics[f"mcic_rt_{bench_name}"] = np.nan
            
    # --- Metrics for gamma_k and lambda_k (Proposed Model) ---
    #true_betas_abs = sim_data["true_beta_abs_0"]
    #true_lambdas = sim_data["true_lambda_0"]
    num_interventions = sim_data["num_interventions_true_K"]

    if num_interventions > 0:
        # Add a small epsilon to avoid log(0) if a true beta_abs is ever 0
        true_gammas = np.log(sim_data["true_beta_abs_0"] + 1e-9) 
        true_lambdas = sim_data["true_lambda_0"]
        
        est_gamma_mean, est_gamma_lower, est_gamma_upper = get_posterior_estimates(posterior_samples, "gamma")
        est_lambda_mean, est_lambda_lower, est_lambda_upper = get_posterior_estimates(posterior_samples, "lambda")

        # Ensure estimates are always treated as 1D arrays for consistent indexing
        est_gamma_mean, est_gamma_lower, est_gamma_upper = map(np.atleast_1d, [est_gamma_mean, est_gamma_lower, est_gamma_upper])
        est_lambda_mean, est_lambda_lower, est_lambda_upper = map(np.atleast_1d, [est_lambda_mean, est_lambda_lower, est_lambda_upper])

        for k_idx in range(num_interventions):
            # Create metric keys for gamma, not beta_abs
            if k_idx < len(est_gamma_mean):
                metrics[f"bias_gamma_{k_idx+1}"] = float(calculate_param_bias(true_gammas[k_idx], est_gamma_mean[k_idx]))
                metrics[f"width_gamma_{k_idx+1}"] = float(calculate_param_cri_width(est_gamma_lower[k_idx], est_gamma_upper[k_idx]))
                metrics[f"cover_gamma_{k_idx+1}"] = bool(calculate_param_cri_coverage(true_gammas[k_idx], est_gamma_lower[k_idx], est_gamma_upper[k_idx]))
            
            if k_idx < len(est_lambda_mean):
                metrics[f"bias_lambda_{k_idx+1}"] = float(calculate_param_bias(true_lambdas[k_idx], est_lambda_mean[k_idx]))
                metrics[f"width_lambda_{k_idx+1}"] = float(calculate_param_cri_width(est_lambda_lower[k_idx], est_lambda_upper[k_idx]))
                metrics[f"cover_lambda_{k_idx+1}"] = bool(calculate_param_cri_coverage(true_lambdas[k_idx], est_lambda_lower[k_idx], est_lambda_upper[k_idx]))


    # --- Metrics for ITS Factual and Counterfactual r_t ---
    its_factual_analysis = its_results["its_factual_mean"][:T_analyze]
    its_cf_analysis = its_results["its_counterfactual_mean"][:T_analyze]
    
    metrics["mae_rt_its"] = float(calculate_mae_rt(true_r_0_t_analysis, its_factual_analysis))
    metrics["mae_rcf_its"] = float(calculate_mae_rt(true_rcf_0_t_analysis, its_cf_analysis))

    metrics["mciw_rt_its"] = float(calculate_mciw_rt(its_results["its_factual_lower"][:T_analyze], its_results["its_factual_upper"][:T_analyze]))
    metrics["mciw_rcf_its"] = float(calculate_mciw_rt(its_results["its_counterfactual_lower"][:T_analyze], its_results["its_counterfactual_upper"][:T_analyze]))

    metrics["mcic_rt_its"] = float(calculate_mcic_rt(true_r_0_t_analysis, its_results["its_factual_lower"][:T_analyze], its_results["its_factual_upper"][:T_analyze]))
    metrics["mcic_rcf_its"] = float(calculate_mcic_rt(true_rcf_0_t_analysis, its_results["its_counterfactual_lower"][:T_analyze], its_results["its_counterfactual_upper"][:T_analyze]))

    # --- Metrics for ITS estimated parameters (gamma, lambda) ---
    if num_interventions > 0:
        true_gammas = np.log(sim_data["true_beta_abs_0"] + 1e-9)
        true_lambdas = sim_data["true_lambda_0"]
        
        for k_idx in range(num_interventions):
            # Gamma metrics from ITS
            metrics[f"bias_gamma_{k_idx+1}_its"] = float(calculate_param_bias(true_gammas[k_idx], its_results["its_gamma_est"][k_idx]))
            metrics[f"width_gamma_{k_idx+1}_its"] = float(calculate_param_cri_width(its_results["its_gamma_lower"][k_idx], its_results["its_gamma_upper"][k_idx]))
            metrics[f"cover_gamma_{k_idx+1}_its"] = bool(calculate_param_cri_coverage(true_gammas[k_idx], its_results["its_gamma_lower"][k_idx], its_results["its_gamma_upper"][k_idx]))
            # Lambda metrics from ITS
            metrics[f"bias_lambda_{k_idx+1}_its"] = float(calculate_param_bias(true_lambdas[k_idx], its_results["its_lambda_est"][k_idx]))
            metrics[f"width_lambda_{k_idx+1}_its"] = float(calculate_param_cri_width(its_results["its_lambda_lower"][k_idx], its_results["its_lambda_upper"][k_idx]))
            metrics[f"cover_lambda_{k_idx+1}_its"] = bool(calculate_param_cri_coverage(true_lambdas[k_idx], its_results["its_lambda_lower"][k_idx], its_results["its_lambda_upper"][k_idx]))


    return metrics
    
    # if num_interventions > 0:
        
    #     est_beta_abs_mean, est_beta_abs_lower, est_beta_abs_upper = get_posterior_estimates(posterior_samples, "beta_abs")
    #     est_lambda_mean, est_lambda_lower, est_lambda_upper = get_posterior_estimates(posterior_samples, "lambda")

    #     # Ensure estimates are always treated as 1D arrays for consistent indexing
    #     est_beta_abs_mean = np.atleast_1d(est_beta_abs_mean)
    #     est_beta_abs_lower = np.atleast_1d(est_beta_abs_lower)
    #     est_beta_abs_upper = np.atleast_1d(est_beta_abs_upper)
    #     est_lambda_mean = np.atleast_1d(est_lambda_mean)
    #     est_lambda_lower = np.atleast_1d(est_lambda_lower)
    #     est_lambda_upper = np.atleast_1d(est_lambda_upper)

    #     for k_idx in range(num_interventions):
    #         if k_idx < len(est_beta_abs_mean):
    #             metrics[f"bias_beta_abs_{k_idx+1}"] = float(calculate_param_bias(true_betas_abs[k_idx], est_beta_abs_mean[k_idx]))
    #             metrics[f"width_beta_abs_{k_idx+1}"] = float(calculate_param_cri_width(est_beta_abs_lower[k_idx], est_beta_abs_upper[k_idx]))
    #             metrics[f"cover_beta_abs_{k_idx+1}"] = bool(calculate_param_cri_coverage(true_betas_abs[k_idx], est_beta_abs_lower[k_idx], est_beta_abs_upper[k_idx]))
            
    #         if k_idx < len(est_lambda_mean):
    #             metrics[f"bias_lambda_{k_idx+1}"] = float(calculate_param_bias(true_lambdas[k_idx], est_lambda_mean[k_idx]))
    #             metrics[f"width_lambda_{k_idx+1}"] = float(calculate_param_cri_width(est_lambda_lower[k_idx], est_lambda_upper[k_idx]))
    #             metrics[f"cover_lambda_{k_idx+1}"] = bool(calculate_param_cri_coverage(true_lambdas[k_idx], est_lambda_lower[k_idx], est_lambda_upper[k_idx]))
            
    # return metrics