import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
import config

def get_posterior_estimates(posterior_samples, key_name, percentiles=(2.5, 97.5)):
    """
    Extracts posterior mean and credible intervals for a given parameter.

    Args:
        posterior_samples (dict): A dictionary of posterior samples.
        key_name (str): The name of the parameter.
        percentiles (tuple, optional): The percentiles for the credible interval. Defaults to (2.5, 97.5).

    Returns:
        tuple: A tuple containing the mean, lower bound, and upper bound of the parameter estimates.
    """
    if not posterior_samples or key_name not in posterior_samples:
        shape_ref = config.T_SERIES_LENGTH_SIM
        nan_array = np.full(shape_ref, np.nan) if key_name in ['p', 'p_cf'] else np.array([np.nan])
        return nan_array, nan_array, nan_array
    samples = posterior_samples[key_name]
    mean_est, lower_est, upper_est = np.mean(samples, axis=0), np.percentile(samples, percentiles[0], axis=0), np.percentile(samples, percentiles[1], axis=0)
    return mean_est, lower_est, upper_est

def calculate_mae_rt(true_r_t, estimated_r_t_mean):
    """
    Calculates the Mean Absolute Error (MAE) for the fatality rate.

    Args:
        true_r_t (numpy.ndarray): The true fatality rate time series.
        estimated_r_t_mean (numpy.ndarray): The estimated mean fatality rate time series.

    Returns:
        float: The Mean Absolute Error.
    """
    return np.mean(np.abs(estimated_r_t_mean - true_r_t))

def calculate_mciw_rt(r_t_lower, r_t_upper):
    """
    Calculates the Mean Credible Interval Width (MCIW) for the fatality rate.

    Args:
        r_t_lower (numpy.ndarray): The lower bound of the credible interval.
        r_t_upper (numpy.ndarray): The upper bound of the credible interval.

    Returns:
        float: The Mean Credible Interval Width.
    """
    return np.mean(r_t_upper - r_t_lower)

def calculate_mcic_rt(true_r_t, r_t_lower, r_t_upper):
    """
    Calculates the Mean Credible Interval Coverage (MCIC) for the fatality rate.

    Args:
        true_r_t (numpy.ndarray): The true fatality rate time series.
        r_t_lower (numpy.ndarray): The lower bound of the credible interval.
        r_t_upper (numpy.ndarray): The upper bound of the credible interval.

    Returns:
        float: The Mean Credible Interval Coverage.
    """
    return np.mean((true_r_t >= r_t_lower) & (true_r_t <= r_t_upper))

def calculate_param_bias(true_param_val, est_param_mean):
    """
    Calculates the bias for a single parameter.

    Args:
        true_param_val (float): The true value of the parameter.
        est_param_mean (float): The estimated mean of the parameter.

    Returns:
        float: The bias of the parameter estimate.
    """
    return est_param_mean - true_param_val

def calculate_param_cri_width(param_lower_ci, param_upper_ci):
    """
    Calculates the credible interval width for a single parameter.

    Args:
        param_lower_ci (float): The lower bound of the credible interval.
        param_upper_ci (float): The upper bound of the credible interval.

    Returns:
        float: The width of the credible interval.
    """
    return param_upper_ci - param_lower_ci

def calculate_param_cri_coverage(true_param_val, param_lower_ci, param_upper_ci):
    """
    Calculates the credible interval coverage for a single parameter.

    Args:
        true_param_val (float): The true value of the parameter.
        param_lower_ci (float): The lower bound of the credible interval.
        param_upper_ci (float): The upper bound of the credible interval.

    Returns:
        bool: True if the true value is within the credible interval, False otherwise.
    """
    return (true_param_val >= param_lower_ci) & (true_param_val <= param_upper_ci)

def collect_all_metrics(sim_data, posterior_scfr, benchmarks_r_t, benchmark_cis, its_results):
    """
    Collects all specified metrics for the sCFR model and all benchmarks.

    Args:
        sim_data (dict): A dictionary of the simulated data.
        posterior_scfr (dict): A dictionary of posterior samples from the sCFR model.
        benchmarks_r_t (dict): A dictionary of the estimated fatality rates from benchmark models.
        benchmark_cis (dict): A dictionary of the credible/confidence intervals from benchmark models.
        its_results (dict): A dictionary of results from the ITS model.

    Returns:
        dict: A dictionary containing all calculated metrics.
    """
    metrics = {"scenario_id": sim_data["scenario_id"], "run_seed": sim_data["run_seed"]}
    T_analyze = config.T_ANALYSIS_LENGTH
    true_r_t_analysis, true_rcf_0_t_analysis = sim_data["true_r_0_t"][:T_analyze], sim_data["true_rcf_0_t"][:T_analyze]
    
    if posterior_scfr:
        r_t_mean, r_t_lower, r_t_upper = get_posterior_estimates(posterior_scfr, "p")
        rcf_t_mean, rcf_t_lower, rcf_t_upper = get_posterior_estimates(posterior_scfr, "p_cf")
        
        metrics["mae_rt_sCFR"] = float(calculate_mae_rt(true_r_t_analysis, r_t_mean[:T_analyze]))
        metrics["mciw_rt_sCFR"] = float(calculate_mciw_rt(r_t_lower[:T_analyze], r_t_upper[:T_analyze]))
        metrics["mcic_rt_sCFR"] = float(calculate_mcic_rt(true_r_t_analysis, r_t_lower[:T_analyze], r_t_upper[:T_analyze]))
        
        metrics["mae_rcf_sCFR"] = float(calculate_mae_rt(true_rcf_0_t_analysis, rcf_t_mean[:T_analyze]))
        metrics["mciw_rcf_sCFR"] = float(calculate_mciw_rt(rcf_t_lower[:T_analyze], rcf_t_upper[:T_analyze]))
        metrics["mcic_rcf_sCFR"] = float(calculate_mcic_rt(true_rcf_0_t_analysis, rcf_t_lower[:T_analyze], rcf_t_upper[:T_analyze]))

    num_interventions = sim_data["num_interventions_true_K"]
    if num_interventions > 0:
        true_beta_abs = sim_data["true_beta_abs_0"]
        true_lambdas = sim_data["true_lambda_0"]
        
        if posterior_scfr:
            est_beta_abs_m, est_beta_abs_l, est_beta_abs_u = map(np.atleast_1d, get_posterior_estimates(posterior_scfr, "beta_abs"))
            est_lambda_m, est_lambda_l, est_lambda_u = map(np.atleast_1d, get_posterior_estimates(posterior_scfr, "lambda"))
            
            for k in range(num_interventions):
                if k < len(est_beta_abs_m):
                    metrics[f"bias_gamma_{k+1}_sCFR"] = float(calculate_param_bias(true_beta_abs[k], est_beta_abs_m[k]))
                    metrics[f"width_gamma_{k+1}_sCFR"] = float(calculate_param_cri_width(est_beta_abs_l[k], est_beta_abs_u[k]))
                    metrics[f"cover_gamma_{k+1}_sCFR"] = bool(calculate_param_cri_coverage(true_beta_abs[k], est_beta_abs_l[k], est_beta_abs_u[k]))
                if k < len(est_lambda_m):
                    metrics[f"bias_lambda_{k+1}_sCFR"] = float(calculate_param_bias(true_lambdas[k], est_lambda_m[k]))
                    metrics[f"width_lambda_{k+1}_sCFR"] = float(calculate_param_cri_width(est_lambda_l[k], est_lambda_u[k]))
                    metrics[f"cover_lambda_{k+1}_sCFR"] = bool(calculate_param_cri_coverage(true_lambdas[k], est_lambda_l[k], est_lambda_u[k]))
        
    if benchmarks_r_t and benchmark_cis:
        for bench_name, r_t_est_bench in benchmarks_r_t.items():
            metrics[f"mae_rt_{bench_name}"] = float(calculate_mae_rt(true_r_t_analysis, r_t_est_bench[:T_analyze]))
            lower_ci, upper_ci = benchmark_cis.get(f"{bench_name}_lower", []), benchmark_cis.get(f"{bench_name}_upper", [])
            if len(lower_ci) > 0 and len(upper_ci) > 0:
                metrics[f"mciw_rt_{bench_name}"] = float(calculate_mciw_rt(lower_ci[:T_analyze], upper_ci[:T_analyze]))
                metrics[f"mcic_rt_{bench_name}"] = float(calculate_mcic_rt(true_r_t_analysis, lower_ci[:T_analyze], upper_ci[:T_analyze]))

    if its_results:
        metrics["mae_rt_its"] = float(calculate_mae_rt(true_r_t_analysis, its_results["its_factual_mean"][:T_analyze]))
        metrics["mciw_rt_its"] = float(calculate_mciw_rt(its_results["its_factual_lower"][:T_analyze], its_results["its_factual_upper"][:T_analyze]))
        metrics[f"mcic_rt_its"] = float(calculate_mcic_rt(true_r_t_analysis, its_results["its_factual_lower"][:T_analyze], its_results["its_factual_upper"][:T_analyze]))
        metrics[f"mae_rcf_its"] = float(calculate_mae_rt(true_rcf_0_t_analysis, its_results["its_counterfactual_mean"][:T_analyze]))
        metrics[f"mciw_rcf_its"] = float(calculate_mciw_rt(its_results["its_counterfactual_lower"][:T_analyze], its_results["its_counterfactual_upper"][:T_analyze]))
        metrics[f"mcic_rcf_its"] = float(calculate_mcic_rt(true_rcf_0_t_analysis, its_results["its_counterfactual_lower"][:T_analyze], its_results["its_counterfactual_upper"][:T_analyze]))

        if num_interventions > 0:
            true_beta_abs = sim_data["true_beta_abs_0"]
            true_lambdas = sim_data["true_lambda_0"]
            for k in range(num_interventions):
                if k < len(its_results["its_beta_abs_est"]):
                    metrics[f"bias_gamma_{k+1}_its"] = float(calculate_param_bias(true_beta_abs[k], its_results["its_beta_abs_est"][k]))
                    metrics[f"width_gamma_{k+1}_its"] = float(calculate_param_cri_width(its_results["its_beta_abs_lower"][k], its_results["its_beta_abs_upper"][k]))
                    metrics[f"cover_gamma_{k+1}_its"] = bool(calculate_param_cri_coverage(true_beta_abs[k], its_results["its_beta_abs_lower"][k], its_results["its_beta_abs_upper"][k]))
                
                if k < len(its_results["its_lambda_est"]):
                    metrics[f"bias_lambda_{k+1}_its"] = float(calculate_param_bias(true_lambdas[k], its_results["its_lambda_est"][k]))
                    metrics[f"width_lambda_{k+1}_its"] = float(calculate_param_cri_width(its_results["its_lambda_lower"][k], its_results["its_lambda_upper"][k]))
                    metrics[f"cover_lambda_{k+1}_its"] = bool(calculate_param_cri_coverage(true_lambdas[k], its_results["its_lambda_lower"][k], its_results["its_lambda_upper"][k]))
            
    return metrics
