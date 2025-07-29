import numpy as np
import pandas as pd
from scipy.stats import norm, beta as beta_dist
from scipy.special import logit, expit as sigmoid
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
from joblib import Parallel, delayed
import itertools
import config

def construct_Q_matrix(c_t, f_s, T_sim):
    """
    Constructs the convolution matrix Q from daily cases and the delay distribution.

    Args:
        c_t (numpy.ndarray): Array of daily case counts.
        f_s (numpy.ndarray): The onset-to-death delay distribution.
        T_sim (int): The total simulation time length.

    Returns:
        numpy.ndarray: The convolution matrix Q.
    """
    N = T_sim
    fs_padded = np.zeros(N)
    len_f_s = len(f_s)
    if len_f_s > 0:
        fs_padded[:min(len_f_s, N)] = f_s[:min(len_f_s, N)]
    
    f_matrix_conv = np.zeros((N,N))
    for i in range(N):
        relevant_fs = fs_padded[:i+1][::-1]
        f_matrix_conv[i, :i+1] = relevant_fs
        
    return f_matrix_conv @ np.diag(c_t)

def calculate_crude_cfr(d_t, c_t, window=None, cumulative=False):
    """
    Calculates the crude Case Fatality Rate (cCFR).

    Args:
        d_t (numpy.ndarray): Array of daily death counts.
        c_t (numpy.ndarray): Array of daily case counts.
        window (int, optional): The rolling window size. Defaults to None.
        cumulative (bool, optional): If True, calculates cumulative CFR. Defaults to False.

    Returns:
        numpy.ndarray: The calculated crude CFR.
    """
    if cumulative:
        d_t_agg, c_t_agg = np.cumsum(d_t), np.cumsum(c_t)
    elif window:
        d_t_agg = pd.Series(d_t).rolling(window=window, min_periods=1).sum().values
        c_t_agg = pd.Series(c_t).rolling(window=window, min_periods=1).sum().values
    else:
        d_t_agg, c_t_agg = d_t, c_t

    crude_cfr = np.zeros_like(d_t_agg, dtype=float)
    non_zero_cases = c_t_agg > 0
    crude_cfr[non_zero_cases] = d_t_agg[non_zero_cases] / c_t_agg[non_zero_cases]
    crude_cfr = np.clip(crude_cfr, 0, 1)
    return crude_cfr

def calculate_nishiura_cfr_cumulative(d_t, c_t, f_s):
    """
    Calculates the Nishiura-style adjusted cumulative CFR (aCFR).

    Args:
        d_t (numpy.ndarray): Array of daily death counts.
        c_t (numpy.ndarray): Array of daily case counts.
        f_s (numpy.ndarray): The onset-to-death delay distribution.

    Returns:
        numpy.ndarray: The adjusted cumulative CFR.
    """
    T = len(c_t)
    d_t_cumulative = np.cumsum(d_t)
    
    fs_padded = np.zeros(T)
    fs_padded[:min(len(f_s), T)] = f_s[:min(len(f_s), T)]
    
    convolved_cases = np.convolve(c_t, fs_padded, mode='full')[:T]
    c_t_d_cumulative = np.cumsum(convolved_cases)

    nishiura_cfr = np.zeros_like(d_t_cumulative, dtype=float)
    non_zero_adj_cases = c_t_d_cumulative > 1e-9
    nishiura_cfr[non_zero_adj_cases] = d_t_cumulative[non_zero_adj_cases] / c_t_d_cumulative[non_zero_adj_cases]
    nishiura_cfr = np.clip(nishiura_cfr, 0, 1)
    return nishiura_cfr
    
def calculate_benchmark_cis_with_bayesian(d_t, c_t, f_s, alpha=0.05):
    """
    Generates credible intervals for cCFR and aCFR using a cumulative Beta-Binomial model.

    Args:
        d_t (numpy.ndarray): Array of daily death counts.
        c_t (numpy.ndarray): Array of daily case counts.
        f_s (numpy.ndarray): The onset-to-death delay distribution.
        alpha (float, optional): The significance level for the credible intervals. Defaults to 0.05.

    Returns:
        dict: A dictionary containing the lower and upper credible intervals for cCFR and aCFR.
    """
    T = len(c_t)
    alpha_prior, beta_prior = 1e-3, 1e-3

    d_t_cumulative = np.cumsum(d_t)
    c_t_cumulative = np.cumsum(c_t)
    alpha_posterior_cCFR = alpha_prior + d_t_cumulative
    beta_posterior_cCFR = beta_prior + (c_t_cumulative - d_t_cumulative)
    ci_cCFR_lower = beta_dist.ppf(alpha / 2, alpha_posterior_cCFR, beta_posterior_cCFR)
    ci_cCFR_upper = beta_dist.ppf(1 - alpha / 2, alpha_posterior_cCFR, beta_posterior_cCFR)
    
    fs_padded = np.zeros(T)
    fs_padded[:min(len(f_s), T)] = f_s[:min(len(f_s), T)]
    convolved_cases = np.convolve(c_t, fs_padded, mode='full')[:T]
    c_t_adj_cumulative = np.cumsum(convolved_cases)
    alpha_posterior_aCFR = alpha_prior + d_t_cumulative
    beta_posterior_aCFR = beta_prior + (c_t_adj_cumulative - d_t_cumulative)
    ci_aCFR_lower = beta_dist.ppf(alpha / 2, alpha_posterior_aCFR, beta_posterior_aCFR)
    ci_aCFR_upper = beta_dist.ppf(1 - alpha / 2, alpha_posterior_aCFR, beta_posterior_aCFR)

    return {
        "cCFR_cumulative_lower": np.nan_to_num(ci_cCFR_lower),
        "cCFR_cumulative_upper": np.nan_to_num(ci_cCFR_upper),
        "aCFR_cumulative_lower": np.nan_to_num(ci_aCFR_lower),
        "aCFR_cumulative_upper": np.nan_to_num(ci_aCFR_upper)
    }

def calculate_its_with_penalized_mle(d_t, c_t, f_s, Bm, intervention_times_abs, intervention_signs, alpha=0.05):
    """
    Estimates parameters and curves using a Penalized Poisson MLE Interrupted Time Series (ITS) model.

    Args:
        d_t (numpy.ndarray): Array of daily death counts.
        c_t (numpy.ndarray): Array of daily case counts.
        f_s (numpy.ndarray): The onset-to-death delay distribution.
        Bm (numpy.ndarray): The B-spline basis matrix.
        intervention_times_abs (list): A list of absolute intervention times.
        intervention_signs (list): A list of signs for the intervention effects.
        alpha (float, optional): The significance level for confidence intervals. Defaults to 0.05.

    Returns:
        dict: A dictionary of results including estimated curves and parameters.
    """
    T, K_spline = Bm.shape
    K_interventions = len(intervention_times_abs)
    num_params = K_spline + 2 * K_interventions

    Q_matrix = construct_Q_matrix(c_t, f_s, T)
    Z_its = np.zeros((T, K_interventions))
    t_array = np.arange(T)
    if K_interventions > 0:
        for k in range(K_interventions):
            Z_its[:, k] = np.maximum(0, t_array - intervention_times_abs[k])

    def objective_func(params, Bm, Z, d_t_data, Q, signs, p_alpha, p_beta):
        alphas = params[:K_spline]
        baseline_trend = Bm @ alphas
        intervention_effect, penalty2 = 0.0, 0.0

        if Z.shape[1] > 0:
            beta_abs = params[K_spline : K_spline + K_interventions]
            lambdas = params[K_spline + K_interventions:]
            
            betas = beta_abs * signs
            intervention_effect = np.sum((1 - np.exp(-lambdas * Z)) * betas, axis=1)
            penalty2 = p_beta * np.sum(beta_abs**2)

        r_t_pred = sigmoid(baseline_trend + intervention_effect)
        mu_pred = np.maximum(Q @ r_t_pred, 1e-9)
        nll = -np.sum(d_t_data * np.log(mu_pred) - mu_pred)
        
        penalty1 = p_alpha * np.sum(np.abs(np.diff(alphas, n=2)))
        
        return nll + penalty1 + penalty2

    p_alpha_grid = config.PENALTY_GRID_ALPHA
    p_beta_grid = config.PENALTY_GRID_BETA
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = {}

    p0 = np.zeros(num_params)
    if K_interventions > 0: p0[K_spline + K_interventions:] = 0.1
    bounds = [(None, None)] * K_spline + [(0, None)] * K_interventions + [(1e-6, None)] * K_interventions

    for p_alpha, p_beta in itertools.product(p_alpha_grid, p_beta_grid):
        if K_interventions == 0 and p_beta != p_beta_grid[0]:
            continue
        
        fold_scores = []
        for train_idx, test_idx in tscv.split(t_array):
            res_cv = minimize(objective_func, x0=p0, 
                              args=(Bm[train_idx], Z_its[train_idx], d_t[train_idx], Q_matrix[np.ix_(train_idx, train_idx)], intervention_signs, p_alpha, p_beta),
                              method='L-BFGS-B', bounds=bounds)
            if res_cv.success:
                score = objective_func(res_cv.x, Bm[test_idx], Z_its[test_idx], d_t[test_idx], Q_matrix[np.ix_(test_idx, test_idx)], intervention_signs, 0, 0)
                fold_scores.append(score)
        cv_scores[(p_alpha, p_beta)] = np.mean(fold_scores) if fold_scores else np.inf
    
    best_penalties = min(cv_scores, key=cv_scores.get) if cv_scores else (1.0, 1.0)
    best_penalty_alpha, best_penalty_beta = best_penalties

    final_result = minimize(
        objective_func, x0=p0, 
        args=(Bm, Z_its, d_t, Q_matrix, intervention_signs, best_penalty_alpha, best_penalty_beta), 
        method='L-BFGS-B', bounds=bounds
    )
    popt = final_result.x if final_result.success else np.full(num_params, np.nan)

    alphas_hat, beta_abs_hat, lambdas_hat = (popt[:K_spline], 
                                             popt[K_spline:-K_interventions] if K_interventions > 0 else np.array([]), 
                                             popt[-K_interventions:] if K_interventions > 0 else np.array([]))
    
    baseline_trend_hat = Bm @ alphas_hat
    intervention_effect_hat = 0.0
    if K_interventions > 0:
        betas_hat = beta_abs_hat * intervention_signs
        intervention_effect_hat = np.sum((1 - np.exp(-lambdas_hat * Z_its)) * betas_hat, axis=1)
    
    r_t_hat = sigmoid(baseline_trend_hat + intervention_effect_hat)
    mu_hat = np.maximum(Q_matrix @ r_t_hat, 1e-9)
    
    def bootstrap_iteration(seed):
        rng_boot = np.random.default_rng(seed)
        d_t_boot = rng_boot.poisson(mu_hat)
        
        boot_res = minimize(
            objective_func, x0=popt, 
            args=(Bm, Z_its, d_t_boot, Q_matrix, intervention_signs, best_penalty_alpha, best_penalty_beta),
            method='L-BFGS-B', bounds=bounds)
        return boot_res.x if boot_res.success else np.full(num_params, np.nan)
    
    boot_seeds = np.random.randint(0, 1e6, size=config.N_BOOTSTRAPS_ITS)
    boot_params = Parallel(n_jobs=-1)(delayed(bootstrap_iteration)(seed) for seed in boot_seeds)
    boot_params = np.array(boot_params)

    param_ci_lower = np.nanpercentile(boot_params, (alpha/2)*100, axis=0)
    param_ci_upper = np.nanpercentile(boot_params, (1-alpha/2)*100, axis=0)
    
    pred_factual_mc = np.full((config.N_BOOTSTRAPS_ITS, T), np.nan)
    pred_counterfactual_mc = np.full((config.N_BOOTSTRAPS_ITS, T), np.nan)
    
    for i, p_sample in enumerate(boot_params):
        if np.any(np.isnan(p_sample)): continue
        alphas_s = p_sample[:K_spline]
        baseline_trend_s = Bm @ alphas_s
        intervention_effect_s = 0.0
        
        if K_interventions > 0:
            beta_abs_s = p_sample[K_spline : K_spline + K_interventions]
            lambdas_s = p_sample[K_spline + K_interventions:]
            betas_s = beta_abs_s * intervention_signs
            intervention_effect_s = np.sum((1 - np.exp(-lambdas_s * Z_its)) * betas_s, axis=1)
        
        pred_factual_mc[i, :] = sigmoid(baseline_trend_s + intervention_effect_s)
        pred_counterfactual_mc[i, :] = sigmoid(baseline_trend_s)
    
    return {
        "its_factual_mean": np.nanmean(pred_factual_mc, axis=0),
        "its_factual_lower": np.nanpercentile(pred_factual_mc, (alpha/2)*100, axis=0),
        "its_factual_upper": np.nanpercentile(pred_factual_mc, (1-alpha/2)*100, axis=0),
        "its_counterfactual_mean": np.nanmean(pred_counterfactual_mc, axis=0),
        "its_counterfactual_lower": np.nanpercentile(pred_counterfactual_mc, (alpha/2)*100, axis=0),
        "its_counterfactual_upper": np.nanpercentile(pred_counterfactual_mc, (1-alpha/2)*100, axis=0),
        "its_beta_abs_est": np.nanmedian(boot_params, axis=0)[K_spline:K_spline+K_interventions] if K_interventions > 0 else np.array([]),
        "its_beta_abs_lower": param_ci_lower[K_spline:K_spline+K_interventions] if K_interventions > 0 else np.array([]),
        "its_beta_abs_upper": param_ci_upper[K_spline:K_spline+K_interventions] if K_interventions > 0 else np.array([]),
        "its_lambda_est": np.nanmedian(boot_params, axis=0)[K_spline+K_interventions:] if K_interventions > 0 else np.array([]),
        "its_lambda_lower": param_ci_lower[K_spline+K_interventions:] if K_interventions > 0 else np.array([]),
        "its_lambda_upper": param_ci_upper[K_spline+K_interventions:] if K_interventions > 0 else np.array([]),
    }

def run_all_benchmarks(sim_data):
    
    benchmark_r_t_estimates = {
        "cCFR_cumulative": calculate_crude_cfr(sim_data["d_t"], sim_data["c_t"], cumulative=True),
        "aCFR_cumulative": calculate_nishiura_cfr_cumulative(sim_data["d_t"], sim_data["c_t"], sim_data["f_s_true"])
    }
    benchmark_cis = calculate_benchmark_cis_with_bayesian(sim_data["d_t"], sim_data["c_t"], sim_data["f_s_true"])
    
    its_results = calculate_its_with_penalized_mle(
        d_t=sim_data["d_t"], c_t=sim_data["c_t"], f_s=sim_data["f_s_true"],
        Bm=sim_data["Bm_true"],
        intervention_times_abs=sim_data["true_intervention_times_0_abs"],
        intervention_signs=sim_data["beta_signs_true"]
    )

    all_benchmark_results = {**benchmark_r_t_estimates, **benchmark_cis, **its_results}

    return all_benchmark_results