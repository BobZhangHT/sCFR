# File: benchmarks.py

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
    """Constructs the convolution matrix Q from cases and delay distribution."""
    N = T_sim
    fs_padded = np.zeros(N)
    len_f_s = len(f_s)
    if len_f_s > 0:
        fs_padded[:min(len_f_s, N)] = f_s[:min(len_f_s, N)]
    f_matrix_conv = np.zeros((N, N))
    for i in range(N):
        relevant_fs = fs_padded[:i+1][::-1]
        f_matrix_conv[i, :i+1] = relevant_fs
    return f_matrix_conv @ np.diag(c_t)

def calculate_crude_cfr(d_t, c_t, window=None, cumulative=False):
    """
    Calculates crude CFR (cCFR).

    Args:
        d_t (np.ndarray): Time series of daily deaths.
        c_t (np.ndarray): Time series of daily cases.
        window (int, optional): Size of the moving window for a rolling average. Defaults to None.
        cumulative (bool, optional): If True, calculates cumulative CFR. Defaults to False.

    Returns:
        np.ndarray: The calculated crude CFR time series.
    """
    if cumulative:
        d_t_agg, c_t_agg = np.cumsum(d_t), np.cumsum(c_t)
    elif window:
        d_t_agg = pd.Series(d_t).rolling(window=window, min_periods=1).sum().values
        c_t_agg = pd.Series(c_t).rolling(window=window, min_periods=1).sum().values
    else: # Daily naive d_t / c_t
        d_t_agg, c_t_agg = d_t, c_t

    crude_cfr = np.zeros_like(d_t_agg, dtype=float)
    non_zero_cases = c_t_agg > 0
    crude_cfr[non_zero_cases] = d_t_agg[non_zero_cases] / c_t_agg[non_zero_cases]
    crude_cfr = np.clip(crude_cfr, 0, 1)
    return crude_cfr

def calculate_nishiura_cfr_cumulative(d_t, c_t, f_s):
    """
    Calculates Nishiura-style adjusted cumulative CFR (aCFR).

    Args:
        d_t (np.ndarray): Time series of daily deaths.
        c_t (np.ndarray): Time series of daily cases.
        f_s (np.ndarray): The onset-to-death probability distribution.

    Returns:
        np.ndarray: The calculated aCFR time series.
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
    This method is computationally very fast.

    It assumes a Binomial likelihood for the number of deaths given cases and a uniform Beta(1e-3,1e-3) prior
    for the case fatality rate parameter. The posterior is also a Beta distribution.

    - Campbell, H., & Gustafson, P. (2021). Inferring the COVID-19 infection fatality rate in the community-dwelling population: a simple Bayesian evidence synthesis of seroprevalence study data and imprecise mortality data. Epidemiology & Infection, 149, e243.
    - Mieskolainen, M., Bainbridge, R., Buchmueller, O., Lyons, L., & Wardle, N. (2020). Statistical techniques to estimate the SARS-CoV-2 infection fatality rate. arXiv preprint arXiv:2012.02100.

    Args:
        d_t, c_t, f_s (np.ndarray): Original data and delay distribution.
        alpha (float): Significance level for the CIs (e.g., 0.05 for 95% CI).

    Returns:
        dict: A dictionary containing the lower and upper credible interval bounds.
    """
    T = len(c_t)
    # Define prior parameters 
    # alpha_prior, beta_prior = 1.0, 1.0 # (Beta(1,1) is a uniform prior)
    alpha_prior, beta_prior = 1e-3, 1e-3 # (Beta(1,1) is a uniform prior)

    # --- For cCFR ---
    d_t_cumulative = np.cumsum(d_t)
    c_t_cumulative = np.cumsum(c_t)
    
    # Posterior parameters at each time t: alpha' = alpha + D_t, beta' = beta + (C_t - D_t)
    alpha_posterior_cCFR = alpha_prior + d_t_cumulative
    beta_posterior_cCFR = beta_prior + (c_t_cumulative - d_t_cumulative)
    
    # Calculate quantiles from the posterior Beta distribution for the CI
    ci_cCFR_lower = beta_dist.ppf(alpha / 2, alpha_posterior_cCFR, beta_posterior_cCFR)
    ci_cCFR_upper = beta_dist.ppf(1 - alpha / 2, alpha_posterior_cCFR, beta_posterior_cCFR)
    
    # --- For aCFR ---
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
    [LATEST BENCHMARK] Estimates parameters and curves using a Penalized Poisson MLE ITS model.
    This version correctly handles scenarios with K=0 interventions.
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

    def objective_func(params, Bm, Z, d_t_data, Q, signs, p_alpha, p_gamma):
        """The penalized negative Poisson log-likelihood objective function."""
        alphas = params[:K_spline]
        baseline_trend = Bm @ alphas
        intervention_effect = 0.0
        penalty2 = 0.0
        
        # ** FIX APPLIED HERE: Conditional logic for parameter unpacking and calculation **
        if Z.shape[1] > 0:
            gammas = params[K_spline : K_spline + K_interventions]
            lambdas = params[K_spline + K_interventions:]
            
            betas = np.exp(gammas) * signs
            intervention_effect = np.sum((1 - np.exp(-lambdas * Z)) * betas, axis=1)
            
            penalty2 = p_gamma * np.sum(gammas**2)

        r_t_pred = sigmoid(baseline_trend + intervention_effect)
        mu_pred = np.maximum(Q @ r_t_pred, 1e-9)
        nll = -np.sum(d_t_data * np.log(mu_pred) - mu_pred)
        
        penalty1 = p_alpha * np.sum(np.abs(np.diff(alphas, n=2)))
        
        return nll + penalty1 + penalty2

    # --- Step 1: Tune penalty strength using Grid-Search Time-Series CV ---
    p_alpha_grid = config.PENALTY_GRID_ALPHA
    p_gamma_grid = config.PENALTY_GRID_GAMMA_LAMBDA
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = {}

    p0 = np.zeros(num_params)
    if K_interventions > 0: p0[K_spline + K_interventions:] = 0.1
    bounds = [(None, None)] * (K_spline + K_interventions) + [(1e-6, None)] * K_interventions

    for p_alpha, p_gamma in itertools.product(p_alpha_grid, p_gamma_grid):
        if K_interventions == 0 and p_gamma != p_gamma_grid[0]:
            continue
        
        fold_scores = []
        for train_idx, test_idx in tscv.split(t_array):
            res_cv = minimize(objective_func, x0=p0, 
                              args=(Bm[train_idx], Z_its[train_idx], d_t[train_idx], Q_matrix[np.ix_(train_idx, train_idx)], intervention_signs, p_alpha, p_gamma),
                              method='L-BFGS-B', bounds=bounds)
            if res_cv.success:
                score = objective_func(res_cv.x, Bm[test_idx], Z_its[test_idx], d_t[test_idx], Q_matrix[np.ix_(test_idx, test_idx)], intervention_signs, 0, 0)
                fold_scores.append(score)
        cv_scores[(p_alpha, p_gamma)] = np.mean(fold_scores) if fold_scores else np.inf
    
    best_penalties = min(cv_scores, key=cv_scores.get) if cv_scores else (1.0, 1.0)
    best_penalty_alpha, best_penalty_gamma = best_penalties

    # --- Step 2: Fit final model ---
    final_result = minimize(
        objective_func, x0=p0, 
        args=(Bm, Z_its, d_t, Q_matrix, intervention_signs, best_penalty_alpha, best_penalty_gamma), 
        method='L-BFGS-B', bounds=bounds
    )
    popt = final_result.x if final_result.success else np.full(num_params, np.nan)

    # --- Step 3: Parametric Bootstrap for CIs ---
    alphas_hat = popt[:K_spline]
    baseline_trend_hat = Bm @ alphas_hat
    intervention_effect_hat = 0.0
    gammas_hat, lambdas_hat = np.array([]), np.array([])
    
    if K_interventions > 0:
        gammas_hat = popt[K_spline : K_spline + K_interventions]
        lambdas_hat = popt[K_spline + K_interventions:]
        betas_hat = np.exp(gammas_hat) * intervention_signs
        intervention_effect_hat = np.sum((1 - np.exp(-lambdas_hat * Z_its)) * betas_hat, axis=1)
    
    r_t_hat = sigmoid(baseline_trend_hat + intervention_effect_hat)
    mu_hat = np.maximum(Q_matrix @ r_t_hat, 1e-9)
    
    def bootstrap_iteration(seed):
        rng_boot = np.random.default_rng(seed)
        d_t_boot = rng_boot.poisson(mu_hat)
        boot_res = minimize(objective_func, x0=popt, 
                            args=(Bm, Z_its, d_t_boot, Q_matrix, intervention_signs, best_penalty_alpha, best_penalty_gamma),
                            method='L-BFGS-B', bounds=bounds)
        return boot_res.x if boot_res.success else np.full(num_params, np.nan)
    
    boot_seeds = np.random.randint(0, 1e6, size=config.N_BOOTSTRAPS_ITS)
    boot_params = Parallel(n_jobs=-1)(delayed(bootstrap_iteration)(seed) for seed in boot_seeds)
    boot_params = np.array(boot_params)

    # --- Step 4: Calculate final estimates and CIs ---
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
            gammas_s = p_sample[K_spline : K_spline + K_interventions]
            lambdas_s = p_sample[K_spline + K_interventions:]
            betas_s = np.exp(gammas_s) * intervention_signs
            intervention_effect_s = np.sum((1 - np.exp(-lambdas_s * Z_its)) * betas_s, axis=1)
        
        pred_factual_mc[i, :] = sigmoid(baseline_trend_s + intervention_effect_s)
        pred_counterfactual_mc[i, :] = sigmoid(baseline_trend_s)
    
    return {
        "its_factual_mean": sigmoid(Bm @ alphas_hat + intervention_effect_hat),
        "its_factual_lower": np.nanpercentile(pred_factual_mc, (alpha/2)*100, axis=0),
        "its_factual_upper": np.nanpercentile(pred_factual_mc, (1-alpha/2)*100, axis=0),
        "its_counterfactual_mean": sigmoid(Bm @ alphas_hat),
        "its_counterfactual_lower": np.nanpercentile(pred_counterfactual_mc, (alpha/2)*100, axis=0),
        "its_counterfactual_upper": np.nanpercentile(pred_counterfactual_mc, (1-alpha/2)*100, axis=0),
        "its_gamma_est": gammas_hat,
        "its_gamma_lower": param_ci_lower[K_spline : K_spline + K_interventions] if K_interventions > 0 else np.array([]),
        "its_gamma_upper": param_ci_upper[K_spline : K_spline + K_interventions] if K_interventions > 0 else np.array([]),
        "its_lambda_est": lambdas_hat,
        "its_lambda_lower": param_ci_lower[K_spline + K_interventions:] if K_interventions > 0 else np.array([]),
        "its_lambda_upper": param_ci_upper[K_spline + K_interventions:] if K_interventions > 0 else np.array([]),
    }

    
# def calculate_its_with_parametric_bootstrap(d_t, c_t, f_s, Bm, 
#                                             intervention_times_abs, intervention_signs, 
#                                             alpha=0.05, n_bootstraps=200):
#     """
#     Estimates CFR and parameters using an ITS model with a Poisson likelihood (MLE).

#     Args:
#         d_t, c_t, f_s (np.ndarray): Daily deaths, cases, and delay distribution.
#         Bm (np.ndarray): The B-spline basis matrix for the baseline trend.
#         intervention_times_abs (np.ndarray): Absolute day numbers for intervention starts.
#         intervention_signs (np.ndarray): The signs of the intervention effects.
#         alpha (float): Significance level for the CIs (e.g., 0.05 for 95% CI).
#         n_bootstraps (int): Number of bootstrapping samples for generating prediction CIs.

#     Returns:
#         dict: A comprehensive dictionary of all point estimates and CI bounds.
#     """

#     T, K_spline = Bm.shape
#     K_interventions = len(intervention_times_abs)
#     num_params = K_spline + 2 * K_interventions
    
#     Q_matrix = construct_Q_matrix(c_t, f_s, T)
#     Z_its = np.zeros((T, K_interventions))
#     t_array = np.arange(T)
#     if K_interventions > 0:
#         for k in range(K_interventions):
#             Z_its[:, k] = np.maximum(0, t_array - intervention_times_abs[k])
    
#     def objective_func(params, Bm, Z, d_t_data, Q, signs, p_alpha, p_gamma):
#         """The penalized negative Poisson log-likelihood objective function."""
#         alphas = params[:K_spline]
#         baseline_trend = Bm @ alphas
#         intervention_effect = 0.0
#         penalty2 = 0.0
        
#         # ** FIX APPLIED HERE: Conditional logic for parameter unpacking and calculation **
#         if Z.shape[1] > 0:
#             gammas = params[K_spline : K_spline + K_interventions]
#             lambdas = params[K_spline + K_interventions:]
            
#             betas = np.exp(gammas) * signs
#             intervention_effect = np.sum((1 - np.exp(-lambdas * Z)) * betas, axis=1)
            
#             penalty2 = p_gamma * np.sum(gammas**2)
        
#         r_t_pred = sigmoid(baseline_trend + intervention_effect)
#         mu_pred = np.maximum(Q @ r_t_pred, 1e-9)
#         nll = -np.sum(d_t_data * np.log(mu_pred) - mu_pred)

#         penalty1 = p_alpha * np.sum(np.abs(np.diff(alphas, n=2)))

#         return nll + penalty1 + penalty2
    
#     # --- Step 1: Fit final model on all original data to get point estimates (popt) ---
#     penalty_grid = [0.01, 0.1, 1, 10, 100]
#     tscv = TimeSeriesSplit(n_splits=3)
#     cv_scores = {}
    
#     p0 = np.zeros(num_params)
#     if K_interventions > 0: p0[K_spline + K_interventions:] = 0.1
#     bounds = [(None, None)] * (K_spline + K_interventions) + [(1e-6, None)] * K_interventions

#     for p_alpha, p_gamma in itertools.product(penalty_grid,penalty_grid):
#         if K_interventions == 0 and (p_gamma != penalty_grid[0]):
#             continue
#         fold_scores = []
#         for train_idx, test_idx in tscv.split(t_array):
#             res_cv = minimize(objective_func, x0=p0, 
#                               args=(Bm[train_idx], Z_its[train_idx], d_t[train_idx], Q_matrix[np.ix_(train_idx, train_idx)], intervention_signs, p_alpha, p_gamma),
#                               method='L-BFGS-B', bounds=bounds)
#             if res_cv.success:
#                 score = objective_func(res_cv.x, Bm[test_idx], Z_its[test_idx], d_t[test_idx], Q_matrix[np.ix_(test_idx, test_idx)], intervention_signs, 0, 0)
#                 fold_scores.append(score)
#         cv_scores[(p_alpha,p_gamma)] = np.mean(fold_scores) if fold_scores else np.inf
    
#     best_penalties = min(cv_scores, key=cv_scores.get)
#     best_penalty_alpha, best_penalty_gamma = best_penalties
    
#     final_result = minimize(
#         objective_func, x0=p0, 
#         args=(Bm, Z_its, d_t, Q_matrix, intervention_signs, best_penalty_alpha, best_penalty_gamma), 
#         method='L-BFGS-B', bounds=bounds
#     )
#     popt = final_result.x if final_result.success else np.full(num_params, np.nan)
    
#     # --- Step 2: Parametric Bootstrap for Confidence Intervals ---
#     alphas_hat, gammas_hat, lambdas_hat = popt[:K_spline], popt[K_spline:-K_interventions], popt[-K_interventions:]
#     intervention_effect = 0.0
#     if K_interventions > 0:
#         betas_hat = np.exp(gammas_hat) * intervention_signs
#         intervention_effect = np.sum((1 - np.exp(-lambdas_hat * Z_its)) * betas_hat, axis=1)
#     r_t_hat = sigmoid(Bm @ alphas_hat + intervention_effect)
#     mu_hat = np.maximum(Q_matrix @ r_t_hat, 1e-9)
    
#     def bootstrap_iteration(seed):
#         rng_boot = np.random.default_rng(seed)
#         d_t_boot = rng_boot.poisson(mu_hat)
#         boot_res = minimize(
#             objective_func, x0=popt, 
#             args=(Bm, Z_its, d_t_boot, Q_matrix, intervention_signs, best_penalty_alpha, best_penalty_gamma),
#             method='L-BFGS-B', bounds=bounds)
#         return boot_res.x if boot_res.success else np.full(num_params, np.nan)
    
#     boot_seeds = np.random.randint(0, 1e6, size=n_bootstraps)
#     boot_params = Parallel(n_jobs=-1)(delayed(bootstrap_iteration)(seed) for seed in boot_seeds)
#     boot_params = np.array(boot_params)

#     # --- Step 3: Calculate Prediction CIs via Monte Carlo from the bootstrap parameter distribution ---
#     param_ci_lower = np.nanpercentile(boot_params, (alpha/2)*100, axis=0)
#     param_ci_upper = np.nanpercentile(boot_params, (1-alpha/2)*100, axis=0)
    
#     pred_factual_mc = np.full((n_bootstraps, T), np.nan)
#     pred_counterfactual_mc = np.full((n_bootstraps, T), np.nan)
    
#     for i, p_sample in enumerate(boot_params):
#         if np.any(np.isnan(p_sample)): continue
#         alphas_s, gammas_s, lambdas_s = p_sample[:K_spline], p_sample[K_spline:-K_interventions], p_sample[-K_interventions:]
#         intervention_effect_s = 0.0
#         if K_interventions > 0:
#             betas_s = np.exp(gammas_s) * intervention_signs
#             intervention_effect_s = np.sum((1 - np.exp(-lambdas_s * Z_its)) * betas_s, axis=1)
#         baseline_trend_s = Bm @ alphas_s
        
#         pred_factual_mc[i, :] = sigmoid(baseline_trend_s + intervention_effect_s)
#         pred_counterfactual_mc[i, :] = sigmoid(baseline_trend_s)

#     factual_ci_lower = np.nanpercentile(pred_factual_mc, (alpha/2)*100, axis=0)
#     factual_ci_upper = np.nanpercentile(pred_factual_mc, (1-alpha/2)*100, axis=0)
#     cf_ci_lower = np.nanpercentile(pred_counterfactual_mc, (alpha/2)*100, axis=0)
#     cf_ci_upper = np.nanpercentile(pred_counterfactual_mc, (1-alpha/2)*100, axis=0)
    
#     its_results = {
#         "its_factual_mean": r_t_hat,
#         "its_factual_lower": factual_ci_lower,
#         "its_factual_upper": factual_ci_upper,
#         "its_counterfactual_mean": sigmoid(Bm @ alphas_hat),
#         "its_counterfactual_lower": cf_ci_lower,
#         "its_counterfactual_upper": cf_ci_upper,
#         "its_gamma_est": gammas_hat,
#         "its_gamma_lower": param_ci_lower[K_spline:-K_interventions],
#         "its_gamma_upper": param_ci_upper[K_spline:-K_interventions],
#         "its_lambda_est": lambdas_hat,
#         "its_lambda_lower": param_ci_lower[-K_interventions:],
#         "its_lambda_upper": param_ci_upper[-K_interventions:],
#     }

#     return its_results