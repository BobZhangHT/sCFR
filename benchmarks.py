# File: benchmarks.py

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, norm
from scipy.special import logit, expit as sigmoid
from scipy.optimize import curve_fit

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

def calculate_its_with_nls(d_t, c_t, Bm, intervention_times_abs, intervention_signs, alpha=0.05, n_mc_ci=500):
    """
    Estimates factual/counterfactual CFR and parameters (gamma, lambda) using a 
    Non-Linear Least Squares (NLS) Interrupted Time Series model. Also provides CIs.

    Returns:
        dict: A comprehensive dictionary of all point estimates and CI bounds.
    """
    T = len(d_t)
    K_spline = Bm.shape[1]
    K_interventions = len(intervention_times_abs)

    # --- Define the non-linear model function for curve_fit ---
    def nls_model(t_indices_and_Z, *params):
        # Unpack design matrices and parameters
        _t, _Z = t_indices_and_Z # This way we pass Z without recomputing it
        alphas = params[:K_spline]
        gammas = params[K_spline : K_spline + K_interventions]
        lambdas = params[K_spline + K_interventions:]
        
        # Reconstruct intervention effect with current lambda
        betas = np.exp(gammas) * intervention_signs
        lag_matrix = 1 - np.exp(-lambdas * _Z)
        intervention_effect = np.sum(lag_matrix * betas, axis=1)

        # Predicted logit(CFR)
        logit_pred = Bm @ alphas + intervention_effect
        return logit_pred

    # Prepare target variable (smoothed daily logit CFR)
    daily_d_smooth = pd.Series(d_t).rolling(window=7, min_periods=1, center=True).mean().values
    daily_c_smooth = pd.Series(c_t).rolling(window=7, min_periods=1, center=True).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        naive_cfr = daily_d_smooth / (daily_c_smooth + 1e-6)
        logit_cfr_target = logit(np.clip(naive_cfr, 1e-6, 1 - 1e-6))
    logit_cfr_target = np.nan_to_num(logit_cfr_target, nan=np.median(logit_cfr_target[~np.isnan(logit_cfr_target)]))

    # Prepare Z matrix for interventions (t - t_k)_+
    Z_nls = np.zeros((T, K_interventions))
    t_array = np.arange(T)
    for k in range(K_interventions):
        Z_nls[:, k] = np.maximum(0, t_array - intervention_times_abs[k])
    
    # Initial guesses for parameters
    p0 = np.zeros(K_spline + 2 * K_interventions)
    if K_interventions > 0:
        p0[K_spline + K_interventions:] = 0.1 # Initial guess for lambdas

    # Fit the NLS model
    try:
        popt, pcov = curve_fit(nls_model, (t_array, Z_nls), logit_cfr_target, p0=p0, maxfev=5000)
        perr = np.sqrt(np.diag(pcov)) # Standard errors of parameters
    except (RuntimeError, ValueError):
        popt = np.full_like(p0, np.nan)
        perr = np.full_like(p0, np.nan)
        pcov = np.full_like((p0,p0), np.nan)

    # --- Extract Parameter Estimates and CIs ---
    gamma_est = popt[K_spline : K_spline + K_interventions]
    lambda_est = popt[K_spline + K_interventions:]
    gamma_se, lambda_se = perr[K_spline : K_spline + K_interventions], perr[K_spline + K_interventions:]

    z_crit = norm.ppf(1 - alpha / 2)
    gamma_ci_lower, gamma_ci_upper = gamma_est - z_crit * gamma_se, gamma_est + z_crit * gamma_se
    lambda_ci_lower, lambda_ci_upper = lambda_est - z_crit * lambda_se, lambda_est + z_crit * lambda_se

    # --- Calculate Prediction CIs via Monte Carlo ---
    pred_factual_mc = np.zeros((n_mc_ci, T))
    pred_counterfactual_mc = np.zeros((n_mc_ci, T))

    try:
        param_samples = np.random.multivariate_normal(popt, pcov, size=n_mc_ci)
        for i, p_sample in enumerate(param_samples):
            pred_factual_mc[i, :] = nls_model((t_array, Z_nls), *p_sample)
            # For counterfactual, we only use the spline part of the parameters
            pred_counterfactual_mc[i, :] = Bm @ p_sample[:K_spline]
    except (np.linalg.LinAlgError, ValueError): # If pcov is not positive semi-definite
        pred_factual_mc[:] = np.nan
        pred_counterfactual_mc[:] = np.nan

    factual_ci_lower = sigmoid(np.percentile(pred_factual_mc, (alpha/2)*100, axis=0))
    factual_ci_upper = sigmoid(np.percentile(pred_factual_mc, (1-alpha/2)*100, axis=0))
    cf_ci_lower = sigmoid(np.percentile(pred_counterfactual_mc, (alpha/2)*100, axis=0))
    cf_ci_upper = sigmoid(np.percentile(pred_counterfactual_mc, (1-alpha/2)*100, axis=0))

    return {
        "its_factual_mean": sigmoid(nls_model((t_array, Z_nls), *popt)),
        "its_factual_lower": factual_ci_lower,
        "its_factual_upper": factual_ci_upper,
        "its_counterfactual_mean": sigmoid(Bm @ popt[:K_spline]),
        "its_counterfactual_lower": cf_ci_lower,
        "its_counterfactual_upper": cf_ci_upper,
        "its_gamma_est": gamma_est,
        "its_gamma_lower": gamma_ci_lower,
        "its_gamma_upper": gamma_ci_upper,
        "its_lambda_est": lambda_est,
        "its_lambda_lower": lambda_ci_lower,
        "its_lambda_upper": lambda_ci_upper,
    }