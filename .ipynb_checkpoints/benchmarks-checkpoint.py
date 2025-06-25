# File: benchmarks.py

import numpy as np
import pandas as pd
from scipy.stats import norm, beta_dist
from scipy.special import logit, expit as sigmoid
from scipy.optimize import minimize

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

def calculate_its_with_poisson_mle(d_t, c_t, f_s, Bm, intervention_times_abs, intervention_signs, alpha=0.05, n_mc_ci=500):
    """
    Estimates CFR and parameters using an ITS model with a Poisson likelihood (MLE).

    Args:
        d_t, c_t, f_s (np.ndarray): Daily deaths, cases, and delay distribution.
        Bm (np.ndarray): The B-spline basis matrix for the baseline trend.
        intervention_times_abs (np.ndarray): Absolute day numbers for intervention starts.
        intervention_signs (np.ndarray): The signs of the intervention effects.
        alpha (float): Significance level for the CIs (e.g., 0.05 for 95% CI).
        n_mc_ci (int): Number of Monte Carlo samples for generating prediction CIs.

    Returns:
        dict: A comprehensive dictionary of all point estimates and CI bounds.
    """
    T, K_spline = Bm.shape
    K_interventions = len(intervention_times_abs)
    num_params = K_spline + 2 * K_interventions

    # Pre-calculate the Q matrix and Z matrix, which are fixed for the optimization
    Q_matrix = construct_Q_matrix(c_t, f_s, T)
    Z_its = np.zeros((T, K_interventions))
    t_array = np.arange(T)
    for k in range(K_interventions):
        Z_its[:, k] = np.maximum(0, t_array - intervention_times_abs[k])

    def negative_log_likelihood(params, Bm, Z, d_t, Q, intervention_signs):
        """The objective function to minimize (Poisson Negative Log-Likelihood)."""
        alphas = params[:K_spline]
        gammas = params[K_spline : K_spline + K_interventions]
        lambdas = params[K_spline + K_interventions:]

        betas = np.exp(gammas) * intervention_signs
        intervention_effect = np.sum((1 - np.exp(-lambdas * Z)) * betas, axis=1)
        baseline_trend = Bm @ alphas
        
        r_t_pred = sigmoid(baseline_trend + intervention_effect)
        mu_pred = np.maximum(Q @ r_t_pred, 1e-9) # Ensure mean is positive

        # Poisson negative log-likelihood: sum(mu - d*log(mu))
        # Add a small epsilon to d_t * log(mu_pred) for stability if mu_pred is zero
        log_likelihood = np.sum(d_t * np.log(mu_pred + 1e-9) - mu_pred)
        
        # We minimize the negative of the log-likelihood
        return -log_likelihood

    # Initial guesses for parameters
    p0 = np.zeros(num_params)
    if K_interventions > 0: p0[K_spline + K_interventions:] = 0.1 # Initial guess for lambdas

    # --- Fit the model using scipy.optimize.minimize ---
    result = minimize(
        negative_log_likelihood, 
        x0=p0, 
        args=(Bm, Z_its, d_t, Q_matrix, intervention_signs), 
        method='BFGS',
        options={'maxiter': 5000, 'disp': False}
    )
    
    popt = result.x if result.success else np.full(num_params, np.nan)
    # The inverse of the Hessian matrix approximates the covariance matrix
    pcov = result.hess_inv if hasattr(result, 'hess_inv') and isinstance(result.hess_inv, np.ndarray) else np.full((num_params, num_params), np.nan)

    # --- Calculate Parameter CIs ---
    perr = np.sqrt(np.diag(pcov)) if not np.any(np.isnan(pcov)) else np.full_like(popt, np.nan)
    z_crit = norm.ppf(1 - alpha / 2)
    
    gammas_est = popt[K_spline : K_spline + K_interventions]
    lambdas_est = popt[K_spline + K_interventions:]
    gammas_se, lambdas_se = perr[K_spline : K_spline + K_interventions], perr[K_spline + K_interventions:]
    
    # --- Calculate Prediction CIs via Monte Carlo ---
    pred_factual_mc = np.full((n_mc_ci, T), np.nan)
    pred_counterfactual_mc = np.full((n_mc_ci, T), np.nan)

    if not np.any(np.isnan(pcov)):
        try:
            param_samples = np.random.multivariate_normal(popt, pcov, size=n_mc_ci)
            for i in range(n_mc_ci):
                p_sample = param_samples[i, :]
                alphas_s, gammas_s, lambdas_s = p_sample[:K_spline], p_sample[K_spline:K_spline+K_interventions], p_sample[K_spline+K_interventions:]
                
                betas_s = np.exp(gammas_s) * intervention_signs
                intervention_effect_s = np.sum((1 - np.exp(-lambdas_s * Z_its)) * betas_s, axis=1)
                baseline_trend_s = Bm @ alphas_s
                
                pred_factual_mc[i, :] = sigmoid(baseline_trend_s + intervention_effect_s)
                pred_counterfactual_mc[i, :] = sigmoid(baseline_trend_s)
        except (np.linalg.LinAlgError, ValueError):
            print(f"Warning: NLS Covariance matrix was not valid for CI sampling.")

    # --- Final Point Estimates ---
    alphas_hat, gammas_hat, lambdas_hat = popt[:K_spline], popt[K_spline:K_spline+K_interventions], popt[K_spline+K_interventions:]
    betas_hat = np.exp(gammas_hat) * intervention_signs
    
    factual_mean = sigmoid(Bm @ alphas_hat + np.sum((1 - np.exp(-lambdas_hat * Z_its)) * betas_hat, axis=1))
    counterfactual_mean = sigmoid(Bm @ alphas_hat)
    
    return {
        "its_factual_mean": factual_mean,
        "its_factual_lower": np.percentile(pred_factual_mc, (alpha/2)*100, axis=0),
        "its_factual_upper": np.percentile(pred_factual_mc, (1-alpha/2)*100, axis=0),
        "its_counterfactual_mean": counterfactual_mean,
        "its_counterfactual_lower": np.percentile(pred_counterfactual_mc, (alpha/2)*100, axis=0),
        "its_counterfactual_upper": np.percentile(pred_counterfactual_mc, (1-alpha/2)*100, axis=0),
        "its_gamma_est": gammas_hat,
        "its_gamma_lower": gammas_hat - z_crit * gammas_se,
        "its_gamma_upper": gammas_hat + z_crit * gammas_se,
        "its_lambda_est": lambdas_hat,
        "its_lambda_lower": lambdas_hat - z_crit * lambdas_se,
        "its_lambda_upper": lambdas_hat + z_crit * lambdas_se,
    }