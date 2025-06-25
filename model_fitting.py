import numpy as np
import jax
from sklearn.linear_model import LinearRegression
from scipy.special import logit
import pandas as pd # For pd.isna check

from sampler import model as numpyro_model, sample as run_numpyro_sampler 
import config

def get_ols_initial_values(Bm_obs, Z_input_obs, dt_obs, ct_obs, K_spline_obs, num_interventions_K):
    """
    Calculates initial values for MCMC sampler using a simple OLS regression.
    
    Args:
        Bm_obs (np.ndarray): The B-spline basis matrix.
        Z_input_obs (np.ndarray): The intervention time-lag matrix.
        dt_obs (np.ndarray): Observed daily deaths.
        ct_obs (np.ndarray): Observed daily cases.
        K_spline_obs (int): Number of B-spline knots.
        num_interventions_K (int): Number of interventions.

    Returns:
        dict: A dictionary of initial values for the sampler.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        naive_cCFR = np.cumsum(dt_obs) / (np.cumsum(ct_obs) + 1e-6)
        naive_cCFR = np.clip(naive_cCFR, 1e-6, 1 - 1e-6)
        logit_cCFR_target = logit(naive_cCFR)
        if np.any(np.isnan(logit_cCFR_target)) or np.any(np.isinf(logit_cCFR_target)):
             median_valid_logit = np.median(logit_cCFR_target[~np.isnan(logit_cCFR_target) & ~np.isinf(logit_cCFR_target)])
             if pd.isna(median_valid_logit): median_valid_logit = logit(0.01)
             logit_cCFR_target = np.nan_to_num(logit_cCFR_target, nan=median_valid_logit, posinf=logit(0.99), neginf=logit(0.01))

    combined_basis = Bm_obs
    if num_interventions_K > 0:
        Z_input_obs_2d = Z_input_obs if Z_input_obs.ndim == 2 else Z_input_obs.reshape(-1, 1)
        if Z_input_obs_2d.shape[1] > 0:
            combined_basis = np.concatenate([Bm_obs, Z_input_obs_2d], axis=1)
    
    try:
        ols = LinearRegression(fit_intercept=False).fit(combined_basis, logit_cCFR_target)
        alpha_beta_init = ols.coef_
    except Exception: 
        alpha_beta_init = np.zeros(combined_basis.shape[1])

    alpha_init = alpha_beta_init[:K_spline_obs]
    
    # Initialize the dictionary
    data_init = {"alpha": alpha_init}

# --- Assemble the final initial values dictionary ---
    data_init = {
        "alpha": alpha_init,
        "lambda_alpha": 1.0,
        "phi": 0.1
    }
    
    if num_interventions_K > 0:
        beta_init_signed = alpha_beta_init[K_spline_obs:] 
        beta_abs_init = np.abs(beta_init_signed)
        beta_abs_init = np.maximum(beta_abs_init, 1e-3) 
        lambda_init_rates = np.ones(num_interventions_K) * 0.1 
        gamma_init = np.log(beta_abs_init)
        
        data_init["gamma"] = gamma_init
        # data_init["beta_abs"] = beta_abs_init
        data_init["lambda"] = lambda_init_rates
    
    # Add initial values for penalty parameters if they are in the model
    # These may not be strictly necessary for NUTS but can sometimes help.
    if "lambda_alpha" in run_numpyro_sampler.__code__.co_varnames: data_init["lambda_alpha"] = 1.0
    if "lambda_beta" in run_numpyro_sampler.__code__.co_varnames: data_init["lambda_beta"] = 1.0
    if "phi" in run_numpyro_sampler.__code__.co_varnames: data_init["phi"] = 0.1
        
    return data_init

# def get_multistage_initial_values(Bm_obs, Z_input_obs, dt_obs, ct_obs, K_spline_obs, num_interventions_K, intervention_times_abs):
#     """
#     Calculates improved initial values using a multi-stage, difference-in-differences
#     style approach to better separate baseline from intervention effects.

#     Args:
#         Bm_obs (np.ndarray): The B-spline basis matrix.
#         Z_input_obs (np.ndarray): The intervention time-lag matrix.
#         dt_obs, ct_obs (np.ndarray): Daily deaths and cases.
#         K_spline_obs (int): Number of B-spline knots.
#         num_interventions_K (int): Number of interventions.
#         intervention_times_abs (np.ndarray): Array of absolute start times for interventions.

#     Returns:
#         dict: A dictionary of well-informed initial values for the sampler.
#     """
#     with np.errstate(divide='ignore', invalid='ignore'):
#         naive_cCFR = np.cumsum(dt_obs) / (np.cumsum(ct_obs) + 1e-6)
#         naive_cCFR = np.clip(naive_cCFR, 1e-6, 1 - 1e-6)
#         logit_cCFR_target = logit(naive_cCFR)
#         logit_cCFR_target = np.nan_to_num(logit_cCFR_target, nan=np.median(logit_cCFR_target[~np.isnan(logit_cCFR_target)]))

#     # --- Stage 1: Estimate baseline (alpha) using only pre-intervention data ---
#     # Find the start time of the first intervention
#     first_intervention_t = int(intervention_times_abs[0]) if num_interventions_K > 0 else len(dt_obs)
    
#     # Use data only up to this point to estimate the baseline spline
#     pre_int_indices = np.arange(first_intervention_t)
#     if len(pre_int_indices) > K_spline_obs: # Ensure enough data points to fit OLS
#         ols_alpha = LinearRegression(fit_intercept=False).fit(
#             Bm_obs[pre_int_indices, :], 
#             logit_cCFR_target[pre_int_indices]
#         )
#         alpha_init = ols_alpha.coef_
#     else: # Fallback if not enough pre-intervention data
#         alpha_init = np.zeros(K_spline_obs)

#     # --- Stage 2: Estimate intervention effects (beta) using the residuals ---
#     if num_interventions_K > 0:
#         # Calculate the estimated baseline trend for the whole period
#         baseline_trend = Bm_obs @ alpha_init
#         # The residuals are what's left to be explained by interventions
#         residuals = logit_cCFR_target - baseline_trend
        
#         # Fit the intervention matrix on the residuals for post-intervention periods
#         ols_beta = LinearRegression(fit_intercept=False).fit(Z_input_obs, residuals)
#         beta_init_signed = ols_beta.coef_
#         beta_abs_init = np.abs(beta_init_signed)
#         beta_abs_init = np.maximum(beta_abs_init, 1e-3)
#         lambda_init_rates = np.ones(num_interventions_K) * 0.1 # Start with a reasonable rate
#     else:
#         beta_abs_init = np.array([])
#         lambda_init_rates = np.array([])
        
#     # --- Assemble the final initial values dictionary ---
#     data_init = {
#         "alpha": alpha_init,
#         "lambda_alpha": 1.0,
#         "lambda_beta": 1.0,
#         "phi": 0.1
#     }
#     if num_interventions_K > 0:
#         data_init["beta_abs"] = beta_abs_init
#         data_init["lambda"] = lambda_init_rates
        
#     return data_init

def fit_proposed_model(sim_data, jax_prng_key):
    """
    Prepares data, performs orthogonalization, and fits the proposed Bayesian model.

    Args:
        sim_data (dict): The dictionary of simulated data.
        jax_prng_key (jax.random.PRNGKey): The random key for MCMC.

    Returns:
        tuple: (posterior_samples, mcmc_object).
    """
    Bm = sim_data["Bm_true"]
    Z_orig = sim_data["Z_input_true"]

    # --- Orthogonalization Step ---
    # We remove the part of Z that can be explained by the B-spline basis Bm.
    try:
        # Calculate (Bm.T @ Bm)^-1 @ Bm.T @ Z_orig, which are the coefficients of Z projected onto Bm
        # Using pseudo-inverse (pinv) for numerical stability
        B_inv_B_T = np.linalg.pinv(Bm) @ Z_orig
        
        # The new Z is the original Z minus its projection onto the B-spline space
        Z_for_sampler = Z_orig - (Bm @ B_inv_B_T)
    except np.linalg.LinAlgError:
        print("Warning: Could not perform orthogonalization due to singular matrix. Using original Z.")
        Z_for_sampler = Z_orig

    if sim_data["num_interventions_true_K"] == 0:
        Z_for_sampler = np.empty((sim_data["N_obs"], 0))

    # The data for the sampler now uses the orthogonalized Z matrix
    data_for_sampler = {
        'N': sim_data["N_obs"], 'K': sim_data["K_spline_obs"], 'dt': sim_data["d_t"],
        'fc_mat': sim_data["Q_true"], 'Bm': Bm, 
        'Z': Z_for_sampler, 
        'beta_signs': sim_data["beta_signs_true"]
    }
    
    # Initialization can use the original Z matrix for simplicity
    data_init = get_ols_initial_values(
        sim_data["Bm_true"], Z_for_sampler, sim_data["d_t"], sim_data["c_t"],
        sim_data["K_spline_obs"], sim_data["num_interventions_true_K"]
    )
            
    posterior_samples, mcmc_obj = run_numpyro_sampler(
        numpyro_model, data_for_sampler, data_init=data_init,
        method='NUTS', num_warmup=config.NUM_WARMUP,
        num_samples=config.NUM_SAMPLES, num_chains=config.NUM_CHAINS,
        rng_key=jax_prng_key 
    )
    return posterior_samples, mcmc_obj

# def fit_proposed_model(sim_data, jax_prng_key):
#     """
#     Prepares data and fits the proposed Bayesian model using NumPyro.

#     Args:
#         sim_data (dict): The dictionary of simulated data.
#         jax_prng_key (jax.random.PRNGKey): The random key for MCMC.

#     Returns:
#         tuple: (posterior_samples, mcmc_object).
#     """
#     Z_for_sampler = sim_data["Z_input_true"]
#     if sim_data["num_interventions_true_K"] == 0:
#         Z_for_sampler = np.empty((sim_data["N_obs"], 0))

#     data_for_sampler = {
#         'N': sim_data["N_obs"], 'K': sim_data["K_spline_obs"], 'dt': sim_data["d_t"],
#         'fc_mat': sim_data["Q_true"], 'Bm': sim_data["Bm_true"], 
#         'Z': Z_for_sampler, 
#         'beta_signs': sim_data["beta_signs_true"]
#     }
    
#     data_init = get_ols_initial_values(
#         sim_data["Bm_true"], Z_for_sampler, sim_data["d_t"], sim_data["c_t"],
#         sim_data["K_spline_obs"], sim_data["num_interventions_true_K"]
#     )

#     # data_init = get_multistage_initial_values(
#     #     sim_data["Bm_true"],
#     #     Z_for_sampler,
#     #     sim_data["d_t"],
#     #     sim_data["c_t"],
#     #     sim_data["K_spline_obs"],
#     #     sim_data["num_interventions_true_K"],
#     #     sim_data["true_intervention_times_0_abs"] 
#     # )
            
#     posterior_samples, mcmc_obj = run_numpyro_sampler(
#         numpyro_model, data_for_sampler, data_init=data_init,
#         method='NUTS', num_warmup=config.NUM_WARMUP,
#         num_samples=config.NUM_SAMPLES, num_chains=config.NUM_CHAINS,
#         rng_key=jax_prng_key 
#     )
#     return posterior_samples, mcmc_obj