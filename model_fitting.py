# File: model_fitting.py
# Description: This version prepares data and initial values for the unconstrained
#              sCFR Bayesian model.

import numpy as np
import jax
from sklearn.linear_model import LinearRegression
from scipy.special import logit
import pandas as pd

from sampler import model as numpyro_model, sample as run_numpyro_sampler 
import config

def get_ols_initial_values(Bm_obs, Z_input_obs, dt_obs, ct_obs, K_spline_obs, num_interventions_K):
    """
    Calculates initial values for the MCMC sampler using a single OLS regression.
    This version provides an initial value directly for `beta_abs`.
    
    Args:
        (Arguments are the same as before)

    Returns:
        dict: A dictionary of initial values for the sampler.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        # Use cumulative CFR for a smoother target variable for OLS
        naive_cCFR = np.cumsum(dt_obs) / (np.cumsum(ct_obs) + 1e-6)
        naive_cCFR = np.clip(naive_cCFR, 1e-6, 1 - 1e-6)
        logit_cCFR_target = logit(naive_cCFR)
        if np.any(np.isnan(logit_cCFR_target)):
             median_valid_logit = np.median(logit_cCFR_target[~np.isnan(logit_cCFR_target)])
             if pd.isna(median_valid_logit): median_valid_logit = logit(0.01)
             logit_cCFR_target = np.nan_to_num(logit_cCFR_target, nan=median_valid_logit)

    combined_basis = Bm_obs
    if num_interventions_K > 0 and Z_input_obs.shape[1] > 0:
        combined_basis = np.concatenate([Bm_obs, Z_input_obs], axis=1)
    
    try:
        ols = LinearRegression(fit_intercept=False).fit(combined_basis, logit_cCFR_target)
        alpha_beta_init = ols.coef_
    except Exception: 
        alpha_beta_init = np.zeros(combined_basis.shape[1])

    alpha_init = alpha_beta_init[:K_spline_obs]
    
    # --- Assemble the final initial values dictionary ---
    data_init = {
        "alpha": alpha_init,
        "lambda_alpha": 1.0,
        "phi": 0.1
    }
    
    if num_interventions_K > 0:
        beta_init_signed = alpha_beta_init[K_spline_obs:]
        
        beta_abs_init = np.maximum(np.abs(beta_init_signed), 1e-3) 
        
        data_init["beta_abs"] = beta_abs_init
        data_init["lambda"] = np.ones(num_interventions_K) * 0.1
        
    return data_init

def fit_proposed_model(sim_data, jax_prng_key):
    """
    Prepares data and fits the proposed Bayesian sCFR model using the
    standard (unconstrained) B-spline basis.
    """
    Z_for_sampler = sim_data["Z_input_true"]
    if sim_data["num_interventions_true_K"] == 0:
        Z_for_sampler = np.empty((sim_data["N_obs"], 0))

    # Pass the original basis and number of knots to the sampler
    data_for_sampler = {
        'N': sim_data["N_obs"], 
        'K': sim_data["K_spline_obs"], 
        'dt': sim_data["d_t"],
        'fc_mat': sim_data["Q_true"], 
        'Bm': sim_data["Bm_true"], 
        'Z': Z_for_sampler, 
        'beta_signs': sim_data["beta_signs_true"]
    }
    
    data_init = get_ols_initial_values(
        sim_data["Bm_true"],
        Z_for_sampler,
        sim_data["d_t"], 
        sim_data["c_t"],
        sim_data["K_spline_obs"],
        sim_data["num_interventions_true_K"]
    )
            
    posterior_samples, mcmc_obj = run_numpyro_sampler(
        numpyro_model, data_for_sampler, data_init=data_init,
        method='NUTS', num_warmup=config.NUM_WARMUP,
        num_samples=config.NUM_SAMPLES, num_chains=config.NUM_CHAINS,
        rng_key=jax_prng_key 
    )
    return posterior_samples, mcmc_obj