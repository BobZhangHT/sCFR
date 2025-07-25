import numpy as np
import jax
from sklearn.linear_model import LinearRegression
from scipy.special import logit
import pandas as pd
from sampler import model as numpyro_model, sample as run_numpyro_sampler 
import config

def get_ols_initial_values(Bm_obs, Z_input_obs, dt_obs, ct_obs, K_spline_obs, num_interventions_K, use_constraint):
    """
    Calculates initial values for the MCMC sampler using Ordinary Least Squares (OLS).

    Args:
        Bm_obs (numpy.ndarray): The observed B-spline basis matrix.
        Z_input_obs (numpy.ndarray): The observed intervention input matrix.
        dt_obs (numpy.ndarray): The observed daily death counts.
        ct_obs (numpy.ndarray): The observed daily case counts.
        K_spline_obs (int): The number of B-spline knots.
        num_interventions_K (int): The number of interventions.
        use_constraint (bool): Flag for using sum-to-zero constraint.

    Returns:
        dict: A dictionary of initial values for the sampler.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        naive_cCFR = np.cumsum(dt_obs) / (np.cumsum(ct_obs) + 1e-6)
        naive_cCFR = np.clip(naive_cCFR, 1e-6, 1 - 1e-6)
        logit_cCFR_target = logit(naive_cCFR)
        if np.any(np.isnan(logit_cCFR_target)):
             median_valid_logit = np.median(logit_cCFR_target[~np.isnan(logit_cCFR_target)])
             if pd.isna(median_valid_logit): median_valid_logit = logit(0.01)
             logit_cCFR_target = np.nan_to_num(logit_cCFR_target, nan=median_valid_logit)

    if use_constraint:
        C_contrast = np.vstack([np.eye(K_spline_obs - 1), -np.ones((1, K_spline_obs - 1))])
        Bm_con = Bm_obs @ C_contrast
        combined_basis = Bm_con
    else:
        combined_basis = Bm_obs

    if num_interventions_K > 0 and Z_input_obs.shape[1] > 0:
        combined_basis = np.concatenate([combined_basis, Z_input_obs], axis=1)
        
    try:
        ols = LinearRegression(fit_intercept=False).fit(combined_basis, logit_cCFR_target)
        all_coefs_init = ols.coef_
    except Exception: 
        all_coefs_init = np.zeros(combined_basis.shape[1])

    data_init = {"lambda_alpha": 1.0}

    if use_constraint:
        alpha_tilde_init = all_coefs_init[:(K_spline_obs - 1)]
        data_init["alpha_tilde"] = alpha_tilde_init
    else:
        alpha_init = all_coefs_init[:K_spline_obs]
        data_init["alpha"] = alpha_init
        
    if num_interventions_K > 0:
        beta_init_signed = all_coefs_init[-num_interventions_K:]
        beta_abs_init = np.maximum(np.abs(beta_init_signed), 1e-3) 
        data_init["beta_abs"] = beta_abs_init
        data_init["lambda"] = np.ones(num_interventions_K) * 0.1

    return data_init

def fit_proposed_model(sim_data, jax_prng_key):
    """
    Prepares data and fits the proposed Bayesian sCFR model.

    Args:
        sim_data (dict): A dictionary containing the simulated data.
        jax_prng_key (jax.random.PRNGKey): The random key for JAX.

    Returns:
        tuple: A tuple containing the posterior samples and the MCMC object.
    """
    Z_for_sampler = sim_data["Z_input_true"]
    if sim_data["num_interventions_true_K"] == 0:
        Z_for_sampler = np.empty((sim_data["N_obs"], 0))

    data_for_sampler = {
        'N': sim_data["N_obs"], 
        'K': sim_data["K_spline_obs"], 
        'dt': sim_data["d_t"],
        'fc_mat': sim_data["Q_true"], 
        'Bm': sim_data["Bm_true"], 
        'Z': Z_for_sampler, 
        'beta_signs': sim_data["beta_signs_true"],
        'use_constraint': config.USE_CONSTRAINT
    }
    
    data_init = get_ols_initial_values(
        sim_data["Bm_true"],
        Z_for_sampler,
        sim_data["d_t"], 
        sim_data["c_t"],
        sim_data["K_spline_obs"],
        sim_data["num_interventions_true_K"],
        config.USE_CONSTRAINT
    )
            
    posterior_samples, mcmc_obj = run_numpyro_sampler(
        numpyro_model, data_for_sampler, data_init=data_init,
        method='NUTS', num_warmup=config.NUM_WARMUP,
        num_samples=config.NUM_SAMPLES, num_chains=config.NUM_CHAINS,
        rng_key=jax_prng_key 
    )
    return posterior_samples, mcmc_obj