"""
Data generation module for sCFR simulation study.

This module provides functions for generating synthetic epidemic data including:
- Case counts with various temporal patterns
- Onset-to-death delay distributions
- B-spline basis matrices
- Intervention effect matrices
- True CFR time series (factual and counterfactual)
"""

import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.special import expit as sigmoid, logit
from scipy.interpolate import BSpline
import config


def generate_deterministic_baseline_case_counts(T_sim, T_analysis, function_type, params, min_floor_cases):
    """Generate deterministic baseline case counts."""
    baseline_c_t = np.zeros(T_sim)
    
    if function_type == 'v_shape':
        peak_t_abs = params["v_shape_peak_time_factor"] * T_analysis 
        for t_idx in range(T_sim):
            baseline_c_t[t_idx] = params["v_shape_max_cases"] - params["v_shape_slope"] * abs(peak_t_abs - t_idx)
    elif function_type == 'constant':
        baseline_c_t = np.full(T_sim, params["constant_cases"])
    else:
        raise ValueError(f"Unknown C_T_FUNCTION_TYPE '{function_type}'.")
            
    baseline_c_t = np.maximum(min_floor_cases, baseline_c_t)
    return np.round(baseline_c_t).astype(int)


def generate_delay_distribution(T_delay_max, mean_delay, shape_delay):
    """Generate onset-to-death delay distribution (PMF) using Gamma distribution."""
    scale_delay = mean_delay / shape_delay
    s_array = np.arange(T_delay_max + 1) 
    f_s_unnormalized = np.diff(gamma_dist.cdf(s_array, a=shape_delay, scale=scale_delay))
    return f_s_unnormalized[:T_delay_max]


def construct_Q_matrix(c_t, f_s, T_sim):
    """Construct convolution matrix Q for expected deaths calculation."""
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

    
def generate_bspline_basis(T_sim, num_knots_J, order):
    """Generate B-spline basis matrix."""
    t_array_scaled = np.linspace(0, 1, T_sim)
    
    if num_knots_J <= order - 2:
        interior_knots = np.array([])
    else:
        interior_knots = np.linspace(0, 1, num_knots_J - (order - 2))[1:-1]

    boundary_knots = np.array([0, 1])
    knots = np.sort(np.concatenate([
        np.repeat(boundary_knots[0], order), interior_knots, np.repeat(boundary_knots[1], order)
    ]))
    
    Bm = BSpline(knots, np.eye(num_knots_J), k=order - 1)(t_array_scaled)
    return Bm


def generate_intervention_input_matrix(T_array_unscaled_sim, intervention_times_true_abs, num_interventions_K):
    """Generate step function intervention matrix Z where Z[t,k] = I(t >= t_k)."""
    N_sim = len(T_array_unscaled_sim)
    if num_interventions_K == 0:
        return np.empty((N_sim, 0))
    
    denom = max(N_sim - 1, 1)
    t_broadcast_unscaled = (T_array_unscaled_sim / denom).reshape(-1, 1)
    ti_broadcast_unscaled = (np.array(intervention_times_true_abs) / denom).reshape(1, num_interventions_K)
    
    mask = t_broadcast_unscaled >= ti_broadcast_unscaled
    Z_input = mask.astype(np.float32)
    return Z_input


def generate_intervention_hinge_matrix(Z_step):
    """Generate hinge basis matrix from step indicators: h(t) = (t - t_k)_+ normalized."""
    if Z_step.size == 0:
        return Z_step
    T = Z_step.shape[0]
    denom = max(T - 1, 1)
    return (np.cumsum(Z_step, axis=0) - Z_step) / denom

    
def generate_random_effects_eta(T_sim, sigma_eta_true, rng_numpy):
    """Generate i.i.d. Gaussian random effects."""
    if sigma_eta_true is None or sigma_eta_true == 0:
        return np.zeros(T_sim)
    return rng_numpy.normal(loc=0, scale=sigma_eta_true, size=T_sim)


def generate_baseline_cfr_zeta(T_sim, T_analysis, cfr_type_code, cfr_params):
    """Generate true systematic baseline logit-CFR."""
    t_array_sim = np.arange(T_sim)
    t_prime_for_pattern = t_array_sim / (T_analysis - 1 if T_analysis > 1 else 1)

    if cfr_type_code == "C1":
        zeta_0_t = np.full(T_sim, logit(cfr_params["cfr_const"]))
    elif cfr_type_code == "C2":
        cfr_t = cfr_params["cfr_start"] - (cfr_params["cfr_start"] - cfr_params["cfr_end"]) * t_prime_for_pattern
        if T_sim > T_analysis:
            cfr_t[T_analysis:] = cfr_params["cfr_end"]
        zeta_0_t = logit(np.clip(cfr_t, 1e-7, 1-1e-7))
    elif cfr_type_code == "C3":
        cfr_t = cfr_params["cfr_mean"] + cfr_params["amp"] * np.sin(2 * np.pi * cfr_params["freq"] * t_prime_for_pattern)
        zeta_0_t = logit(np.clip(cfr_t, 1e-7, 1-1e-7))
    elif cfr_type_code == "C4":
        peak_t = cfr_params.get('peak_t', cfr_params.get('peak_t_factor', 0) * T_analysis)
        peak_w = cfr_params.get('peak_w', cfr_params.get('peak_w_factor', 0) * T_analysis)
        cfr_t = cfr_params["cfr_base"] + cfr_params["peak_h"] * \
                np.exp(-(t_array_sim - peak_t)**2 / (2 * peak_w**2))
        zeta_0_t = logit(np.clip(cfr_t, 1e-7, 1-1e-7))
    else:
        raise ValueError(f"Unknown CFR type: {cfr_type_code}")
    return zeta_0_t


def simulate_scenario_data(scenario_config_dict, run_seed):
    """Simulate complete dataset for a single scenario."""
    rng_numpy = np.random.default_rng(run_seed)
    T_sim = config.T_SERIES_LENGTH_SIM
    T_analyze = config.T_ANALYSIS_LENGTH
    t_array_unscaled_sim = np.arange(T_sim)
    
    # Generate case counts
    baseline_ct_params = {
        "v_shape_max_cases": config.C_T_VSHAPE_MAX_CASES,
        "v_shape_slope": config.C_T_VSHAPE_SLOPE,
        "v_shape_peak_time_factor": config.C_T_VSHAPE_PEAK_TIME_FACTOR,
        "constant_cases": config.C_T_CONSTANT_CASES 
    }
    c_t_final = generate_deterministic_baseline_case_counts(
        T_sim, T_analyze, config.C_T_FUNCTION_TYPE, baseline_ct_params, config.MIN_DRAWN_CASES
    )

    # Generate B-spline basis
    Bm_true = generate_bspline_basis(T_sim, config.N_SPLINE_KNOTS_J, config.SPLINE_ORDER)
    
    # Extract intervention parameters
    num_interventions_K_true = scenario_config_dict["num_interventions_K_true"]
    true_beta_abs_0 = scenario_config_dict["true_beta_abs_0"]
    true_beta_slope_abs_0 = scenario_config_dict["true_beta_slope_abs_0"]
    true_intervention_times_0_abs = scenario_config_dict["true_intervention_times_0"]
    true_beta_signs_0 = scenario_config_dict["true_beta_signs_0"]
    true_beta_slope_signs_0 = scenario_config_dict["true_beta_slope_signs_0"]
    
    # Generate intervention matrices
    Z_step_true = generate_intervention_input_matrix(
        t_array_unscaled_sim, true_intervention_times_0_abs, num_interventions_K_true
    )
    Z_hinge_true = generate_intervention_hinge_matrix(Z_step_true)
    
    # Calculate intervention effect
    intervention_effect_on_logit_cfr = np.zeros(T_sim)
    true_beta_0_vector = np.array([])
    true_beta_slope_0_vector = np.array([])
    if num_interventions_K_true > 0:
        true_beta_0_vector = true_beta_abs_0 * true_beta_signs_0
        true_beta_slope_0_vector = true_beta_slope_abs_0 * true_beta_slope_signs_0
        intervention_effect_on_logit_cfr = (
            np.dot(Z_step_true, true_beta_0_vector) +
            np.dot(Z_hinge_true, true_beta_slope_0_vector)
        )
    
    # Generate delay distribution and Q matrix
    f_s = generate_delay_distribution(config.F_DELAY_MAX, config.F_MEAN, config.F_SHAPE)
    Q_true = construct_Q_matrix(c_t_final, f_s, T_sim)
    
    # Generate CFR components
    zeta_0_t = generate_baseline_cfr_zeta(T_sim, T_analyze, scenario_config_dict["cfr_type_code"], scenario_config_dict["cfr_params"])
    eta_0_t = generate_random_effects_eta(T_sim, scenario_config_dict["sigma_delta_true"], rng_numpy)
    
    # Compute factual and counterfactual CFR
    true_logit_rcf_0_t = zeta_0_t + eta_0_t
    true_rcf_0_t = sigmoid(true_logit_rcf_0_t)
    true_logit_r_0_t = zeta_0_t + intervention_effect_on_logit_cfr + eta_0_t
    true_r_0_t = sigmoid(true_logit_r_0_t)
    
    # Generate deaths from Poisson distribution
    mu_0_t = Q_true @ true_r_0_t
    mu_0_t_clipped = np.maximum(1e-9, mu_0_t)
    d_t = rng_numpy.poisson(mu_0_t_clipped)
    
    return {
        "scenario_id": scenario_config_dict["id"], 
        "c_t": c_t_final, 
        "d_t": d_t, 
        "f_s_true": f_s,
        "Q_true": Q_true, 
        "Bm_true": Bm_true,
        "Z_input_true": Z_step_true,
        "Z_hinge_true": Z_hinge_true,
        "beta_signs_true": true_beta_signs_0, 
        "N_obs": T_sim, 
        "K_spline_obs": config.N_SPLINE_KNOTS_J,
        "num_interventions_true_K": num_interventions_K_true,
        "true_beta_abs_0": true_beta_abs_0, 
        "true_beta_slope_abs_0": true_beta_slope_abs_0,
        "true_lambda_0": np.zeros_like(true_beta_abs_0),
        "true_beta_0": true_beta_0_vector,
        "true_beta_slope_0": true_beta_slope_0_vector,
        "true_r_0_t": true_r_0_t, 
        "true_rcf_0_t": true_rcf_0_t,
        "true_zeta_0_t": zeta_0_t,
        "true_eta_0_t": eta_0_t,
        "true_intervention_times_0_abs": true_intervention_times_0_abs,
        "true_beta_slope_signs_0": true_beta_slope_signs_0,
        "run_seed": run_seed
    }
