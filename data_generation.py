# File: data_generation.py
# Description: Functions for generating simulated data for each scenario.

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist, norm, poisson
from scipy.special import expit as sigmoid, logit
from scipy.interpolate import BSpline
from sklearn.preprocessing import MinMaxScaler

import config # Import parameters from config.py

def generate_deterministic_baseline_case_counts(T_sim, T_analysis, function_type, params, min_floor_cases):
    """
    Generates a deterministic baseline for case counts based on a simple function.
    """
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
    """Generates onset-to-death probability mass function f_s."""
    scale_delay = mean_delay / shape_delay
    s_array = np.arange(T_delay_max + 1) 
    f_s_unnormalized = np.diff(gamma_dist.cdf(s_array, a=shape_delay, scale=scale_delay))
    return f_s_unnormalized[:T_delay_max]

def construct_Q_matrix(c_t, f_s, T_sim):
    """Constructs the convolution matrix Q from daily cases and the delay distribution."""
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
    
def generate_bspline_basis(T_sim, num_knots_J, order):
    """Generates B-spline basis matrix Bm."""
    t_array = np.arange(1, T_sim + 1)
    # Scaled time is only used for placing knots, not for the intervention matrix
    t_array_scaled = np.linspace(0, 1, T_sim)
    
    if num_knots_J <= order -2 : # Not enough knots for interior ones
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
    """
    CORRECTED: Generates the Z_input matrix where Z_input[t,k] = (t - t_k)_+
               This now uses the ORIGINAL time scale for interpretability.
    """
    N_sim = len(T_array_unscaled_sim)
    if num_interventions_K == 0: return np.empty((N_sim, 0))
    
    Z_input = np.zeros((N_sim, num_interventions_K))
    t_broadcast_unscaled = T_array_unscaled_sim.reshape(-1, 1) 
    ti_broadcast_unscaled = np.array(intervention_times_true_abs).reshape(1, num_interventions_K)
    
    mask = t_broadcast_unscaled >= ti_broadcast_unscaled
    Z_input = (t_broadcast_unscaled - ti_broadcast_unscaled) * mask.astype(np.float32)
    return Z_input
    
def generate_random_effects_eta(T_sim, sigma_eta_true, rng_numpy):
    """Generates i.i.d. Gaussian random effects eta_0_t."""
    if sigma_eta_true is None or sigma_eta_true == 0:
        return np.zeros(T_sim)
    return rng_numpy.normal(loc=0, scale=sigma_eta_true, size=T_sim)

def generate_baseline_cfr_zeta(T_sim, T_analysis, cfr_type_code, cfr_params):
    """Generates the true systematic baseline logit-CFR zeta_0_t."""
    t_array_sim = np.arange(T_sim)
    t_prime_for_pattern = t_array_sim / (T_analysis - 1 if T_analysis > 1 else 1)

    if cfr_type_code == "C1":
        zeta_0_t = np.full(T_sim, logit(cfr_params["cfr_const"]))
    elif cfr_type_code == "C2":
        cfr_t = cfr_params["cfr_start"] - (cfr_params["cfr_start"] - cfr_params["cfr_end"]) * t_prime_for_pattern
        if T_sim > T_analysis: cfr_t[T_analysis:] = cfr_params["cfr_end"] 
        zeta_0_t = logit(np.clip(cfr_t, 1e-7, 1-1e-7))
    elif cfr_type_code == "C3":
        cfr_t = cfr_params["cfr_mean"] + cfr_params["amp"] * np.sin(2 * np.pi * cfr_params["freq"] * t_prime_for_pattern)
        zeta_0_t = logit(np.clip(cfr_t, 1e-7, 1-1e-7))
    elif cfr_type_code == "C4":
        peak_t = cfr_params.get('peak_t', cfr_params['peak_t_factor'] * T_analysis)
        peak_w = cfr_params.get('peak_w', cfr_params['peak_w_factor'] * T_analysis)
        cfr_t = cfr_params["cfr_base"] + cfr_params["peak_h"] * \
                np.exp(-(t_array_sim - peak_t)**2 / (2 * peak_w**2))
        zeta_0_t = logit(np.clip(cfr_t, 1e-7, 1-1e-7))
    else: raise ValueError(f"Unknown CFR type: {cfr_type_code}")
    return zeta_0_t

def simulate_scenario_data(scenario_config_dict, run_seed):
    """
    Simulates a complete dataset for a single scenario.
    CORRECTED: This version strictly follows the PDF model.
    """
    rng_numpy = np.random.default_rng(run_seed)
    T_sim = config.T_SERIES_LENGTH_SIM
    T_analyze = config.T_ANALYSIS_LENGTH
    t_array_unscaled_sim = np.arange(T_sim)
    
    baseline_ct_params = {
        "v_shape_max_cases": config.C_T_VSHAPE_MAX_CASES,
        "v_shape_slope": config.C_T_VSHAPE_SLOPE,
        "v_shape_peak_time_factor": config.C_T_VSHAPE_PEAK_TIME_FACTOR,
        "constant_cases": config.C_T_CONSTANT_CASES 
    }
    # CORRECTED: Case counts are generated independently of the intervention effect on CFR.
    c_t_final = generate_deterministic_baseline_case_counts(
        T_sim, T_analyze, config.C_T_FUNCTION_TYPE, baseline_ct_params, config.MIN_DRAWN_CASES
    )

    # Generate basis and intervention matrices
    Bm_true = generate_bspline_basis(T_sim, config.N_SPLINE_KNOTS_J, config.SPLINE_ORDER)
    
    num_interventions_K_true = scenario_config_dict["num_interventions_K_true"]
    true_beta_abs_0 = scenario_config_dict["true_beta_abs_0"]
    true_lambda_0 = scenario_config_dict["true_lambda_0"]
    true_intervention_times_0_abs = scenario_config_dict["true_intervention_times_0"]
    true_beta_signs_0 = scenario_config_dict["true_beta_signs_0"]
    
    # CORRECTED: Generate Z matrix using unscaled time
    Z_input_true = generate_intervention_input_matrix(
        t_array_unscaled_sim, true_intervention_times_0_abs, num_interventions_K_true
    )
    
    intervention_effect_on_logit_cfr = np.zeros(T_sim)
    true_beta_0_vector = np.array([])
    if num_interventions_K_true > 0:
        true_beta_0_vector = true_beta_abs_0 * true_beta_signs_0
        lag_matrix_true = (1 - np.exp(-true_lambda_0 * Z_input_true))
        intervention_effect_on_logit_cfr = np.dot(lag_matrix_true, true_beta_0_vector)
    
    f_s = generate_delay_distribution(config.F_DELAY_MAX, config.F_MEAN, config.F_SHAPE)
    Q_true = construct_Q_matrix(c_t_final, f_s, T_sim)
    
    zeta_0_t = generate_baseline_cfr_zeta(T_sim, T_analyze, scenario_config_dict["cfr_type_code"], scenario_config_dict["cfr_params"])
    eta_0_t = generate_random_effects_eta(T_sim, config.ETA_SIGMA_TRUE, rng_numpy)
    
    true_logit_rcf_0_t = zeta_0_t + eta_0_t
    true_rcf_0_t = sigmoid(true_logit_rcf_0_t)
    true_logit_r_0_t = zeta_0_t + intervention_effect_on_logit_cfr + eta_0_t
    true_r_0_t = sigmoid(true_logit_r_0_t)
    
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
        "Z_input_true": Z_input_true,
        "beta_signs_true": true_beta_signs_0, 
        "N_obs": T_sim, 
        "K_spline_obs": config.N_SPLINE_KNOTS_J,
        "num_interventions_true_K": num_interventions_K_true,
        "true_beta_abs_0": true_beta_abs_0, 
        "true_lambda_0": true_lambda_0,
        "true_beta_0": true_beta_0_vector,
        "true_r_0_t": true_r_0_t, 
        "true_rcf_0_t": true_rcf_0_t,
        "true_intervention_times_0_abs": true_intervention_times_0_abs, 
        "run_seed": run_seed
    }

# def generate_deterministic_baseline_case_counts(T_sim, T_analysis, function_type, params, min_floor_cases):
#     """
#     Generates a deterministic baseline for case counts based on a simple function.
#     """
#     baseline_c_t = np.zeros(T_sim)
    
#     if function_type == 'v_shape':
#         peak_t_abs = params["v_shape_peak_time_factor"] * T_analysis 
#         for t_idx in range(T_sim):
#             baseline_c_t[t_idx] = params["v_shape_max_cases"] - params["v_shape_slope"] * abs(peak_t_abs - t_idx)
#     elif function_type == 'constant':
#         baseline_c_t = np.full(T_sim, params["constant_cases"])
#     else:
#         raise ValueError(f"Unknown C_T_FUNCTION_TYPE '{function_type}'.")
            
#     baseline_c_t = np.maximum(min_floor_cases, baseline_c_t)
#     return np.round(baseline_c_t).astype(int)

# def generate_delay_distribution(T_delay_max, mean_delay, shape_delay):
#     """Generates onset-to-death probability mass function f_s."""
#     scale_delay = mean_delay / shape_delay
#     s_array = np.arange(T_delay_max + 1) 
#     f_s_unnormalized = np.diff(gamma_dist.cdf(s_array, a=shape_delay, scale=scale_delay))
#     return f_s_unnormalized[:T_delay_max]

# def construct_Q_matrix(c_t, f_s, T_sim):
#     """Constructs the convolution matrix Q from daily cases and the delay distribution."""
#     N = T_sim
#     fs_padded = np.zeros(N)
#     len_f_s = len(f_s)
#     if len_f_s > 0:
#         fs_padded[:min(len_f_s, N)] = f_s[:min(len_f_s, N)]
#     f_matrix_conv = np.zeros((N,N))
#     for i in range(N):
#         relevant_fs = fs_padded[:i+1][::-1]
#         f_matrix_conv[i, :i+1] = relevant_fs
#     return f_matrix_conv @ np.diag(c_t)
    
# def generate_bspline_basis(T_sim, num_knots_J, order):
#     """Generates B-spline basis matrix Bm and scaled time."""
#     t_array = np.arange(1, T_sim + 1)
#     scaler = MinMaxScaler()
#     t_array_scaled = scaler.fit_transform(t_array.reshape(-1,1))
    
#     if num_knots_J <= order -2 : # Not enough knots for interior ones
#         interior_knots = np.array([])
#     else:
#         interior_knots = np.linspace(0, 1, num_knots_J - (order - 2))[1:-1]

#     boundary_knots = np.array([0, 1])
#     knots = np.sort(np.concatenate([
#         np.repeat(boundary_knots[0], order), interior_knots, np.repeat(boundary_knots[1], order)
#     ]))
#     Bm = BSpline(knots, np.eye(num_knots_J), k=order - 1)(t_array_scaled)[:,0,:]
#     return Bm, t_array_scaled.flatten(), scaler

# def generate_intervention_input_matrix(T_array_scaled_sim, intervention_times_true_abs, time_scaler, num_interventions_K):
#     """Generates the Z_input matrix where Z_input[t,k] = (t_scaled - t_k_scaled)_+"""
#     N_sim = len(T_array_scaled_sim)
#     if num_interventions_K == 0: return np.empty((N_sim, 0))
#     intervention_times_true_scaled = time_scaler.transform(np.array(intervention_times_true_abs).reshape(-1,1)).flatten()
#     Z_input = np.zeros((N_sim, num_interventions_K))
#     t_broadcast_scaled_sim = T_array_scaled_sim.reshape(-1, 1) 
#     ti_broadcast_scaled = np.array(intervention_times_true_scaled).reshape(1, num_interventions_K)
#     mask = t_broadcast_scaled_sim >= ti_broadcast_scaled
#     Z_input = (t_broadcast_scaled_sim - ti_broadcast_scaled) * mask.astype(np.float32)
#     return Z_input
    
# def generate_random_effects_eta(T_sim, sigma_eta_true, rng_numpy):
#     """Generates i.i.d. Gaussian random effects eta_0_t."""
#     if sigma_eta_true is None or sigma_eta_true == 0:
#         return np.zeros(T_sim)
#     return rng_numpy.normal(loc=0, scale=sigma_eta_true, size=T_sim)

# def generate_baseline_cfr_zeta(T_sim, T_analysis, cfr_type_code, cfr_params):
#     """Generates the true systematic baseline logit-CFR zeta_0_t."""
#     t_array_sim = np.arange(T_sim)
#     t_prime_for_pattern = t_array_sim / (T_analysis - 1 if T_analysis > 1 else 1)

#     if cfr_type_code == "C1":
#         zeta_0_t = np.full(T_sim, logit(cfr_params["cfr_const"]))
#     elif cfr_type_code == "C2":
#         cfr_t = cfr_params["cfr_start"] - (cfr_params["cfr_start"] - cfr_params["cfr_end"]) * t_prime_for_pattern
#         if T_sim > T_analysis: cfr_t[T_analysis:] = cfr_params["cfr_end"] 
#         zeta_0_t = logit(np.clip(cfr_t, 1e-7, 1-1e-7))
#     elif cfr_type_code == "C3":
#         cfr_t = cfr_params["cfr_mean"] + cfr_params["amp"] * np.sin(2 * np.pi * cfr_params["freq"] * t_prime_for_pattern)
#         zeta_0_t = logit(np.clip(cfr_t, 1e-7, 1-1e-7))
#     elif cfr_type_code == "C4":
#         cfr_t = cfr_params["cfr_base"] + cfr_params["peak_h"] * \
#                 np.exp(-(t_array_sim - cfr_params["peak_t"])**2 / (2 * cfr_params["peak_w"]**2))
#         zeta_0_t = logit(np.clip(cfr_t, 1e-7, 1-1e-7))
#     else: raise ValueError(f"Unknown CFR type: {cfr_type_code}")
#     return zeta_0_t

# def create_sum_to_zero_contrast_matrix(J):
#     """Creates a contrast matrix C of shape (J, J-1) such that alpha = C @ alpha_tilde sums to zero."""
#     C = np.vstack([np.eye(J - 1), -np.ones(J - 1)])
#     return C

# def simulate_scenario_data(scenario_config_dict, run_seed):
#     """
#     Simulates a complete dataset for a single scenario using the unconstrained approach.
#     """
#     rng_numpy = np.random.default_rng(run_seed)
#     T_sim = config.T_SERIES_LENGTH_SIM
#     T_analyze = config.T_ANALYSIS_LENGTH
    
#     baseline_ct_params = {
#         "v_shape_max_cases": config.C_T_VSHAPE_MAX_CASES,
#         "v_shape_slope": config.C_T_VSHAPE_SLOPE,
#         "v_shape_peak_time_factor": config.C_T_VSHAPE_PEAK_TIME_FACTOR,
#         "constant_cases": config.C_T_CONSTANT_CASES 
#     }
#     baseline_c_t = generate_deterministic_baseline_case_counts(
#         T_sim, T_analyze, config.C_T_FUNCTION_TYPE, baseline_ct_params, config.MIN_DRAWN_CASES
#     )

#     # Generate basis and intervention matrices
#     Bm_true, t_array_scaled_sim, time_scaler = generate_bspline_basis(T_sim, config.N_SPLINE_KNOTS_J, config.SPLINE_ORDER)
    
#     num_interventions_K_true = scenario_config_dict["num_interventions_K_true"]
#     true_beta_abs_0 = scenario_config_dict["true_beta_abs_0"]
#     true_lambda_0 = scenario_config_dict["true_lambda_0"]
#     true_intervention_times_0_abs = scenario_config_dict["true_intervention_times_0"]
#     true_beta_signs_0 = scenario_config_dict["true_beta_signs_0"]
    
#     Z_input_true = generate_intervention_input_matrix(
#         t_array_scaled_sim, true_intervention_times_0_abs, 
#         time_scaler, num_interventions_K_true
#     )
    
#     intervention_effect_on_logit_cfr = np.zeros(T_sim)
#     true_beta_0_vector = np.array([])
#     if num_interventions_K_true > 0:
#         true_beta_0_vector = true_beta_abs_0 * true_beta_signs_0
#         lag_matrix_true = (1 - np.exp(-true_lambda_0 * Z_input_true))
#         intervention_effect_on_logit_cfr = np.dot(lag_matrix_true, true_beta_0_vector)
    
#     effect_on_log_ct = config.C_T_INTERVENTION_EFFECT_SCALAR * intervention_effect_on_logit_cfr
#     c_t_final = baseline_c_t * np.exp(effect_on_log_ct)
#     c_t_final = np.maximum(config.MIN_DRAWN_CASES, np.round(c_t_final)).astype(int)
    
#     f_s = generate_delay_distribution(config.F_DELAY_MAX, config.F_MEAN, config.F_SHAPE)
#     Q_true = construct_Q_matrix(c_t_final, f_s, T_sim)
    
#     zeta_0_t = generate_baseline_cfr_zeta(T_sim, T_analyze, scenario_config_dict["cfr_type_code"], scenario_config_dict["cfr_params"])
#     eta_0_t = generate_random_effects_eta(T_sim, config.ETA_SIGMA_TRUE, rng_numpy)
    
#     true_logit_rcf_0_t = zeta_0_t + eta_0_t
#     true_rcf_0_t = sigmoid(true_logit_rcf_0_t)
#     true_logit_r_0_t = zeta_0_t + intervention_effect_on_logit_cfr + eta_0_t
#     true_r_0_t = sigmoid(true_logit_r_0_t)
    
#     mu_0_t = Q_true @ true_r_0_t
#     mu_0_t_clipped = np.maximum(1e-9, mu_0_t)
#     d_t = rng_numpy.poisson(mu_0_t_clipped)
    
#     return {
#         "scenario_id": scenario_config_dict["id"], 
#         "c_t": c_t_final, 
#         "d_t": d_t, 
#         "f_s_true": f_s,
#         "Q_true": Q_true, 
#         "Bm_true": Bm_true, # <-- Return the original basis matrix
#         "Z_input_true": Z_input_true,
#         "beta_signs_true": true_beta_signs_0, 
#         "N_obs": T_sim, 
#         "K_spline_obs": config.N_SPLINE_KNOTS_J, # <-- Return the original number of knots
#         "num_interventions_true_K": num_interventions_K_true,
#         "true_beta_abs_0": true_beta_abs_0, 
#         "true_lambda_0": true_lambda_0,
#         "true_beta_0": true_beta_0_vector,
#         "true_r_0_t": true_r_0_t, 
#         "true_rcf_0_t": true_rcf_0_t,
#         "true_intervention_times_0_abs": true_intervention_times_0_abs, 
#         "run_seed": run_seed
#     }



# def simulate_scenario_data(scenario_config_dict, run_seed):
#     rng_numpy = np.random.default_rng(run_seed)
#     T_sim = config.T_SERIES_LENGTH_SIM
#     T_analyze = config.T_ANALYSIS_LENGTH
    
#     # 1. Generate baseline_c_t (deterministic part)
#     baseline_ct_params = {
#         "function_type": config.C_T_FUNCTION_TYPE,
#         "v_shape_max_cases": config.C_T_VSHAPE_MAX_CASES,
#         "v_shape_slope": config.C_T_VSHAPE_SLOPE,
#         "v_shape_peak_time_factor": config.C_T_VSHAPE_PEAK_TIME_FACTOR,
#         "constant_cases": config.C_T_CONSTANT_CASES 
#     }
#     baseline_c_t = generate_deterministic_baseline_case_counts(
#         T_sim, T_analyze, config.C_T_FUNCTION_TYPE, baseline_ct_params, config.MIN_DRAWN_CASES
#     )

#     # --- Intermediate steps for intervention effect on c_t ---
#     # B-spline basis and time scaler are needed for Z_input_true, which models intervention timing
#     _, t_array_scaled_sim, time_scaler = generate_bspline_basis(
#         T_sim, config.N_SPLINE_KNOTS_J, config.SPLINE_ORDER
#     )
    
#     num_interventions_K_true = scenario_config_dict["num_interventions_K_true"]
#     true_beta_abs_0 = scenario_config_dict["true_beta_abs_0"]
#     true_lambda_0 = scenario_config_dict["true_lambda_0"]
#     true_intervention_times_0_abs = scenario_config_dict["true_intervention_times_0"]
#     true_beta_signs_0 = scenario_config_dict["true_beta_signs_0"]

#     Z_input_true_for_ct_effect = generate_intervention_input_matrix(
#         t_array_scaled_sim, true_intervention_times_0_abs, 
#         time_scaler, num_interventions_K_true
#     )
    
#     Z_lambda_beta_0_t_for_cfr = np.zeros(T_sim) # This is the effect on logit(CFR)
#     if num_interventions_K_true > 0:
#         true_beta_0_vector_for_cfr = true_beta_abs_0 * true_beta_signs_0
#         lag_matrix_true = (1 - np.exp(-true_lambda_0 * Z_input_true_for_ct_effect))
#         Z_lambda_beta_0_t_for_cfr = np.dot(lag_matrix_true, true_beta_0_vector_for_cfr)
    
#     # 2. Modulate baseline_c_t by intervention effects
#     effect_on_log_ct = config.C_T_INTERVENTION_EFFECT_SCALAR * Z_lambda_beta_0_t_for_cfr
#     c_t_final = baseline_c_t * np.exp(effect_on_log_ct)
#     c_t_final = np.maximum(config.MIN_DRAWN_CASES, np.round(c_t_final)).astype(int)
#     # --- End of c_t generation with intervention effects ---

#     # Proceed with DGP using c_t_final
#     f_s = generate_delay_distribution(config.F_DELAY_MAX, config.F_MEAN, config.F_SHAPE)
#     Q_true = construct_Q_matrix(c_t_final, f_s, T_sim) # Use c_t_final
    
#     # Bm_true is independent of c_t, uses T_sim
#     Bm_true, _, _ = generate_bspline_basis(T_sim, config.N_SPLINE_KNOTS_J, config.SPLINE_ORDER)
    
#     zeta_0_t = generate_baseline_cfr_zeta(T_sim, T_analyze, scenario_config_dict["cfr_type_code"], scenario_config_dict["cfr_params"])
        
#     eta_0_t = generate_random_effects_eta(T_sim, config.ETA_SIGMA_TRUE, rng_numpy)
    
#     true_logit_rcf_0_t = zeta_0_t + eta_0_t
#     true_rcf_0_t = sigmoid(true_logit_rcf_0_t)
#     # Z_lambda_beta_0_t_for_cfr is the intervention effect on logit(CFR)
#     true_logit_r_0_t = zeta_0_t + Z_lambda_beta_0_t_for_cfr + eta_0_t
#     true_r_0_t = sigmoid(true_logit_r_0_t)
    
#     mu_0_t = Q_true @ true_r_0_t # Q_true is now based on c_t_final
#     mu_0_t_clipped = np.maximum(1e-9, mu_0_t)
#     d_t = rng_numpy.poisson(mu_0_t_clipped)
    
#     return {
#         "scenario_id": scenario_config_dict["id"], 
#         "c_t": c_t_final, # Use the final c_t
#         "d_t": d_t, 
#         "f_s_true": f_s,
#         "Q_true": Q_true, 
#         "Bm_true": Bm_true, 
#         "Z_input_true": Z_input_true_for_ct_effect, # This is the (t-tk)_+ matrix for sampler
#         "beta_signs_true": true_beta_signs_0, 
#         "N_obs": T_sim, 
#         "K_spline_obs": config.N_SPLINE_KNOTS_J,
#         "num_interventions_true_K": num_interventions_K_true,
#         "true_zeta_0_t": zeta_0_t, 
#         "true_beta_abs_0": true_beta_abs_0, 
#         "true_lambda_0": true_lambda_0, 
#         "true_beta_0": true_beta_0_vector_for_cfr if num_interventions_K_true > 0 else np.array([]),
#         "true_eta_0_t": eta_0_t, 
#         "true_r_0_t": true_r_0_t, 
#         "true_rcf_0_t": true_rcf_0_t, 
#         "true_logit_rcf_0_t": true_logit_rcf_0_t,
#         "run_seed": run_seed, 
#         "T_analysis_length": T_analyze 
#     }


# def simulate_scenario_data(scenario_config_dict, run_seed):
#     rng_numpy = np.random.default_rng(run_seed)
#     T = config.T_SERIES_LENGTH
    
#     # Use new case generation parameters from config
#     c_t = generate_case_counts(T, 
#                                config.C_T_PEAK_TIME_FACTOR, 
#                                config.C_T_PEAK_WIDTH_FACTOR, 
#                                config.C_T_BASELINE_CASES,
#                                config.C_T_PEAK_MAX_ADDITIONAL_CASES, 
#                                rng_numpy)
    
#     f_s = generate_delay_distribution(config.F_DELAY_MAX, config.F_MEAN, config.F_SHAPE)
#     Q_true = construct_Q_matrix(c_t, f_s, T)
#     Bm_true, t_array_scaled, time_scaler = generate_bspline_basis(T, config.N_SPLINE_KNOTS_J, config.SPLINE_ORDER)
    
#     zeta_0_t = generate_baseline_cfr_zeta(T, scenario_config_dict["cfr_type_code"], scenario_config_dict["cfr_params"])
    
#     num_interventions_K_true = scenario_config_dict["num_interventions_K_true"]
#     true_beta_abs_0 = scenario_config_dict["true_beta_abs_0"]
#     true_lambda_0 = scenario_config_dict["true_lambda_0"]
#     true_intervention_times_0 = scenario_config_dict["true_intervention_times_0"]
#     true_beta_signs_0 = scenario_config_dict["true_beta_signs_0"]

#     Z_input_true = generate_intervention_input_matrix(t_array_scaled, true_intervention_times_0, time_scaler, num_interventions_K_true)
    
#     Z_lambda_beta_0_t = np.zeros(T)
#     true_beta_0_vector = np.array([])
#     if num_interventions_K_true > 0:
#         true_beta_0_vector = true_beta_abs_0 * true_beta_signs_0
#         lag_matrix_true = (1 - np.exp(-true_lambda_0 * Z_input_true))
#         Z_lambda_beta_0_t = np.dot(lag_matrix_true, true_beta_0_vector)
        
#     eta_0_t = generate_random_effects_eta(T, config.ETA_SIGMA_TRUE, rng_numpy)
    
#     true_logit_rcf_0_t = zeta_0_t + eta_0_t
#     true_rcf_0_t = sigmoid(true_logit_rcf_0_t)
#     true_logit_r_0_t = zeta_0_t + Z_lambda_beta_0_t + eta_0_t
#     true_r_0_t = sigmoid(true_logit_r_0_t)
    
#     mu_0_t = Q_true @ true_r_0_t
#     mu_0_t_clipped = np.maximum(1e-9, mu_0_t)
#     d_t = rng_numpy.poisson(mu_0_t_clipped)
    
#     return {
#         "scenario_id": scenario_config_dict["id"], "c_t": c_t, "d_t": d_t, "f_s_true": f_s,
#         "Q_true": Q_true, "Bm_true": Bm_true, "Z_input_true": Z_input_true, 
#         "beta_signs_true": true_beta_signs_0, "N_obs": T, "K_spline_obs": config.N_SPLINE_KNOTS_J,
#         "num_interventions_true_K": num_interventions_K_true,
#         "true_zeta_0_t": zeta_0_t, "true_beta_abs_0": true_beta_abs_0, 
#         "true_lambda_0": true_lambda_0, "true_beta_0": true_beta_0_vector,
#         "true_eta_0_t": eta_0_t, "true_r_0_t": true_r_0_t, 
#         "true_rcf_0_t": true_rcf_0_t, "true_logit_rcf_0_t": true_logit_rcf_0_t,
#         "run_seed": run_seed
#     }

# def simulate_scenario_data(scenario_config_dict, run_seed): # Added run_seed
#     """
#     Simulates data for a single scenario.
#     Uses a dedicated numpy RNG for its stochastic parts for reproducibility.
#     """
#     rng_numpy = np.random.default_rng(run_seed) # Create a local RNG

#     T = config.T_SERIES_LENGTH
    
#     c_t = generate_case_counts(T, config.C_T_PEAK_TIME_FACTOR, config.C_T_PEAK_WIDTH_FACTOR, 
#                                config.C_T_AVG_DAILY_CASES, rng_numpy) # Pass RNG
#     f_s = generate_delay_distribution(config.F_DELAY_MAX, config.F_MEAN, config.F_SHAPE)
#     Q_true = construct_Q_matrix(c_t, f_s, T)
#     Bm_true, t_array_scaled, time_scaler = generate_bspline_basis(T, config.N_SPLINE_KNOTS_J, config.SPLINE_ORDER)
    
#     zeta_0_t = generate_baseline_cfr_zeta(T, scenario_config_dict["cfr_type_code"], scenario_config_dict["cfr_params"])
    
#     num_interventions_K_true = scenario_config_dict["num_interventions_K_true"]
#     true_beta_abs_0 = scenario_config_dict["true_beta_abs_0"]
#     true_lambda_0 = scenario_config_dict["true_lambda_0"]
#     true_intervention_times_0 = scenario_config_dict["true_intervention_times_0"]
#     true_beta_signs_0 = scenario_config_dict["true_beta_signs_0"]

#     Z_input_true = generate_intervention_input_matrix(t_array_scaled, true_intervention_times_0, time_scaler, num_interventions_K_true)
    
#     Z_lambda_beta_0_t = np.zeros(T)
#     true_beta_0_vector = np.array([]) # Initialize
#     if num_interventions_K_true > 0:
#         true_beta_0_vector = true_beta_abs_0 * true_beta_signs_0
#         lag_matrix_true = (1 - np.exp(-true_lambda_0 * Z_input_true))
#         Z_lambda_beta_0_t = np.dot(lag_matrix_true, true_beta_0_vector)
        
#     eta_0_t = generate_random_effects_eta(T, config.ETA_SIGMA_TRUE, rng_numpy) # Pass RNG
    
#     true_logit_rcf_0_t = zeta_0_t + eta_0_t
#     true_rcf_0_t = sigmoid(true_logit_rcf_0_t)
#     true_logit_r_0_t = zeta_0_t + Z_lambda_beta_0_t + eta_0_t
#     true_r_0_t = sigmoid(true_logit_r_0_t)
    
#     mu_0_t = Q_true @ true_r_0_t
#     mu_0_t_clipped = np.maximum(1e-9, mu_0_t)
#     d_t = rng_numpy.poisson(mu_0_t_clipped) # Pass RNG
    
#     return {
#         "scenario_id": scenario_config_dict["id"], "c_t": c_t, "d_t": d_t, "f_s_true": f_s,
#         "Q_true": Q_true, "Bm_true": Bm_true, "Z_input_true": Z_input_true, 
#         "beta_signs_true": true_beta_signs_0, "N_obs": T, "K_spline_obs": config.N_SPLINE_KNOTS_J,
#         "num_interventions_true_K": num_interventions_K_true,
#         "true_zeta_0_t": zeta_0_t, "true_beta_abs_0": true_beta_abs_0, 
#         "true_lambda_0": true_lambda_0, "true_beta_0": true_beta_0_vector, # Store the signed beta vector
#         "true_eta_0_t": eta_0_t, "true_r_0_t": true_r_0_t, 
#         "true_rcf_0_t": true_rcf_0_t, "true_logit_rcf_0_t": true_logit_rcf_0_t,
#         "run_seed": run_seed # Store the seed used for this DGP
#     }