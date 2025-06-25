# File: data_generation.py
# Description: Functions for generating simulated data for each scenario.

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist, norm, poisson
from scipy.special import expit as sigmoid, logit
from scipy.interpolate import BSpline
from sklearn.preprocessing import MinMaxScaler

import config # Import parameters from config.py

# def generate_case_counts(T, peak_time_factor, peak_width_factor, avg_daily_cases, rng_numpy):
#     """Generates daily case counts c_t. Uses a numpy RNG."""
#     t_array = np.arange(T)
#     peak_time = peak_time_factor * T
#     peak_width = peak_width_factor * T
#     profile = norm.pdf(t_array, loc=peak_time, scale=peak_width)
#     total_cases_target = T * avg_daily_cases
#     if np.sum(profile) > 1e-6:
#         lambda_t = profile / np.sum(profile) * total_cases_target
#     else:
#         lambda_t = np.full(T, avg_daily_cases)
#     c_t = rng_numpy.poisson(np.maximum(1, lambda_t)).astype(int)
#     return c_t

# def generate_case_counts(T, peak_time_factor, peak_width_factor, 
#                          baseline_cases, peak_max_additional_cases, rng_numpy):
#     """
#     Generates daily case counts c_t with a baseline and a peak.
#     Uses a numpy RNG for stochasticity.
#     """
#     t_array = np.arange(T)
#     peak_time = peak_time_factor * T
#     peak_width = peak_width_factor * T
    
#     # Generate a Gaussian profile (unnormalized height)
#     peak_profile_unnormalized = norm.pdf(t_array, loc=peak_time, scale=peak_width)
    
#     # Scale it so its max is 1 (if peak_width is not too large), then multiply by peak_max_additional_cases
#     if np.max(peak_profile_unnormalized) > 1e-9: # Avoid division by zero if profile is effectively flat
#         peak_component = (peak_profile_unnormalized / np.max(peak_profile_unnormalized)) * peak_max_additional_cases
#     else: 
#         # If peak_width makes the profile flat and near zero, no significant peak component
#         peak_component = np.zeros(T)
            
#     lambda_t = baseline_cases + peak_component
#     lambda_t = np.maximum(1.0, lambda_t) # Ensure lambda is at least 1.0 for Poisson mean
    
#     c_t = rng_numpy.poisson(lambda_t).astype(int)
#     c_t = np.maximum(1, c_t) # Ensure at least 1 case is actually drawn for stability
#     return c_t

# def generate_case_counts(T_total_sim_length, T_analysis_length,
#                          peak_time_factor, peak_width_factor,
#                          baseline_cases, peak_max_additional_cases,
#                          tail_flat_duration_factor, min_drawn_cases, rng_numpy):
#     t_array = np.arange(T_total_sim_length)
#     peak_time_abs = peak_time_factor * T_analysis_length
#     peak_width_abs = peak_width_factor * T_analysis_length
#     if peak_width_abs == 0: peak_width_abs = T_analysis_length * 0.01 # Avoid zero width

#     peak_profile_unnormalized = norm.pdf(t_array, loc=peak_time_abs, scale=peak_width_abs)
    
#     if np.max(peak_profile_unnormalized) > 1e-9:
#         peak_component = (peak_profile_unnormalized / np.max(peak_profile_unnormalized)) * peak_max_additional_cases
#     else: 
#         peak_component = np.zeros(T_total_sim_length)
            
#     lambda_t = baseline_cases + peak_component
    
#     # Ensure lambda_t doesn't drop too low during the analysis period's tails and buffer
#     tail_flat_duration_points_in_analysis = int(T_analysis_length * tail_flat_duration_factor)

#     # Start of analysis window
#     if tail_flat_duration_points_in_analysis > 0:
#          lambda_t[:tail_flat_duration_points_in_analysis] = np.maximum(
#             lambda_t[:tail_flat_duration_points_in_analysis], baseline_cases
#         )
#     # End of analysis window up to start of buffer
#     if T_analysis_length > tail_flat_duration_points_in_analysis:
#         lambda_t[T_analysis_length - tail_flat_duration_points_in_analysis : T_analysis_length] = np.maximum(
#             lambda_t[T_analysis_length - tail_flat_duration_points_in_analysis : T_analysis_length], baseline_cases
#         )
#     # Buffer period
#     if T_total_sim_length > T_analysis_length:
#         lambda_t[T_analysis_length:] = np.maximum(lambda_t[T_analysis_length:], baseline_cases)

#     lambda_t = np.maximum(float(min_drawn_cases), lambda_t) # Ensure lambda is at least min_drawn_cases
    
#     c_t = rng_numpy.poisson(lambda_t).astype(int)
#     c_t = np.maximum(min_drawn_cases, c_t) 
#     return c_t

def generate_deterministic_baseline_case_counts(T_sim, T_analysis, function_type, params, min_floor_cases):
    """
    Generates a deterministic baseline for case counts based on a simple function.

    Args:
        T_sim (int): Total simulation length.
        T_analysis (int): Length of the primary analysis window.
        function_type (str): Type of function to use ('v_shape' or 'constant').
        params (dict): Parameters specific to the function type.
        min_floor_cases (int): The absolute minimum value for the baseline.

    Returns:
        np.ndarray: The deterministic baseline case counts vector of length T_sim.
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
    # f_s = f_s_unnormalized / np.sum(f_s_unnormalized)
    f_s = f_s_unnormalized
    return f_s[:T_delay_max]

def construct_Q_matrix(c_t, f_s, T_sim):
    """Constructs the Q matrix (fc_mat in sampler.py)"""
    N = T_sim 
    fs_padded = np.zeros(N)
    len_f_s = len(f_s)
    if len_f_s > 0: fs_padded[:min(len_f_s, N)] = f_s[:min(len_f_s, N)]
    
    f_matrix_conv = np.zeros((N,N))
    for i in range(N):
        relevant_fs = fs_padded[:i+1][::-1]
        f_matrix_conv[i, :i+1] = relevant_fs
    Q_matrix = f_matrix_conv @ np.diag(c_t)
    return Q_matrix
    
def generate_bspline_basis(T_sim, num_knots_J, order):
    """Generates B-spline basis matrix Bm and scaled time."""
    t_array = np.arange(1, T_sim + 1)
    scaler = MinMaxScaler()
    t_array_scaled = scaler.fit_transform(t_array.reshape(-1,1))
    
    if num_knots_J <= order -2 : # Not enough knots for interior ones
        interior_knots = np.array([])
    else:
        interior_knots = np.linspace(0, 1, num_knots_J - (order - 2))[1:-1]

    boundary_knots = np.array([0, 1])
    knots = np.sort(np.concatenate([
        np.repeat(boundary_knots[0], order), interior_knots, np.repeat(boundary_knots[1], order)
    ]))
    Bm = BSpline(knots, np.eye(num_knots_J), k=order - 1)(t_array_scaled)[:,0,:]
    return Bm, t_array_scaled.flatten(), scaler

def generate_intervention_input_matrix(T_array_scaled_sim, intervention_times_true_abs, time_scaler, num_interventions_K):
    """
    Generates the Z_input matrix (T in sampler.py) where Z_input[t,k] = (t_scaled - t_k_scaled)_+
    intervention_times_true are on the original time scale [0, T-1] or [1, T]
    """
    N_sim = len(T_array_scaled_sim)
    if num_interventions_K == 0: return np.empty((N_sim, 0))
    intervention_times_true_scaled = time_scaler.transform(np.array(intervention_times_true_abs).reshape(-1,1)).flatten()
    Z_input = np.zeros((N_sim, num_interventions_K))
    t_broadcast_scaled_sim = T_array_scaled_sim.reshape(-1, 1) 
    ti_broadcast_scaled = np.array(intervention_times_true_scaled).reshape(1, num_interventions_K)
    mask = t_broadcast_scaled_sim >= ti_broadcast_scaled
    Z_input = (t_broadcast_scaled_sim - ti_broadcast_scaled) * mask.astype(np.float32)
    return Z_input
    
def generate_random_effects_eta(T_sim, sigma_eta_true, rng_numpy):
    """Generates i.i.d. Gaussian random effects eta_0_t. Uses a numpy RNG."""
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
        cfr_t = cfr_params["cfr_base"] + cfr_params["peak_h"] * \
                np.exp(-(t_array_sim - cfr_params["peak_t"])**2 / (2 * cfr_params["peak_w"]**2))
        zeta_0_t = logit(np.clip(cfr_t, 1e-7, 1-1e-7))
    else: raise ValueError(f"Unknown CFR type: {cfr_type_code}")
    return zeta_0_t

def simulate_scenario_data(scenario_config_dict, run_seed):
    """
    Simulates a complete dataset for a single scenario configuration and run.

    This function orchestrates the entire data generation process, including:
    1. Generating a deterministic baseline for case counts ($c_t$).
    2. Modulating $c_t$ based on intervention effects.
    3. Generating the true baseline CFR ($\zeta_t$), intervention effects, and random noise ($\eta_t$).
    4. Calculating the true final CFR ($r_t$) and counterfactual CFR ($r_{cf,t}$).
    5. Constructing the convolution matrix $Q$.
    6. Simulating daily deaths ($d_t$) from a Poisson distribution.
    
    Args:
        scenario_config_dict (dict): Configuration for the specific scenario from `config.SCENARIOS`.
        run_seed (int): A seed for the random number generator to ensure reproducibility.

    Returns:
        dict: A dictionary containing all generated data, true parameters, and inputs for the model.
    """
    rng_numpy = np.random.default_rng(run_seed)
    T_sim = config.T_SERIES_LENGTH_SIM
    T_analyze = config.T_ANALYSIS_LENGTH
    
    # 1. Generate baseline case counts (deterministic part)
    baseline_ct_params = {
        "v_shape_max_cases": config.C_T_VSHAPE_MAX_CASES,
        "v_shape_slope": config.C_T_VSHAPE_SLOPE,
        "v_shape_peak_time_factor": config.C_T_VSHAPE_PEAK_TIME_FACTOR,
        "constant_cases": config.C_T_CONSTANT_CASES 
    }
    baseline_c_t = generate_deterministic_baseline_case_counts(
        T_sim, T_analyze, config.C_T_FUNCTION_TYPE, baseline_ct_params, config.MIN_DRAWN_CASES
    )

    # 2. Generate elements needed to calculate intervention effect on c_t
    _, t_array_scaled_sim, time_scaler = generate_bspline_basis(T_sim, config.N_SPLINE_KNOTS_J, config.SPLINE_ORDER)
    num_interventions_K_true = scenario_config_dict["num_interventions_K_true"]
    true_beta_abs_0 = scenario_config_dict["true_beta_abs_0"]
    true_lambda_0 = scenario_config_dict["true_lambda_0"]
    true_intervention_times_0_abs = scenario_config_dict["true_intervention_times_0"]
    true_beta_signs_0 = scenario_config_dict["true_beta_signs_0"]
    
    Z_input_true = generate_intervention_input_matrix(t_array_scaled_sim, true_intervention_times_0_abs, time_scaler, num_interventions_K_true)
    
    Z_lambda_beta_0_t_for_cfr = np.zeros(T_sim)
    true_beta_0_vector = np.array([])
    if num_interventions_K_true > 0:
        true_beta_0_vector = true_beta_abs_0 * true_beta_signs_0
        lag_matrix_true = (1 - np.exp(-true_lambda_0 * Z_input_true))
        Z_lambda_beta_0_t_for_cfr = np.dot(lag_matrix_true, true_beta_0_vector)
    
    # 3. Modulate baseline c_t by intervention effects
    effect_on_log_ct = config.C_T_INTERVENTION_EFFECT_SCALAR * Z_lambda_beta_0_t_for_cfr
    c_t_final = baseline_c_t * np.exp(effect_on_log_ct)
    c_t_final = np.maximum(config.MIN_DRAWN_CASES, np.round(c_t_final)).astype(int)
    
    # 4. Proceed with DGP using c_t_final
    f_s = generate_delay_distribution(config.F_DELAY_MAX, config.F_MEAN, config.F_SHAPE)
    Q_true = construct_Q_matrix(c_t_final, f_s, T_sim)
    Bm_true, _, _ = generate_bspline_basis(T_sim, config.N_SPLINE_KNOTS_J, config.SPLINE_ORDER)
    zeta_0_t = generate_baseline_cfr_zeta(T_sim, T_analyze, scenario_config_dict["cfr_type_code"], scenario_config_dict["cfr_params"])
    eta_0_t = generate_random_effects_eta(T_sim, config.ETA_SIGMA_TRUE, rng_numpy)
    
    true_logit_rcf_0_t = zeta_0_t + eta_0_t
    true_rcf_0_t = sigmoid(true_logit_rcf_0_t)
    true_logit_r_0_t = zeta_0_t + Z_lambda_beta_0_t_for_cfr + eta_0_t
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
        "run_seed": run_seed, 
        "T_analysis_length": T_analyze 
    }

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