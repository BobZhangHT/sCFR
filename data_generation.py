import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist, norm, poisson
from scipy.special import expit as sigmoid, logit
from scipy.interpolate import BSpline
from sklearn.preprocessing import MinMaxScaler
import config

def generate_deterministic_baseline_case_counts(T_sim, T_analysis, function_type, params, min_floor_cases):
    """
    Generates a deterministic baseline for case counts based on a specified function.

    Args:
        T_sim (int): The total simulation time length.
        T_analysis (int): The analysis time length.
        function_type (str): The type of function to use for generation (e.g., 'v_shape', 'constant').
        params (dict): A dictionary of parameters for the specified function.
        min_floor_cases (int): The minimum number of cases for any given day.

    Returns:
        numpy.ndarray: An array of generated case counts.
    """
    baseline_c_t = np.zeros(T_sim)
    
    # Generate cases based on the chosen functional form
    if function_type == 'v_shape':
        peak_t_abs = params["v_shape_peak_time_factor"] * T_analysis 
        for t_idx in range(T_sim):
            baseline_c_t[t_idx] = params["v_shape_max_cases"] - params["v_shape_slope"] * abs(peak_t_abs - t_idx)
    elif function_type == 'constant':
        baseline_c_t = np.full(T_sim, params["constant_cases"])
    else:
        raise ValueError(f"Unknown C_T_FUNCTION_TYPE '{function_type}'.")
            
    # Ensure case counts do not fall below the minimum floor
    baseline_c_t = np.maximum(min_floor_cases, baseline_c_t)
    return np.round(baseline_c_t).astype(int)

def generate_delay_distribution(T_delay_max, mean_delay, shape_delay):
    """
    Generates the onset-to-death probability mass function (f_s) from a Gamma distribution.

    Args:
        T_delay_max (int): The maximum delay to consider.
        mean_delay (float): The mean of the gamma distribution for the delay.
        shape_delay (float): The shape parameter of the gamma distribution for the delay.

    Returns:
        numpy.ndarray: The probability mass function for the delay.
    """
    scale_delay = mean_delay / shape_delay
    s_array = np.arange(T_delay_max + 1) 
    # The PMF is the difference in the CDF at consecutive time points
    f_s_unnormalized = np.diff(gamma_dist.cdf(s_array, a=shape_delay, scale=scale_delay))
    return f_s_unnormalized[:T_delay_max]

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
    # Pad the delay distribution with zeros to match the simulation length
    fs_padded = np.zeros(N)
    len_f_s = len(f_s)
    if len_f_s > 0:
        fs_padded[:min(len_f_s, N)] = f_s[:min(len_f_s, N)]
    
    # Create the convolution matrix
    f_matrix_conv = np.zeros((N,N))
    for i in range(N):
        relevant_fs = fs_padded[:i+1][::-1] # Reverse for convolution
        f_matrix_conv[i, :i+1] = relevant_fs
        
    # Element-wise product with diagonal matrix of case counts
    return f_matrix_conv @ np.diag(c_t)
    
def generate_bspline_basis(T_sim, num_knots_J, order):
    """
    Generates a B-spline basis matrix Bm.

    Args:
        T_sim (int): The total simulation time length.
        num_knots_J (int): The number of knots for the B-spline.
        order (int): The order of the B-spline.

    Returns:
        numpy.ndarray: The B-spline basis matrix.
    """
    t_array_unscaled = np.arange(1, T_sim + 1)
    # Scale time to [0, 1] for stable knot placement
    t_array_scaled = np.linspace(0, 1, T_sim)
    
    # Define interior knots, spread evenly across the [0, 1] interval
    if num_knots_J <= order - 2: # Not enough knots for interior ones
        interior_knots = np.array([])
    else:
        interior_knots = np.linspace(0, 1, num_knots_J - (order - 2))[1:-1]

    # Define boundary knots, repeated 'order' times for standard B-spline construction
    boundary_knots = np.array([0, 1])
    knots = np.sort(np.concatenate([
        np.repeat(boundary_knots[0], order), interior_knots, np.repeat(boundary_knots[1], order)
    ]))
    
    # Generate the basis matrix using scipy's BSpline
    Bm = BSpline(knots, np.eye(num_knots_J), k=order - 1)(t_array_scaled)
    return Bm

def generate_intervention_input_matrix(T_array_unscaled_sim, intervention_times_true_abs, num_interventions_K):
    """
    Generates the Z_input matrix where Z_input[t,k] = (t - t_k)_+.

    Args:
        T_array_unscaled_sim (numpy.ndarray): Array of unscaled time points.
        intervention_times_true_abs (list): A list of absolute intervention times.
        num_interventions_K (int): The number of interventions.

    Returns:
        numpy.ndarray: The intervention input matrix Z.
    """
    N_sim = len(T_array_unscaled_sim)
    if num_interventions_K == 0: return np.empty((N_sim, 0))
    
    Z_input = np.zeros((N_sim, num_interventions_K))
    # Use broadcasting for efficient calculation of (t - t_k)
    t_broadcast_unscaled = T_array_unscaled_sim.reshape(-1, 1) 
    ti_broadcast_unscaled = np.array(intervention_times_true_abs).reshape(1, num_interventions_K)
    
    # The effect starts only after the intervention time
    mask = t_broadcast_unscaled >= ti_broadcast_unscaled
    Z_input = (t_broadcast_unscaled - ti_broadcast_unscaled) * mask.astype(np.float32)
    return Z_input
    
def generate_random_effects_eta(T_sim, sigma_eta_true, rng_numpy):
    """
    Generates i.i.d. Gaussian random effects eta_0_t.

    Args:
        T_sim (int): The total simulation time length.
        sigma_eta_true (float): The standard deviation of the random effects.
        rng_numpy (numpy.random.Generator): The random number generator.

    Returns:
        numpy.ndarray: An array of random effects.
    """
    if sigma_eta_true is None or sigma_eta_true == 0:
        return np.zeros(T_sim)
    return rng_numpy.normal(loc=0, scale=sigma_eta_true, size=T_sim)

def generate_baseline_cfr_zeta(T_sim, T_analysis, cfr_type_code, cfr_params):
    """
    Generates the true systematic baseline logit-CFR zeta_0_t.

    Args:
        T_sim (int): The total simulation time length.
        T_analysis (int): The analysis time length.
        cfr_type_code (str): The code for the CFR pattern type (e.g., 'C1', 'C2').
        cfr_params (dict): A dictionary of parameters for the CFR pattern.

    Returns:
        numpy.ndarray: The baseline logit-CFR array.
    """
    t_array_sim = np.arange(T_sim)
    # Scale time to [0, 1] over the analysis period for pattern generation
    t_prime_for_pattern = t_array_sim / (T_analysis - 1 if T_analysis > 1 else 1)

    if cfr_type_code == "C1": # Constant CFR
        zeta_0_t = np.full(T_sim, logit(cfr_params["cfr_const"]))
    elif cfr_type_code == "C2": # Linearly decreasing CFR
        cfr_t = cfr_params["cfr_start"] - (cfr_params["cfr_start"] - cfr_params["cfr_end"]) * t_prime_for_pattern
        if T_sim > T_analysis: cfr_t[T_analysis:] = cfr_params["cfr_end"] # Hold constant after analysis period
        zeta_0_t = logit(np.clip(cfr_t, 1e-7, 1-1e-7))
    elif cfr_type_code == "C3": # Sinusoidal CFR
        cfr_t = cfr_params["cfr_mean"] + cfr_params["amp"] * np.sin(2 * np.pi * cfr_params["freq"] * t_prime_for_pattern)
        zeta_0_t = logit(np.clip(cfr_t, 1e-7, 1-1e-7))
    elif cfr_type_code == "C4": # Gaussian kernel CFR
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

    Args:
        scenario_config_dict (dict): A dictionary containing the configuration for the scenario.
        run_seed (int): The seed for the random number generator.

    Returns:
        dict: A dictionary containing the simulated data and true parameters.
    """
    rng_numpy = np.random.default_rng(run_seed)
    T_sim = config.T_SERIES_LENGTH_SIM
    T_analyze = config.T_ANALYSIS_LENGTH
    t_array_unscaled_sim = np.arange(T_sim)
    
    # Define parameters for case count generation
    baseline_ct_params = {
        "v_shape_max_cases": config.C_T_VSHAPE_MAX_CASES,
        "v_shape_slope": config.C_T_VSHAPE_SLOPE,
        "v_shape_peak_time_factor": config.C_T_VSHAPE_PEAK_TIME_FACTOR,
        "constant_cases": config.C_T_CONSTANT_CASES 
    }
    # Generate case counts (c_t)
    c_t_final = generate_deterministic_baseline_case_counts(
        T_sim, T_analyze, config.C_T_FUNCTION_TYPE, baseline_ct_params, config.MIN_DRAWN_CASES
    )

    # Generate B-spline basis and intervention matrices
    Bm_true = generate_bspline_basis(T_sim, config.N_SPLINE_KNOTS_J, config.SPLINE_ORDER)
    
    # Get intervention parameters from the scenario configuration
    num_interventions_K_true = scenario_config_dict["num_interventions_K_true"]
    true_beta_abs_0 = scenario_config_dict["true_beta_abs_0"]
    true_lambda_0 = scenario_config_dict["true_lambda_0"]
    true_intervention_times_0_abs = scenario_config_dict["true_intervention_times_0"]
    true_beta_signs_0 = scenario_config_dict["true_beta_signs_0"]
    
    # Generate the intervention input matrix Z
    Z_input_true = generate_intervention_input_matrix(
        t_array_unscaled_sim, true_intervention_times_0_abs, num_interventions_K_true
    )
    
    # Calculate the total intervention effect on the logit-CFR
    intervention_effect_on_logit_cfr = np.zeros(T_sim)
    true_beta_0_vector = np.array([])
    if num_interventions_K_true > 0:
        true_beta_0_vector = true_beta_abs_0 * true_beta_signs_0
        lag_matrix_true = (1 - np.exp(-true_lambda_0 * Z_input_true))
        intervention_effect_on_logit_cfr = np.dot(lag_matrix_true, true_beta_0_vector)
    
    # Generate the delay distribution and the convolution matrix Q
    f_s = generate_delay_distribution(config.F_DELAY_MAX, config.F_MEAN, config.F_SHAPE)
    Q_true = construct_Q_matrix(c_t_final, f_s, T_sim)
    
    # Generate the components of the true fatality rate
    zeta_0_t = generate_baseline_cfr_zeta(T_sim, T_analyze, scenario_config_dict["cfr_type_code"], scenario_config_dict["cfr_params"])
    eta_0_t = generate_random_effects_eta(T_sim, config.ETA_SIGMA_TRUE, rng_numpy)
    
    # Calculate true counterfactual (rcf) and factual (r) fatality rates
    true_logit_rcf_0_t = zeta_0_t + eta_0_t
    true_rcf_0_t = sigmoid(true_logit_rcf_0_t)
    true_logit_r_0_t = zeta_0_t + intervention_effect_on_logit_cfr + eta_0_t
    true_r_0_t = sigmoid(true_logit_r_0_t)
    
    # Calculate the expected number of deaths (mu)
    mu_0_t = Q_true @ true_r_0_t
    mu_0_t_clipped = np.maximum(1e-9, mu_0_t) # Clip to avoid issues with Poisson lambda=0
    # Generate observed deaths (d_t) from a Poisson distribution
    d_t = rng_numpy.poisson(mu_0_t_clipped)
    
    # Return a dictionary containing all simulated data and ground truth values
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
