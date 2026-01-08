"""
Configuration module for sCFR simulation study.

This module defines all parameters for the simulation study including:
- Monte Carlo simulation settings
- Data generation parameters
- Intervention effect specifications
- Scenario definitions
- MCMC sampler configuration
- Output directory structure
"""

import os
import numpy as np

# =============================================================================
# Simulation Settings
# =============================================================================

NUM_MONTE_CARLO_RUNS = 5
T_ANALYSIS_LENGTH = 200
T_SIMULATION_BUFFER = 60
T_SERIES_LENGTH_SIM = T_ANALYSIS_LENGTH + T_SIMULATION_BUFFER
GLOBAL_BASE_SEED = 2025
NUM_CORES_TO_USE = 5
OVERWRITE_EXISTING_RESULTS = True

# =============================================================================
# Delay Distribution Parameters (Gamma)
# =============================================================================

F_MEAN = 15.43
F_SHAPE = 2.03
F_DELAY_MAX = T_ANALYSIS_LENGTH + T_SIMULATION_BUFFER

# =============================================================================
# B-spline Basis Parameters
# =============================================================================

N_SPLINE_KNOTS_J = 10
SPLINE_ORDER = 4

# =============================================================================
# Case Count Generation Parameters
# =============================================================================

C_T_FUNCTION_TYPE = 'v_shape'
C_T_VSHAPE_MAX_CASES = 10000
C_T_VSHAPE_PEAK_TIME_FACTOR = 0.4
C_T_VSHAPE_SLOPE = 75
C_T_CONSTANT_CASES = 500
MIN_DRAWN_CASES = 20

# =============================================================================
# Intervention Effect Parameters
# =============================================================================

TRUE_BETA_STEP_ABS_K1 = [0.6]
TRUE_BETA_SLOPE_ABS_K1 = [0.6]
TRUE_BETA_STEP_ABS_K2 = [0.6, 0.4]
TRUE_BETA_SLOPE_ABS_K2 = [0.6, 0.4]
TRUE_T_K1_FACTOR = [0.5]
TRUE_T_K2_FACTOR = [0.33, 0.66]
SIGNS_K1_STEP = [-1]
SIGNS_K1_SLOPE = [-1]
SIGNS_K2_STEP = [-1, 1]
SIGNS_K2_SLOPE = [-1, 1]

SIGMA_DELTA_SCENARIOS = {
    "I0": 0.20,
    "I1": 0.00,
    "I2": 0.10
}

# =============================================================================
# Scenario Definitions
# =============================================================================

cfr_types_params = {
    "C1": {"name": "Constant", "params": {"cfr_const": 0.02}},
    "C2": {"name": "Linear Decr.", "params": {"cfr_start": 0.03, "cfr_end": 0.01}},
    "C3": {"name": "Sine Wave", "params": {"cfr_mean": 0.02, "amp": 0.01, "freq": 1.0}},
    "C4": {"name": "Gaussian Kernel", "params": {"cfr_base": 0.01, "peak_h": 0.04,
                                               "peak_t_factor": 0.5, "peak_w_factor": 0.2}}
}

intervention_types_params = {
    "I0": {"name": "K=0", "K": 0, "beta_step_abs": [], "beta_slope_abs": [],
           "times_factor": [], "signs_step": [], "signs_slope": []},
    "I1": {"name": "K=1 (Slope<0)", "K": 1,
           "beta_step_abs": TRUE_BETA_STEP_ABS_K1, "beta_slope_abs": TRUE_BETA_SLOPE_ABS_K1,
           "times_factor": TRUE_T_K1_FACTOR, "signs_step": SIGNS_K1_STEP, "signs_slope": SIGNS_K1_SLOPE},
    "I2": {"name": "K=2 (Slope<0,>0)", "K": 2,
           "beta_step_abs": TRUE_BETA_STEP_ABS_K2, "beta_slope_abs": TRUE_BETA_SLOPE_ABS_K2,
           "times_factor": TRUE_T_K2_FACTOR, "signs_step": SIGNS_K2_STEP, "signs_slope": SIGNS_K2_SLOPE}
}

SCENARIOS = []
scen_counter = 1
for cfr_code, cfr_data in cfr_types_params.items():
    for int_code, int_data in intervention_types_params.items():
        scenario_id = f"S{scen_counter:02d}"
        current_cfr_params = cfr_data["params"].copy()
        
        if cfr_code == "C4":
            current_cfr_params["peak_t"] = current_cfr_params["peak_t_factor"] * T_ANALYSIS_LENGTH
            current_cfr_params["peak_w"] = current_cfr_params["peak_w_factor"] * T_ANALYSIS_LENGTH

        current_int_times = [t_factor * T_ANALYSIS_LENGTH for t_factor in int_data["times_factor"]]

        SCENARIOS.append({
            "id": scenario_id,
            "cfr_type_code": cfr_code, "cfr_type_name": cfr_data["name"],
            "intervention_type_code": int_code, "intervention_type_name": int_data["name"],
            "cfr_params": current_cfr_params,
            "num_interventions_K_true": int_data["K"],
            "true_beta_abs_0": np.array(int_data["beta_step_abs"]),
            "true_beta_slope_abs_0": np.array(int_data["beta_slope_abs"]),
            "true_intervention_times_0": np.array(current_int_times),
            "true_beta_signs_0": np.array(int_data["signs_step"]),
            "true_beta_slope_signs_0": np.array(int_data["signs_slope"]),
            "sigma_delta_true": SIGMA_DELTA_SCENARIOS[int_code]
        })
        scen_counter += 1

# =============================================================================
# MCMC Configuration
# =============================================================================

NUM_WARMUP = 1000
NUM_SAMPLES = 1000
NUM_CHAINS = 1

# =============================================================================
# Output Directory Structure
# =============================================================================

OUTPUT_DIR_BASE = "./simulation_outputs/"
OUTPUT_DIR_PLOTS = os.path.join(OUTPUT_DIR_BASE, "plots/")
OUTPUT_DIR_TABLES = os.path.join(OUTPUT_DIR_BASE, "tables/")
OUTPUT_DIR_RESULTS_CSV = os.path.join(OUTPUT_DIR_BASE, "results_csv/")
OUTPUT_DIR_POSTERIOR_SAMPLES = os.path.join(OUTPUT_DIR_BASE, "posterior_samples_raw/")
OUTPUT_DIR_BENCHMARK_RESULTS = os.path.join(OUTPUT_DIR_BASE, "benchmarks_results/")
OUTPUT_DIR_POSTERIOR_SUMMARIES = os.path.join(OUTPUT_DIR_BASE, "posterior_summaries/")
OUTPUT_DIR_RUN_METRICS_JSON = os.path.join(OUTPUT_DIR_BASE, "run_metrics_json/")
OUTPUT_DIR_LOGS = os.path.join(OUTPUT_DIR_BASE, "logs/")
