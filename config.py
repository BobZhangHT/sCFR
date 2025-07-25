# File: config.py
# Description: Central configuration file for the simulation study.
#              All simulation parameters, scenario definitions, and output paths are defined here.

import os
import numpy as np

# ----------------------------------------------------------------------------
# General Simulation Parameters
# ----------------------------------------------------------------------------
NUM_MONTE_CARLO_RUNS = 10
T_ANALYSIS_LENGTH = 200       # << NEW: The period we actually care about evaluating
T_SIMULATION_BUFFER = 60      # << NEW: Simulate for this many extra days
T_SERIES_LENGTH_SIM = T_ANALYSIS_LENGTH + T_SIMULATION_BUFFER # << NEW: Total days to simulate

GLOBAL_BASE_SEED = 2025
NUM_CORES_TO_USE = 5#os.cpu_count() - 1 if os.cpu_count() > 1 else 1

PENALTY_GRID_ALPHA = [0.01, 0.1, 1, 10, 100]
PENALTY_GRID_GAMMA_LAMBDA = [0.01, 0.1, 1, 10, 100]
N_BOOTSTRAPS_ITS = 200

OVERWRITE_EXISTING_RESULTS = True  # If True, will re-run all simulations regardless of saved files.
                                   # If False, will skip runs with existing metric files.

USE_CONSTRAINT = False # Use constraint in the model

# ----------------------------------------------------------------------------
# Data Generation Process (DGP) Parameters
# ----------------------------------------------------------------------------
# Onset-to-Death Distribution (f_s)
F_MEAN = 15.43
F_SHAPE = 2.03
F_DELAY_MAX = T_ANALYSIS_LENGTH + T_SIMULATION_BUFFER

# B-spline Basis for baseline CFR
N_SPLINE_KNOTS_J = 10 # Number of knots. A key tuning parameter for smoothness.
SPLINE_ORDER = 4      # 4 for cubic B-splines.

# I.i.d. Random Effect (eta_t) in the true DGP
# Set to 0 if the model being tested does not explicitly account for this term.
ETA_SIGMA_TRUE = 0.0 # Standard deviation of the random effect on the logit scale.

# Confirmed Cases (c_t) Generation
# Defines a deterministic baseline shape which is then modulated by intervention effects.
C_T_FUNCTION_TYPE = 'v_shape'      # 'v_shape' or 'constant'
C_T_VSHAPE_MAX_CASES = 10000        # Max cases for the V-shape baseline at its peak.
C_T_VSHAPE_PEAK_TIME_FACTOR = 0.4  # Peak time relative to T_ANALYSIS_LENGTH.
C_T_VSHAPE_SLOPE = 75              # Slope for the V-shape function.
C_T_CONSTANT_CASES = 500           # Case count if C_T_FUNCTION_TYPE is 'constant'.
MIN_DRAWN_CASES = 20               # Absolute minimum for a day's case count.

TRUE_BETA_ABS_K1 = [1.5]
TRUE_BETA_ABS_K2 = [1.5, 1.0]
TRUE_LAMBDA_K1 = [0.1]
TRUE_LAMBDA_K2 = [0.1, 0.15]
TRUE_T_K1_FACTOR = [0.5] 
TRUE_T_K2_FACTOR = [0.33, 0.66]
SIGNS_K1 = [-1]
SIGNS_K2 = [-1, 1]

# # Scalar for how intervention effects on CFR translate to effects on log(c_t)
# C_T_INTERVENTION_EFFECT_SCALAR = 0.5

# True Intervention Parameters (magnitudes increased for visual clarity)
# TRUE_BETA_ABS_K1 = [1]
# TRUE_BETA_ABS_K2 = [1, 1]
# TRUE_LAMBDA_K1 = [0.3]
# TRUE_LAMBDA_K2 = [0.3, 0.3]#[0.15, 0.12]
# TRUE_T_K1_FACTOR = [0.5] 
# TRUE_T_K2_FACTOR = [0.33, 0.66]
# SIGNS_K1 = [-1]
# SIGNS_K2 = [-1, 1]


# TRUE_BETA_ABS_K1 = [1]
# TRUE_BETA_ABS_K2 = [1, 1]
# TRUE_LAMBDA_K1 = [1]
# TRUE_LAMBDA_K2 = [1, 1]
# TRUE_T_K1_FACTOR = [0.5] 
# TRUE_T_K2_FACTOR = [0.33, 0.66]
# SIGNS_K1 = [-1]
# SIGNS_K2 = [-1, 1]

# ----------------------------------------------------------------------------
# Scenario Definitions (12 scenarios)
# ----------------------------------------------------------------------------
SCENARIOS = []
cfr_types_params = {
    "C1": {"name": "Constant", "params": {"cfr_const": 0.02}},
    "C2": {"name": "Linear Decr.", "params": {"cfr_start": 0.03, "cfr_end": 0.01}},
    "C3": {"name": "Sine Wave", "params": {"cfr_mean": 0.02, "amp": 0.01, "freq": 1.0}},
    "C4": {"name": "Gaussian Kernel", "params": {"cfr_base": 0.01, "peak_h": 0.04, 
                                               "peak_t_factor": 0.5, "peak_w_factor": 0.2}}
}

# cfr_types_params = {
#     # ** FIX: Baselines are slightly increased to make intervention effects more evident **
#     "C1": {"name": "Constant", "params": {"cfr_const": 0.04}},
#     "C2": {"name": "Linear Decr.", "params": {"cfr_start": 0.05, "cfr_end": 0.03}},
#     "C3": {"name": "Sine Wave", "params": {"cfr_mean": 0.04, "amp": 0.02, "freq": 1.0}},
#     "C4": {"name": "Gaussian Kernel", "params": {"cfr_base": 0.03, "peak_h": 0.03, 
#                                                "peak_t_factor": 0.5, "peak_w_factor": 0.1}}
# }

intervention_types_params = {
    "I0": {"name": "K=0", "K": 0, "beta_abs": [], "lambda_": [], "times_factor": [], "signs": []},
    "I1": {"name": "K=1 (β<0)", "K": 1, "beta_abs": TRUE_BETA_ABS_K1, "lambda_": TRUE_LAMBDA_K1, "times_factor": TRUE_T_K1_FACTOR, "signs": SIGNS_K1},
    "I2": {"name": "K=2 (β1<0,β2>0)", "K": 2, "beta_abs": TRUE_BETA_ABS_K2, "lambda_": TRUE_LAMBDA_K2, "times_factor": TRUE_T_K2_FACTOR, "signs": SIGNS_K2}
}

scen_counter = 1
for cfr_code, cfr_data in cfr_types_params.items():
    for int_code, int_data in intervention_types_params.items():
        scenario_id = f"S{scen_counter:02d}"
        current_cfr_params = cfr_data["params"].copy()
        if cfr_code == "C4":
            # Peak position relative to analysis window
            current_cfr_params["peak_t"] = current_cfr_params["peak_t_factor"] * T_ANALYSIS_LENGTH 
            current_cfr_params["peak_w"] = current_cfr_params["peak_w_factor"] * T_ANALYSIS_LENGTH
        
        # Intervention times relative to analysis window
        current_int_times = [t_factor * T_ANALYSIS_LENGTH for t_factor in int_data["times_factor"]]

        SCENARIOS.append({
            "id": scenario_id,
            "cfr_type_code": cfr_code, "cfr_type_name": cfr_data["name"],
            "intervention_type_code": int_code, "intervention_type_name": int_data["name"],
            "cfr_params": current_cfr_params,
            "num_interventions_K_true": int_data["K"],
            "true_beta_abs_0": np.array(int_data["beta_abs"]),
            "true_lambda_0": np.array(int_data["lambda_"]),
            "true_intervention_times_0": np.array(current_int_times),
            "true_beta_signs_0": np.array(int_data["signs"])
        })
        scen_counter += 1

# ----------------------------------------------------------------------------
# MCMC and Output Configuration
# ----------------------------------------------------------------------------

# MCMC Parameters
NUM_WARMUP = 1000
NUM_SAMPLES = 1000
NUM_CHAINS = 1

# Output directories
OUTPUT_DIR_BASE = "./simulation_outputs/" 
OUTPUT_DIR_PLOTS = os.path.join(OUTPUT_DIR_BASE, "plots/")
OUTPUT_DIR_TABLES = os.path.join(OUTPUT_DIR_BASE, "tables/")
OUTPUT_DIR_RESULTS_CSV = os.path.join(OUTPUT_DIR_BASE, "results_csv/") # For aggregated and per-scenario CSVs
OUTPUT_DIR_POSTERIOR_SAMPLES = os.path.join(OUTPUT_DIR_BASE, "posterior_samples_raw/") # For full MCMC samples
OUTPUT_DIR_BENCHMARK_RESULTS = os.path.join(OUTPUT_DIR_BASE, "benchmarks_results/")
OUTPUT_DIR_POSTERIOR_SUMMARIES = os.path.join(OUTPUT_DIR_BASE, "posterior_summaries/") # For mean/quantiles
OUTPUT_DIR_RUN_METRICS_JSON = os.path.join(OUTPUT_DIR_BASE, "run_metrics_json/") # For individual run metrics (checkpointing)
