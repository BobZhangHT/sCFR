import os
import numpy as np

# --- General Simulation Parameters ---
NUM_MONTE_CARLO_RUNS = 5  # Number of simulation runs for each scenario
T_ANALYSIS_LENGTH = 200    # The length of the time series used for analysis
T_SIMULATION_BUFFER = 60   # Additional time points for simulation burn-in to avoid edge effects
T_SERIES_LENGTH_SIM = T_ANALYSIS_LENGTH + T_SIMULATION_BUFFER # Total simulation length
GLOBAL_BASE_SEED = 2025    # Base seed for global reproducibility
NUM_CORES_TO_USE = 5       # Number of CPU cores for parallel processing
OVERWRITE_EXISTING_RESULTS = True # If True, overwrite existing result files during a new run

# --- Data Generation Process (DGP) Parameters ---
# Onset-to-Death Distribution (f_s) using a Gamma distribution
F_MEAN = 15.43             # Mean of the delay distribution
F_SHAPE = 2.03             # Shape parameter of the delay distribution
F_DELAY_MAX = T_ANALYSIS_LENGTH + T_SIMULATION_BUFFER # Maximum delay considered

# B-spline Basis for generating the baseline CFR trend
N_SPLINE_KNOTS_J = 10      # Number of knots for the B-spline basis
SPLINE_ORDER = 4           # Order of the B-spline (e.g., 4 for cubic)

# I.i.d. Random Effect (delta_t) on logit-CFR
# NOTE: Set per-scenario via SIGMA_DELTA_SCENARIOS.

# Confirmed Cases (c_t) Generation
C_T_FUNCTION_TYPE = 'v_shape'      # Functional form of the case counts ('v_shape' or 'constant')
C_T_VSHAPE_MAX_CASES = 10000       # Maximum cases at the peak for 'v_shape'
C_T_VSHAPE_PEAK_TIME_FACTOR = 0.4  # Relative time of the peak for 'v_shape' (t_peak=0.4*T)
C_T_VSHAPE_SLOPE = 75              # Slope of the 'v_shape' function
C_T_CONSTANT_CASES = 500           # Number of cases for 'constant' type
MIN_DRAWN_CASES = 20               # Minimum number of cases on any day to ensure stability

# --- INTERVENTION PARAMETERS (Step + Hinge Effects) ---
TRUE_BETA_STEP_ABS_K1 = [0.6]
TRUE_BETA_SLOPE_ABS_K1 = [0.6]
TRUE_BETA_STEP_ABS_K2 = [0.6, 0.4]
TRUE_BETA_SLOPE_ABS_K2 = [0.6, 0.4]
TRUE_T_K1_FACTOR = [0.5]            # Relative timing for K=1 intervention
TRUE_T_K2_FACTOR = [0.33, 0.66]     # Relative timings for K=2 interventions
SIGNS_K1_STEP = [-1]
SIGNS_K1_SLOPE = [-1]
SIGNS_K2_STEP = [-1, 1]
SIGNS_K2_SLOPE = [-1, 1]

# Random effect scale per intervention setting
SIGMA_DELTA_SCENARIOS = {
    "I0": 0.20,
    "I1": 0.00,
    "I2": 0.10
}

# --- Scenario Definitions ---
SCENARIOS = []
# Define different baseline CFR patterns
cfr_types_params = {
    "C1": {"name": "Constant", "params": {"cfr_const": 0.02}},
    "C2": {"name": "Linear Decr.", "params": {"cfr_start": 0.03, "cfr_end": 0.01}},
    "C3": {"name": "Sine Wave", "params": {"cfr_mean": 0.02, "amp": 0.01, "freq": 1.0}},
    "C4": {"name": "Gaussian Kernel", "params": {"cfr_base": 0.01, "peak_h": 0.04,
                                               "peak_t_factor": 0.5, "peak_w_factor": 0.2}}
}

# Define different intervention settings (number of interventions, effects, timing)
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

# Systematically create all 12 scenarios by combining CFR patterns and intervention settings
scen_counter = 1
for cfr_code, cfr_data in cfr_types_params.items():
    for int_code, int_data in intervention_types_params.items():
        scenario_id = f"S{scen_counter:02d}"
        current_cfr_params = cfr_data["params"].copy()
        
        if cfr_code == "C4":
            current_cfr_params["peak_t"] = current_cfr_params["peak_t_factor"] * T_ANALYSIS_LENGTH
            current_cfr_params["peak_w"] = current_cfr_params["peak_w_factor"] * T_ANALYSIS_LENGTH

        # Calculate absolute intervention times from relative factors
        current_int_times = [t_factor * T_ANALYSIS_LENGTH for t_factor in int_data["times_factor"]]

        # Append the fully defined scenario to the list
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

# --- MCMC and Output Configuration ---
NUM_WARMUP = 1000          # Number of warmup steps for MCMC sampler
NUM_SAMPLES = 1000         # Number of posterior samples to draw in MCMC
NUM_CHAINS = 1             # Number of MCMC chains to run

# Define base directory and subdirectories for all outputs
OUTPUT_DIR_BASE = "./simulation_outputs/"
OUTPUT_DIR_PLOTS = os.path.join(OUTPUT_DIR_BASE, "plots/")
OUTPUT_DIR_TABLES = os.path.join(OUTPUT_DIR_BASE, "tables/")
OUTPUT_DIR_RESULTS_CSV = os.path.join(OUTPUT_DIR_BASE, "results_csv/")
OUTPUT_DIR_POSTERIOR_SAMPLES = os.path.join(OUTPUT_DIR_BASE, "posterior_samples_raw/")
OUTPUT_DIR_BENCHMARK_RESULTS = os.path.join(OUTPUT_DIR_BASE, "benchmarks_results/")
OUTPUT_DIR_POSTERIOR_SUMMARIES = os.path.join(OUTPUT_DIR_BASE, "posterior_summaries/")
OUTPUT_DIR_RUN_METRICS_JSON = os.path.join(OUTPUT_DIR_BASE, "run_metrics_json/")
OUTPUT_DIR_LOGS = os.path.join(OUTPUT_DIR_BASE, "logs/")
