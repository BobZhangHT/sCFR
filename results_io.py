# File: results_io.py
# Description: Functions for saving and loading detailed simulation outputs.

import numpy as np
import os
import json
import config
import evaluation 

def save_raw_posterior_samples(scenario_id, mc_run_idx, posterior_samples, output_dir):
    """
    Saves the full, raw MCMC posterior samples dictionary to a compressed .npz file.

    Args:
        scenario_id (str): The ID of the scenario (e.g., "S01").
        mc_run_idx (int): The index of the Monte Carlo run.
        posterior_samples (dict): The dictionary of samples from NumPyro.
        output_dir (str): The directory to save the file in.
    """
    if not posterior_samples: return
    filename = f"scen_{scenario_id}_run_{mc_run_idx+1}_posterior_samples.npz"
    filepath = os.path.join(output_dir, filename)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    np.savez(filepath, **posterior_samples)

def load_raw_posterior_samples(scenario_id, mc_run_idx, input_dir):
    """
    Loads raw MCMC posterior samples from a compressed .npz file.

    Args:
        scenario_id (str): The ID of the scenario.
        mc_run_idx (int): The index of the Monte Carlo run.
        input_dir (str): The directory where the file is located.

    Returns:
        dict or None: The dictionary of samples, or None if the file is not found or fails to load.
    """
    filename = f"scen_{scenario_id}_run_{mc_run_idx+1}_posterior_samples.npz"
    filepath = os.path.join(input_dir, filename)
    if os.path.exists(filepath):
        try:
            loaded_data = np.load(filepath, allow_pickle=True)
            return {key: loaded_data[key] for key in loaded_data}
        except Exception as e:
            print(f"Error loading raw samples {filepath}: {e}")
            return None
    return None

def save_run_metrics(scenario_id, mc_run_idx, metrics_dict, output_dir):
    """
    Saves the calculated metrics dictionary for a single run to a JSON file.
    It handles serialization of NumPy types to standard JSON types.

    Args:
        scenario_id (str): The ID of the scenario.
        mc_run_idx (int): The index of the Monte Carlo run.
        metrics_dict (dict): The dictionary of calculated metrics.
        output_dir (str): The directory to save the file in.
    """
    filename = f"metrics_scen_{scenario_id}_run_{mc_run_idx+1}.json"
    filepath = os.path.join(output_dir, filename)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    with open(filepath, 'w') as f:
        # Convert any remaining numpy types to native Python types for JSON compatibility
        serializable_metrics = {}
        for key, value in metrics_dict.items():
            if isinstance(value, np.generic):
                serializable_metrics[key] = value.item()
            else:
                serializable_metrics[key] = value
        json.dump(serializable_metrics, f, indent=4)

def load_run_metrics(scenario_id, mc_run_idx, input_dir):
    """
    Loads a metrics dictionary for a single run from a JSON file.

    Args:
        scenario_id (str): The ID of the scenario.
        mc_run_idx (int): The index of the Monte Carlo run.
        input_dir (str): The directory where the file is located.

    Returns:
        dict or None: The dictionary of metrics, or None if file not found or fails to load.
    """
    filename = f"metrics_scen_{scenario_id}_run_{mc_run_idx+1}.json"
    filepath = os.path.join(input_dir, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metrics file {filepath}: {e}")
            return None
    return None

def save_posterior_summary_for_run(scenario_id, mc_run_idx, posterior_samples, output_dir):
    """
    Calculates and saves summary statistics (mean, median, quantiles) of key posteriors
    for a single MC run to an .npz file. This is useful for post-hoc analysis without
    loading the full raw samples.

    Args:
        scenario_id (str): The ID of the scenario.
        mc_run_idx (int): The index of the Monte Carlo run.
        posterior_samples (dict): The dictionary of raw MCMC samples.
        output_dir (str): The directory to save the file in.
    """
    if not posterior_samples: return

    summary_data = {}
    
    # Parameters to summarize
    for param_name in ["alpha", "beta_abs", "lambda", "phi", "beta", "I", "M"]:
        if param_name in posterior_samples:
            samples = posterior_samples[param_name]
            summary_data[f"{param_name}_mean"] = np.mean(samples, axis=0)
            summary_data[f"{param_name}_q025"] = np.percentile(samples, 2.5, axis=0)
            summary_data[f"{param_name}_q975"] = np.percentile(samples, 97.5, axis=0)

    # Time series to summarize
    for ts_name in ["p", "p_cf", "eps"]:
        if ts_name in posterior_samples:
            samples = posterior_samples[ts_name]
            summary_data[f"{ts_name}_mean"] = np.mean(samples, axis=0)
            summary_data[f"{ts_name}_q025"] = np.percentile(samples, 2.5, axis=0)
            summary_data[f"{ts_name}_q975"] = np.percentile(samples, 97.5, axis=0)
            
    filename = f"scen_{scenario_id}_run_{mc_run_idx+1}_posterior_summary.npz"
    filepath = os.path.join(output_dir, filename)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    np.savez(filepath, **summary_data)

def load_posterior_summary_for_run(scenario_id, mc_run_idx, input_dir):
    """
    Loads posterior summary from an .npz file.

    Args:
        scenario_id (str): The ID of the scenario.
        mc_run_idx (int): The index of the Monte Carlo run.
        input_dir (str): The directory where the file is located.

    Returns:
        np.lib.npyio.NpzFile or None: The loaded summary object, or None if not found.
    """
    filename = f"scen_{scenario_id}_run_{mc_run_idx+1}_posterior_summary.npz"
    filepath = os.path.join(input_dir, filename)
    if os.path.exists(filepath):
        return np.load(filepath)
    else:
        # This warning is helpful during analysis to know if a run's summary is missing.
        print(f"Warning: Posterior summary file not found: {filepath}")
        return None