import numpy as np
import os
import json
import config

def save_raw_posterior_samples(scenario_id, mc_run_idx, posterior_samples, output_dir):
    """
    Saves the full MCMC posterior samples to a compressed .npz file.

    Args:
        scenario_id (str): The ID of the scenario.
        mc_run_idx (int): The index of the Monte Carlo run.
        posterior_samples (dict): A dictionary of posterior samples.
        output_dir (str): The directory to save the file in.
    """
    if not posterior_samples: return
    filename = f"scen_{scenario_id}_run_{mc_run_idx+1}_posterior_samples.npz"
    filepath = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    np.savez(filepath, **posterior_samples)

def load_raw_posterior_samples(scenario_id, mc_run_idx, input_dir):
    """
    Loads raw MCMC posterior samples from a compressed .npz file.

    Args:
        scenario_id (str): The ID of the scenario.
        mc_run_idx (int): The index of the Monte Carlo run.
        input_dir (str): The directory to load the file from.

    Returns:
        dict: A dictionary of posterior samples, or None if the file doesn't exist.
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

def save_posterior_summary_for_run(scenario_id, mc_run_idx, posterior_samples, output_dir):
    """
    Saves summary statistics of key posteriors for a single MC run.

    Args:
        scenario_id (str): The ID of the scenario.
        mc_run_idx (int): The index of the Monte Carlo run.
        posterior_samples (dict): A dictionary of posterior samples.
        output_dir (str): The directory to save the file in.
    """
    if not posterior_samples: return
    
    summary_data = {}
    for param_name in ["alpha", "gamma", "lambda", "p", "p_cf"]:
        if param_name in posterior_samples:
            samples = posterior_samples[param_name]
            summary_data[f"{param_name}_mean"] = np.mean(samples, axis=0)
            summary_data[f"{param_name}_q025"] = np.percentile(samples, 2.5, axis=0)
            summary_data[f"{param_name}_q975"] = np.percentile(samples, 97.5, axis=0)
            
    filename = f"scen_{scenario_id}_run_{mc_run_idx+1}_posterior_summary.npz"
    filepath = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    np.savez(filepath, **summary_data)

def load_posterior_summary_for_run(scenario_id, mc_run_idx, input_dir):
    """
    Loads posterior summary from an .npz file.

    Args:
        scenario_id (str): The ID of the scenario.
        mc_run_idx (int): The index of the Monte Carlo run.
        input_dir (str): The directory to load the file from.

    Returns:
        numpy.lib.npyio.NpzFile: The loaded summary data, or None if the file doesn't exist.
    """
    filename = f"scen_{scenario_id}_run_{mc_run_idx+1}_posterior_summary.npz"
    filepath = os.path.join(input_dir, filename)
    if os.path.exists(filepath):
        try:
            return np.load(filepath)
        except Exception as e:
            print(f"Error loading summary file {filepath}: {e}")
            return None
    return None

def save_benchmark_results(scenario_id, mc_run_idx, benchmark_results, output_dir):
    """
    Saves the results from all benchmark models to an .npz file.

    Args:
        scenario_id (str): The ID of the scenario.
        mc_run_idx (int): The index of the Monte Carlo run.
        benchmark_results (dict): A dictionary of benchmark results.
        output_dir (str): The directory to save the file in.
    """
    if not benchmark_results: return
    filename = f"scen_{scenario_id}_run_{mc_run_idx+1}_benchmark_results.npz"
    filepath = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    np.savez(filepath, **benchmark_results)

def load_benchmark_results(scenario_id, mc_run_idx, input_dir):
    """
    Loads the results for all benchmark models from an .npz file.

    Args:
        scenario_id (str): The ID of the scenario.
        mc_run_idx (int): The index of the Monte Carlo run.
        input_dir (str): The directory to load the file from.

    Returns:
        numpy.lib.npyio.NpzFile: The loaded benchmark results, or None if the file doesn't exist.
    """
    filename = f"scen_{scenario_id}_run_{mc_run_idx+1}_benchmark_results.npz"
    filepath = os.path.join(input_dir, filename)
    if os.path.exists(filepath):
        try:
            return np.load(filepath)
        except Exception as e:
            print(f"Error loading benchmark results file {filepath}: {e}")
            return None
    return None

def save_run_metrics(scenario_id, mc_run_idx, metrics_dict, output_dir):
    """
    Saves the calculated metrics for a single run to a JSON file.

    Args:
        scenario_id (str): The ID of the scenario.
        mc_run_idx (int): The index of the Monte Carlo run.
        metrics_dict (dict): A dictionary of calculated metrics.
        output_dir (str): The directory to save the file in.
    """
    filename = f"metrics_scen_{scenario_id}_run_{mc_run_idx+1}.json"
    filepath = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    
    def default_converter(o):
        if isinstance(o, np.generic): return o.item()
        raise TypeError
        
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=4, default=default_converter)

def load_run_metrics(scenario_id, mc_run_idx, input_dir):
    """
    Loads a metrics dictionary for a single run from a JSON file.

    Args:
        scenario_id (str): The ID of the scenario.
        mc_run_idx (int): The index of the Monte Carlo run.
        input_dir (str): The directory to load the file from.

    Returns:
        dict: The loaded metrics dictionary, or None if the file doesn't exist.
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
