"""
Unified Evaluation and Visualization Module for sCFR Project.

This module provides a deeply integrated interface for evaluating CFR estimation methods
and automatically generating corresponding visualizations. It combines metric calculation
with plotting functionality to provide a seamless analysis workflow.

Key Features:
- Unified evaluation and visualization workflow
- Comprehensive type hints for all functions
- Detailed docstrings following NumPy/Google style
- Robust exception handling
- Support for both single-run and aggregated analyses
- Backward compatibility with existing evaluation and plotting modules

Author: Refactored for deep integration
Date: 2025-12-25
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple, Optional, Union, Any, Literal
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid, logit

import config
import data_generation

METHOD_COLORS = {
    "sCFR": "tab:blue",
    "cCFR": "tab:green",
    "aCFR": "tab:red",
    "fsCFR": "tab:purple",
}


# =============================================================================
# Evaluation Functions (from merged evaluation.py)
# =============================================================================

def sanitize_metrics_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans a DataFrame by converting list-like values in object columns to scalars."""
    for col in df.columns:
        if df[col].dtype == 'object':
            is_list_like = df[col].notna().any() and isinstance(df[col].dropna().iloc[0], list)
            if is_list_like:
                df[col] = df[col].apply(
                    lambda x: x[0] if isinstance(x, list) and len(x) == 1 else (np.nan if isinstance(x, list) else x)
                ).astype(float, errors='ignore')
    return df

def get_posterior_estimates(
    posterior_samples: Dict[str, np.ndarray],
    param_name: str,
    percentiles: Tuple[float, float] = (2.5, 97.5)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract posterior estimates (mean, lower, upper) from posterior samples.
    
    Args:
        posterior_samples: Dictionary containing posterior samples.
        param_name: Name of the parameter to extract.
        percentiles: Tuple of (lower, upper) percentiles for credible interval.
    
    Returns:
        Tuple of (mean, lower, upper) estimates.
    
    Raises:
        KeyError: If param_name not found in posterior_samples.
    """
    samples = posterior_samples[param_name]
    
    mean = np.mean(samples, axis=0)
    lower = np.percentile(samples, percentiles[0], axis=0)
    upper = np.percentile(samples, percentiles[1], axis=0)
    
    return mean, lower, upper


def calculate_mae_rt(
    true_values: np.ndarray,
    estimated_values: np.ndarray
) -> float:
    """
    Calculate Mean Absolute Error (MAE) for CFR time series.
    
    Args:
        true_values: Array of true CFR values.
        estimated_values: Array of estimated CFR values.
    
    Returns:
        MAE value.
    """
    return float(np.mean(np.abs(true_values - estimated_values)))


def calculate_logit_mae(
    true_values: np.ndarray,
    estimated_values: np.ndarray
) -> float:
    """Calculate MAE on the logit scale for CFR time series."""
    eps = 1e-6
    true_clip = np.clip(true_values, eps, 1 - eps)
    est_clip = np.clip(estimated_values, eps, 1 - eps)
    return float(np.mean(np.abs(logit(est_clip) - logit(true_clip))))


def calculate_mciw_rt(
    lower_values: np.ndarray,
    upper_values: np.ndarray
) -> float:
    """
    Calculate Mean Credible Interval Width (MCIW) for CFR time series.
    
    Args:
        lower_values: Array of lower credible interval bounds.
        upper_values: Array of upper credible interval bounds.
    
    Returns:
        MCIW value.
    """
    return float(np.mean(upper_values - lower_values))


def calculate_mcic_rt(
    true_values: np.ndarray,
    lower_values: np.ndarray,
    upper_values: np.ndarray
) -> float:
    """
    Calculate Mean Credible Interval Coverage (MCIC) for CFR time series.
    
    Args:
        true_values: Array of true CFR values.
        lower_values: Array of lower credible interval bounds.
        upper_values: Array of upper credible interval bounds.
    
    Returns:
        MCIC value (proportion of true values within credible intervals).
    """
    within_interval = (true_values >= lower_values) & (true_values <= upper_values)
    return float(np.mean(within_interval))


def calculate_param_bias(
    true_value: Union[float, np.ndarray],
    estimated_value: Union[float, np.ndarray]
) -> float:
    """
    Calculate bias for a parameter estimate.
    
    Args:
        true_value: True parameter value.
        estimated_value: Estimated parameter value.
    
    Returns:
        Bias (estimated - true).
    """
    return float(np.mean(estimated_value) - np.mean(true_value))


def calculate_param_cri_width(
    lower_value: Union[float, np.ndarray],
    upper_value: Union[float, np.ndarray]
) -> float:
    """
    Calculate credible interval width for a parameter.
    
    Args:
        lower_value: Lower credible interval bound.
        upper_value: Upper credible interval bound.
    
    Returns:
        Credible interval width.
    """
    return float(np.mean(upper_value) - np.mean(lower_value))


def calculate_param_cri_coverage(
    true_value: Union[float, np.ndarray],
    lower_value: Union[float, np.ndarray],
    upper_value: Union[float, np.ndarray]
) -> bool:
    """
    Calculate whether true value is within credible interval for a parameter.
    
    Args:
        true_value: True parameter value.
        lower_value: Lower credible interval bound.
        upper_value: Upper credible interval bound.
    
    Returns:
        True if true value is within credible interval, False otherwise.
    """
    return bool((np.mean(true_value) >= np.mean(lower_value)) and 
                (np.mean(true_value) <= np.mean(upper_value)))


# =============================================================================
# Plotting Functions (from merged plotting.py)
# =============================================================================

def plot_cfr_timeseries_from_data(
    scenario_id: str,
    mc_run_idx: int,
    plot_data: Dict[str, Any],
    output_dir: str
) -> None:
    """
    Plot CFR time series from data dictionary.
    
    Args:
        scenario_id: Scenario identifier.
        mc_run_idx: Monte Carlo run index.
        plot_data: Dictionary containing plotting data.
        output_dir: Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    true_r_t = plot_data["true_r_t"]
    true_rcf_t = plot_data["true_rcf_0_t"]
    intervention_times = plot_data.get("true_intervention_times_0_abs", np.array([]))
    estimated_r_t_dict = plot_data["estimated_r_t_dict"]
    
    T = len(true_r_t)
    t_array = np.arange(T)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Factual CFR
    ax1 = axes[0]
    ax1.plot(t_array, true_r_t, 'k-', linewidth=2, label='True CFR', alpha=0.7)
    
    # Plot sCFR
    if "sCFR" in estimated_r_t_dict:
        sCFR_data = estimated_r_t_dict["sCFR"]
        ax1.plot(t_array, sCFR_data["mean"], 'b-', linewidth=2, label='sCFR', alpha=0.8)
        ax1.fill_between(t_array, sCFR_data["lower"], sCFR_data["upper"], 
                       color='blue', alpha=0.2)
    
    # Plot cCFR
    if "cCFR_model" in estimated_r_t_dict:
        cCFR_data = estimated_r_t_dict["cCFR_model"]
        ax1.plot(t_array, cCFR_data["mean"], 'g--', linewidth=2, label='cCFR', alpha=0.8)
    
    # Plot aCFR
    if "aCFR_model" in estimated_r_t_dict:
        aCFR_data = estimated_r_t_dict["aCFR_model"]
        ax1.plot(t_array, aCFR_data["mean"], 'r--', linewidth=2, label='aCFR', alpha=0.8)
    
    # Plot fsCFR
    if "fsCFR_model" in estimated_r_t_dict:
        its_data = estimated_r_t_dict["fsCFR_model"]
        ax1.plot(t_array, its_data["factual_mean"], 'm:', linewidth=2, label='fsCFR', alpha=0.8)
    
    # Mark interventions
    for t_int in intervention_times:
        if 0 <= t_int < T:
            ax1.axvline(x=t_int, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Case Fatality Rate')
    ax1.set_title(f'{scenario_id} - Run {mc_run_idx}: Factual CFR Estimates')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Counterfactual CFR
    ax2 = axes[1]
    ax2.plot(t_array, true_rcf_t, 'k-', linewidth=2, label='True Counterfactual CFR', alpha=0.7)
    
    # Plot sCFR counterfactual
    if "sCFR" in estimated_r_t_dict and "cf_mean" in estimated_r_t_dict["sCFR"]:
        sCFR_cf = estimated_r_t_dict["sCFR"]
        ax2.plot(t_array, sCFR_cf["cf_mean"], 'b-', linewidth=2, label='sCFR Counterfactual', alpha=0.8)
        ax2.fill_between(t_array, sCFR_cf["cf_lower"], sCFR_cf["cf_upper"], 
                       color='blue', alpha=0.2)
    
    # Plot fsCFR counterfactual
    if "fsCFR_model" in estimated_r_t_dict:
        its_cf = estimated_r_t_dict["fsCFR_model"]
        ax2.plot(t_array, its_cf["cf_mean"], 'm:', linewidth=2, label='fsCFR Counterfactual', alpha=0.8)
    
    # Mark interventions
    for t_int in intervention_times:
        if 0 <= t_int < T:
            ax2.axvline(x=t_int, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Case Fatality Rate')
    ax2.set_title(f'{scenario_id} - Run {mc_run_idx}: Counterfactual CFR Estimates')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{scenario_id}_run_{mc_run_idx}_timeseries.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_path_pdf = os.path.join(output_dir, f"{scenario_id}_run_{mc_run_idx}_timeseries.pdf")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    plt.close()


def plot_aggregated_factual_summary(
    aggregated_plot_data: list,
    output_dir: str
) -> None:
    """
    Plot aggregated factual summary across all scenarios.
    
    Args:
        aggregated_plot_data: List of aggregated plot data dictionaries.
        output_dir: Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cfr_codes = list(config.cfr_types_params.keys())
    int_codes = list(config.intervention_types_params.keys())
    scenario_meta = {s["id"]: s for s in config.SCENARIOS}
    
    fig, axes = plt.subplots(len(cfr_codes), len(int_codes), figsize=(6*len(int_codes), 4.8*len(cfr_codes)))
    if len(cfr_codes) == 1 and len(int_codes) == 1:
        axes = np.array([[axes]])
    elif len(cfr_codes) == 1:
        axes = axes[np.newaxis, :]
    elif len(int_codes) == 1:
        axes = axes[:, np.newaxis]
    
    for plot_dict in aggregated_plot_data:
        scenario_id = plot_dict["scenario_id"]
        meta = scenario_meta.get(scenario_id, {})
        cfr_code = meta.get("cfr_type_code")
        int_code = meta.get("intervention_type_code")
        if cfr_code not in cfr_codes or int_code not in int_codes:
            continue
        ax = axes[cfr_codes.index(cfr_code), int_codes.index(int_code)]
        true_r_t = plot_dict["true_r_t"]
        intervention_times = plot_dict.get("true_intervention_times_0_abs", np.array([]))
        estimated_r_t_dict = plot_dict["estimated_r_t_dict"]
        T = len(true_r_t)
        t_array = np.arange(T)
        
        ax.plot(t_array, true_r_t, 'k-', linewidth=2.5, label='True', alpha=0.8)
        
        if "sCFR" in estimated_r_t_dict:
            sCFR_data = estimated_r_t_dict["sCFR"]
            if "mean" in sCFR_data and len(sCFR_data["mean"]) == T:
                ax.plot(t_array, sCFR_data["mean"], color=METHOD_COLORS["sCFR"],
                        linewidth=2.5, label='sCFR', alpha=0.85)
                if "lower" in sCFR_data and "upper" in sCFR_data and len(sCFR_data["lower"]) == T and len(sCFR_data["upper"]) == T:
                    ax.fill_between(t_array, sCFR_data["lower"], sCFR_data["upper"],
                                    color=METHOD_COLORS["sCFR"], alpha=0.2)
        
        if "cCFR_model" in estimated_r_t_dict:
            cCFR_data = estimated_r_t_dict["cCFR_model"]
            if "mean" in cCFR_data and len(cCFR_data["mean"]) == T:
                ax.plot(t_array, cCFR_data["mean"], linestyle='--',
                        color=METHOD_COLORS["cCFR"], linewidth=2.0, label='cCFR', alpha=0.75)
        
        if "aCFR_model" in estimated_r_t_dict:
            aCFR_data = estimated_r_t_dict["aCFR_model"]
            if "mean" in aCFR_data and len(aCFR_data["mean"]) == T:
                ax.plot(t_array, aCFR_data["mean"], linestyle='--',
                        color=METHOD_COLORS["aCFR"], linewidth=2.0, label='aCFR', alpha=0.75)
        
        if "fsCFR_model" in estimated_r_t_dict:
            its_data = estimated_r_t_dict["fsCFR_model"]
            if "factual_mean" in its_data and len(its_data["factual_mean"]) == T:
                ax.plot(t_array, its_data["factual_mean"], linestyle=':',
                        color=METHOD_COLORS["fsCFR"], linewidth=2.2, label='fsCFR', alpha=0.9)
        
        for t_int in intervention_times:
            if 0 <= t_int < T:
                ax.axvline(x=t_int, color='orange', linestyle='--', linewidth=1.5, alpha=0.6)
        
        ax.set_title(f'{scenario_id}', fontsize=14)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('CFR', fontsize=12)
        if cfr_codes.index(cfr_code) == 0 and int_codes.index(int_code) == 0:
            ax.legend(loc='best', fontsize=10)
        else:
            ax.legend().remove()
        ax.grid(True, alpha=0.3)

    for row_idx, cfr_code in enumerate(cfr_codes):
        row_label = config.cfr_types_params[cfr_code]["name"]
        axes[row_idx, 0].set_ylabel(f"{row_label}\nCFR", fontsize=12)
    for col_idx, int_code in enumerate(int_codes):
        col_label = config.intervention_types_params[int_code]["name"]
        axes[0, col_idx].set_title(f"{col_label}", fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "aggregated_factual_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_path_pdf = os.path.join(output_dir, "aggregated_factual_summary.pdf")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    plt.close()


def plot_aggregated_counterfactual_summary(
    aggregated_plot_data: list,
    output_dir: str
) -> None:
    """
    Plot aggregated counterfactual summary across all scenarios.
    
    Args:
        aggregated_plot_data: List of aggregated plot data dictionaries.
        output_dir: Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cfr_codes = list(config.cfr_types_params.keys())
    int_codes = list(config.intervention_types_params.keys())
    scenario_meta = {s["id"]: s for s in config.SCENARIOS}
    
    fig, axes = plt.subplots(len(cfr_codes), len(int_codes), figsize=(6*len(int_codes), 4.8*len(cfr_codes)))
    if len(cfr_codes) == 1 and len(int_codes) == 1:
        axes = np.array([[axes]])
    elif len(cfr_codes) == 1:
        axes = axes[np.newaxis, :]
    elif len(int_codes) == 1:
        axes = axes[:, np.newaxis]
    
    for plot_dict in aggregated_plot_data:
        scenario_id = plot_dict["scenario_id"]
        meta = scenario_meta.get(scenario_id, {})
        cfr_code = meta.get("cfr_type_code")
        int_code = meta.get("intervention_type_code")
        if cfr_code not in cfr_codes or int_code not in int_codes:
            continue
        ax = axes[cfr_codes.index(cfr_code), int_codes.index(int_code)]
        true_rcf_t = plot_dict["true_rcf_0_t"]
        intervention_times = plot_dict.get("true_intervention_times_0_abs", np.array([]))
        estimated_r_t_dict = plot_dict["estimated_r_t_dict"]
        T = len(true_rcf_t)
        t_array = np.arange(T)
        
        ax.plot(t_array, true_rcf_t, 'k-', linewidth=2.5, label='True CF', alpha=0.8)
        
        if "sCFR" in estimated_r_t_dict and "cf_mean" in estimated_r_t_dict["sCFR"]:
            sCFR_cf = estimated_r_t_dict["sCFR"]
            if "cf_mean" in sCFR_cf and len(sCFR_cf["cf_mean"]) == T:
                ax.plot(t_array, sCFR_cf["cf_mean"], color=METHOD_COLORS["sCFR"],
                        linewidth=2.5, label='sCFR CF', alpha=0.85)
                if "cf_lower" in sCFR_cf and "cf_upper" in sCFR_cf and len(sCFR_cf["cf_lower"]) == T and len(sCFR_cf["cf_upper"]) == T:
                    ax.fill_between(t_array, sCFR_cf["cf_lower"], sCFR_cf["cf_upper"],
                                    color=METHOD_COLORS["sCFR"], alpha=0.2)
        
        if "fsCFR_model" in estimated_r_t_dict:
            its_cf = estimated_r_t_dict["fsCFR_model"]
            if "cf_mean" in its_cf and len(its_cf["cf_mean"]) == T:
                ax.plot(t_array, its_cf["cf_mean"], linestyle=':',
                        color=METHOD_COLORS["fsCFR"], linewidth=2.2, label='fsCFR CF', alpha=0.9)
        
        for t_int in intervention_times:
            if 0 <= t_int < T:
                ax.axvline(x=t_int, color='orange', linestyle='--', linewidth=1.5, alpha=0.6)
        
        ax.set_title(f'{scenario_id}', fontsize=14)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('CFR', fontsize=12)
        if cfr_codes.index(cfr_code) == 0 and int_codes.index(int_code) == 0:
            ax.legend(loc='best', fontsize=10)
        else:
            ax.legend().remove()
        ax.grid(True, alpha=0.3)

    for row_idx, cfr_code in enumerate(cfr_codes):
        row_label = config.cfr_types_params[cfr_code]["name"]
        axes[row_idx, 0].set_ylabel(f"{row_label}\nCFR", fontsize=12)
    for col_idx, int_code in enumerate(int_codes):
        col_label = config.intervention_types_params[int_code]["name"]
        axes[0, col_idx].set_title(f"{col_label}", fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "aggregated_counterfactual_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_path_pdf = os.path.join(output_dir, "aggregated_counterfactual_summary.pdf")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    plt.close()


def plot_metric_summary_boxplots(
    results_df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Plot summary boxplots of evaluation metrics.
    
    Args:
        results_df: DataFrame containing evaluation results.
        output_dir: Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    mae_cols = [col for col in results_df.columns if 'mae_rt_logit' in col]
    if mae_cols:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        
        data_to_plot = []
        labels = []
        for col in mae_cols:
            model = col.split('_')[-1] if col.split('_')[-1] != 'rt' else '_'.join(col.split('_')[-2:])
            data_to_plot.append(results_df[col].dropna().values)
            labels.append(model)
        
        bp = axes.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        axes.set_ylabel('MAE (rt)', fontsize=12)
        axes.set_title('Mean Absolute Error by Model', fontsize=14)
        axes.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        output_path = os.path.join(output_dir, "metric_summary_boxplots_mae.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        output_path_pdf = os.path.join(output_dir, "metric_summary_boxplots_mae.pdf")
        plt.savefig(output_path_pdf, bbox_inches='tight')
        plt.close()


def plot_combined_metrics_summary(
    results_df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Plot combined metrics summary and export beta_abs metric CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    results_df = sanitize_metrics_dataframe(results_df.copy())
    if "error" in results_df.columns:
        results_df = results_df[results_df["error"].isin([None, "None"])]

    def boxplot_data(columns):
        data = []
        labels = []
        for label, col in columns:
            if col in results_df.columns:
                data.append(results_df[col].dropna().values)
                labels.append(label)
        return data, labels

    beta_cols = ["beta_abs_mae_sCFR", "beta_abs_coverage_sCFR", "beta_abs_mae_fsCFR"]
    beta_df = results_df[["scenario_id"] + [c for c in beta_cols if c in results_df.columns]].copy()
    beta_summary = beta_df.groupby("scenario_id", dropna=False).mean(numeric_only=True).reset_index()
    beta_csv_path = os.path.join(output_dir, "beta_abs_metrics_summary.csv")
    beta_summary.to_csv(beta_csv_path, index=False)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.35, wspace=0.3)

    ax_main = fig.add_subplot(gs[0, :])
    data_main, labels_main = boxplot_data([
        ("sCFR", "mae_rt_logit_sCFR"),
        ("cCFR", "mae_rt_logit_cCFR"),
        ("aCFR", "mae_rt_logit_aCFR"),
        ("fsCFR", "mae_rt_logit_fsCFR"),
    ])
    ax_main.boxplot(data_main, labels=labels_main, patch_artist=True)
    ax_main.set_ylabel("Logit MAE (r_F)")
    ax_main.set_title("Factual CFR MAE (Logit Scale)")
    ax_main.grid(True, alpha=0.3, axis="y")

    ax_cf = fig.add_subplot(gs[1, 0])
    data_cf, labels_cf = boxplot_data([
        ("sCFR", "mae_rcf_logit_sCFR"),
        ("fsCFR", "mae_rcf_logit_fsCFR"),
    ])
    ax_cf.boxplot(data_cf, labels=labels_cf, patch_artist=True)
    ax_cf.set_ylabel("Logit MAE (r_CF)")
    ax_cf.set_title("Counterfactual CFR MAE")
    ax_cf.grid(True, alpha=0.3, axis="y")

    ax_base = fig.add_subplot(gs[1, 1])
    data_base, labels_base = boxplot_data([
        ("sCFR", "mae_baseline_logit_sCFR"),
        ("fsCFR", "mae_baseline_logit_fsCFR"),
    ])
    ax_base.boxplot(data_base, labels=labels_base, patch_artist=True)
    ax_base.set_ylabel("MAE (Baseline Logit)")
    ax_base.set_title("Baseline Effect MAE")
    ax_base.grid(True, alpha=0.3, axis="y")

    ax_rand = fig.add_subplot(gs[1, 2])
    data_rand, labels_rand = boxplot_data([
        ("sCFR", "mae_random_logit_sCFR"),
        ("fsCFR", "mae_random_logit_fsCFR"),
    ])
    ax_rand.boxplot(data_rand, labels=labels_rand, patch_artist=True)
    ax_rand.set_ylabel("MAE (Random Effect)")
    ax_rand.set_title("Random Effect MAE")
    ax_rand.grid(True, alpha=0.3, axis="y")

    output_path = os.path.join(output_dir, "combined_metrics_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_path_pdf = os.path.join(output_dir, "combined_metrics_summary.pdf")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    plt.close()


class EstimatorType(Enum):
    """Enumeration of supported CFR estimator types."""
    S_CFR = "sCFR"
    C_CFR = "cCFR"
    A_CFR = "aCFR"
    fsCFR = "fsCFR"


class MetricType(Enum):
    """Enumeration of supported metric types."""
    MAE = "mae"  # Mean Absolute Error
    MCIW = "mciw"  # Mean Credible Interval Width
    MCIC = "mcic"  # Mean Credible Interval Coverage
    BIAS = "bias"  # Parameter bias
    WIDTH = "width"  # Credible interval width
    COVER = "cover"  # Credible interval coverage


class PlotType(Enum):
    """Enumeration of supported plot types."""
    TIME_SERIES = "timeseries"
    AGGREGATED_FACTUAL = "aggregated_factual"
    AGGREGATED_COUNTERFACTUAL = "aggregated_counterfactual"
    METRIC_BOXPLOT = "metric_boxplot"
    COMBINED_METRICS = "combined_metrics"


# =============================================================================
# Data Classes for Structured Results
# =============================================================================

@dataclass
class PosteriorEstimates:
    """
    Container for posterior estimates with credible intervals.
    
    Attributes:
        mean: Mean estimate across posterior samples.
        lower: Lower bound of credible interval.
        upper: Upper bound of credible interval.
        samples: Raw posterior samples (optional).
    """
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    samples: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate array shapes after initialization."""
        if not (self.mean.shape == self.lower.shape == self.upper.shape):
            raise ValueError(
                f"Shape mismatch: mean {self.mean.shape}, "
                f"lower {self.lower.shape}, upper {self.upper.shape}"
            )


@dataclass
class ModelEvaluationResult:
    """
    Complete evaluation result for a single model.
    
    Attributes:
        model_name: Name of the model being evaluated.
        factual_estimates: Estimates for factual CFR.
        counterfactual_estimates: Estimates for counterfactual CFR (optional).
        metrics: Dictionary of calculated metrics.
        parameter_estimates: Dictionary of parameter estimates (optional).
    """
    model_name: str
    factual_estimates: PosteriorEstimates
    counterfactual_estimates: Optional[PosteriorEstimates] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    parameter_estimates: Optional[Dict[str, PosteriorEstimates]] = None
    
    def get_metric(self, metric_name: str) -> float:
        """Safely retrieve a metric value."""
        return self.metrics.get(metric_name, np.nan)


@dataclass
class ScenarioEvaluationResult:
    """
    Complete evaluation result for a single scenario.
    
    Attributes:
        scenario_id: Identifier for the scenario.
        run_seed: Random seed used for the simulation.
        true_factual_cfr: True factual CFR time series.
        true_counterfactual_cfr: True counterfactual CFR time series.
        intervention_times: Times of interventions.
        model_results: Dictionary mapping model names to their evaluation results.
        plot_data: Dictionary containing data for plotting.
    """
    scenario_id: str
    run_seed: int
    true_factual_cfr: np.ndarray
    true_counterfactual_cfr: np.ndarray
    intervention_times: np.ndarray
    model_results: Dict[str, ModelEvaluationResult] = field(default_factory=dict)
    plot_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedEvaluationResult:
    """
    Aggregated evaluation results across multiple runs or scenarios.
    
    Attributes:
        results_df: DataFrame containing all metrics across runs.
        scenario_results: Dictionary mapping scenario IDs to their results.
        aggregated_plot_data: List of aggregated plot data dictionaries.
    """
    results_df: pd.DataFrame
    scenario_results: Dict[str, List[ScenarioEvaluationResult]] = field(default_factory=dict)
    aggregated_plot_data: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# Main Evaluation and Visualization Class
# =============================================================================

class CFREvaluatorVisualizer:
    """
    Unified interface for evaluating CFR estimation methods and generating visualizations.
    
    This class provides a seamless workflow for:
    1. Calculating evaluation metrics for different CFR estimation methods
    2. Generating comprehensive visualizations
    3. Handling both single-run and aggregated analyses
    
    The class maintains backward compatibility with existing evaluation and plotting
    modules while providing a more integrated and user-friendly interface.
    
    Examples:
        >>> evaluator = CFREvaluatorVisualizer(output_dir="./outputs")
        >>> result = evaluator.evaluate_single_run(sim_data, posterior_scfr, benchmarks_r_t, benchmark_cis, its_results)
        >>> evaluator.plot_single_run(result, save_plots=True)
    """
    
    def __init__(
        self,
        output_dir: str = "./simulation_outputs/plots",
        percentiles: Tuple[float, float] = (2.5, 97.5),
        figsize: Tuple[int, int] = (12, 7),
        dpi: int = 300
    ):
        """
        Initialize the CFREvaluatorVisualizer.
        
        Args:
            output_dir: Directory to save generated plots.
            percentiles: Percentiles for credible intervals (lower, upper).
            figsize: Default figure size for plots.
            dpi: Resolution for saved figures.
        """
        self.output_dir = output_dir
        self.percentiles = percentiles
        self.figsize = figsize
        self.dpi = dpi
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configure matplotlib style
        self._configure_plotting_style()
    
    def _configure_plotting_style(self) -> None:
        """Configure matplotlib and seaborn plotting styles."""
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 20,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'figure.dpi': 150,
            'savefig.dpi': self.dpi,
            'lines.linewidth': 2.0,
            'axes.linewidth': 1.5,
        })
    
    # =========================================================================
    # Evaluation Methods
    # =========================================================================
    
    def evaluate_single_run(
        self,
        sim_data: Dict[str, Any],
        posterior_scfr: Optional[Dict[str, np.ndarray]] = None,
        benchmarks_r_t: Optional[Dict[str, np.ndarray]] = None,
        benchmark_cis: Optional[Dict[str, np.ndarray]] = None,
        its_results: Optional[Dict[str, np.ndarray]] = None
    ) -> ScenarioEvaluationResult:
        """
        Evaluate a single simulation run and prepare data for visualization.
        
        This method calculates all relevant metrics for each CFR estimation method
        and structures the results for easy plotting.
        
        Args:
            sim_data: Dictionary containing simulation data including true values.
            posterior_scfr: Posterior samples from the sCFR model.
            benchmarks_r_t: Dictionary of estimated CFR from benchmark methods.
            benchmark_cis: Dictionary of credible intervals for benchmarks.
            its_results: Results from the Interrupted Time Series model.
        
        Returns:
            ScenarioEvaluationResult containing all evaluation results and plot data.
        
        Raises:
            ValueError: If required simulation data is missing.
            KeyError: If expected keys are not found in input dictionaries.
        """
        try:
            # Validate input
            self._validate_simulation_data(sim_data)
            
            # Extract basic information
            scenario_id = sim_data["scenario_id"]
            run_seed = sim_data["run_seed"]
            T_analyze = config.T_ANALYSIS_LENGTH
            
            # Extract true values
            true_r_t = sim_data["true_r_0_t"][:T_analyze]
            true_rcf_t = sim_data["true_rcf_0_t"][:T_analyze]
            intervention_times = sim_data.get("true_intervention_times_0_abs", np.array([]))
            
            # Initialize result container
            result = ScenarioEvaluationResult(
                scenario_id=scenario_id,
                run_seed=run_seed,
                true_factual_cfr=true_r_t,
                true_counterfactual_cfr=true_rcf_t,
                intervention_times=intervention_times
            )
            
            # Evaluate sCFR model
            if posterior_scfr:
                result.model_results["sCFR"] = self._evaluate_scfr(
                    posterior_scfr, true_r_t, true_rcf_t, T_analyze, sim_data
                )
            
            # Evaluate benchmark methods
            if benchmarks_r_t:
                if benchmark_cis is None:
                    benchmark_cis = {}
                result.model_results["cCFR"] = self._evaluate_benchmark(
                    "cCFR", benchmarks_r_t, benchmark_cis, true_r_t, T_analyze
                )
                result.model_results["aCFR"] = self._evaluate_benchmark(
                    "aCFR", benchmarks_r_t, benchmark_cis, true_r_t, T_analyze
                )
            
            # Evaluate fsCFR model
            if its_results:
                result.model_results["fsCFR"] = self._evaluate_its(
                    its_results, true_r_t, true_rcf_t, T_analyze, sim_data
                )
            
            # Prepare plot data
            result.plot_data = self._prepare_plot_data(
                result, benchmarks_r_t, benchmark_cis, its_results
            )
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error evaluating single run: {str(e)}") from e
    
    def _validate_simulation_data(self, sim_data: Dict[str, Any]) -> None:
        """Validate that required simulation data is present."""
        required_keys = [
            "scenario_id", "run_seed", "true_r_0_t", "true_rcf_0_t",
            "true_intervention_times_0_abs", "num_interventions_true_K",
            "true_zeta_0_t", "true_eta_0_t"
        ]
        missing_keys = [k for k in required_keys if k not in sim_data]
        if missing_keys:
            raise ValueError(f"Missing required simulation data keys: {missing_keys}")
    
    def _evaluate_scfr(
        self,
        posterior_scfr: Dict[str, np.ndarray],
        true_r_t: np.ndarray,
        true_rcf_t: np.ndarray,
        T_analyze: int,
        sim_data: Dict[str, Any]
    ) -> ModelEvaluationResult:
        """Evaluate the sCFR model."""
        # Get posterior estimates
        r_t_key = "r_t" if "r_t" in posterior_scfr else "p"
        r_cf_key = "r_cf" if "r_cf" in posterior_scfr else "p_cf"
        r_t_mean, r_t_lower, r_t_upper = get_posterior_estimates(
            posterior_scfr, r_t_key, self.percentiles
        )
        rcf_t_mean, rcf_t_lower, rcf_t_upper = get_posterior_estimates(
            posterior_scfr, r_cf_key, self.percentiles
        )
        
        # Slice to analysis length
        r_t_mean, r_t_lower, r_t_upper = (
            r_t_mean[:T_analyze], r_t_lower[:T_analyze], r_t_upper[:T_analyze]
        )
        rcf_t_mean, rcf_t_lower, rcf_t_upper = (
            rcf_t_mean[:T_analyze], rcf_t_lower[:T_analyze], rcf_t_upper[:T_analyze]
        )
        
        # Calculate metrics (logit scale)
        metrics = {
            "mae_rt_logit": float(calculate_logit_mae(true_r_t, r_t_mean)),
            "mciw_rt": float(calculate_mciw_rt(r_t_lower, r_t_upper)),
            "mcic_rt": float(calculate_mcic_rt(true_r_t, r_t_lower, r_t_upper)),
            "mae_rcf_logit": float(calculate_logit_mae(true_rcf_t, rcf_t_mean)),
            "mciw_rcf": float(calculate_mciw_rt(rcf_t_lower, rcf_t_upper)),
            "mcic_rcf": float(calculate_mcic_rt(true_rcf_t, rcf_t_lower, rcf_t_upper)),
        }

        # Baseline and random effect MAE (logit scale)
        true_zeta = sim_data["true_zeta_0_t"][:T_analyze]
        true_eta = sim_data["true_eta_0_t"][:T_analyze]
        if "baseline_logit" in posterior_scfr:
            baseline_mean = np.mean(posterior_scfr["baseline_logit"], axis=0)[:T_analyze]
            metrics["mae_baseline_logit"] = float(np.mean(np.abs(baseline_mean - true_zeta)))
        if "delta" in posterior_scfr:
            delta_mean = np.mean(posterior_scfr["delta"], axis=0)[:T_analyze]
            metrics["mae_random_logit"] = float(np.mean(np.abs(delta_mean - true_eta)))
        
        # Get parameter estimates
        param_estimates = self._extract_parameter_estimates(posterior_scfr)

        # Beta_abs MAE and coverage (sCFR only)
        if "beta_abs" in posterior_scfr:
            beta_mean = np.mean(posterior_scfr["beta_abs"], axis=0)
            beta_lower = np.percentile(posterior_scfr["beta_abs"], self.percentiles[0], axis=0)
            beta_upper = np.percentile(posterior_scfr["beta_abs"], self.percentiles[1], axis=0)
            true_beta_abs = sim_data.get("true_beta_abs_0", np.array([]))
            if len(true_beta_abs) > 0:
                metrics["beta_abs_mae"] = float(np.mean(np.abs(beta_mean - true_beta_abs)))
                coverage = (true_beta_abs >= beta_lower) & (true_beta_abs <= beta_upper)
                metrics["beta_abs_coverage"] = float(np.mean(coverage))
        
        return ModelEvaluationResult(
            model_name="sCFR",
            factual_estimates=PosteriorEstimates(r_t_mean, r_t_lower, r_t_upper),
            counterfactual_estimates=PosteriorEstimates(rcf_t_mean, rcf_t_lower, rcf_t_upper),
            metrics=metrics,
            parameter_estimates=param_estimates
        )
    
    def _evaluate_benchmark(
        self,
        model_name: str,
        benchmarks_r_t: Dict[str, np.ndarray],
        benchmark_cis: Dict[str, np.ndarray],
        true_r_t: np.ndarray,
        T_analyze: int
    ) -> ModelEvaluationResult:
        """Evaluate a benchmark CFR method."""
        # Get the appropriate key for the benchmark
        if model_name == "cCFR":
            r_t_key = "cCFR_model"
        elif model_name == "aCFR":
            r_t_key = "aCFR_model"
        else:
            raise ValueError(f"Unknown benchmark model: {model_name}")
        
        # Get estimates
        r_t_mean = benchmarks_r_t[r_t_key][:T_analyze]
        r_t_lower = benchmark_cis.get(f"{r_t_key}_lower", r_t_mean)[:T_analyze]
        r_t_upper = benchmark_cis.get(f"{r_t_key}_upper", r_t_mean)[:T_analyze]
        
        # Calculate metrics (logit scale)
        metrics = {
            "mae_rt_logit": float(calculate_logit_mae(true_r_t, r_t_mean)),
        }
        
        return ModelEvaluationResult(
            model_name=model_name,
            factual_estimates=PosteriorEstimates(r_t_mean, r_t_lower, r_t_upper),
            metrics=metrics
        )
    
    def _evaluate_its(
        self,
        its_results: Dict[str, np.ndarray],
        true_r_t: np.ndarray,
        true_rcf_t: np.ndarray,
        T_analyze: int,
        sim_data: Dict[str, Any]
    ) -> ModelEvaluationResult:
        """Evaluate the fsCFR model."""
        # Get factual estimates
        factual_mean = its_results["fsCFR_factual_mean"][:T_analyze]
        factual_lower = its_results.get("fsCFR_factual_lower", factual_mean)[:T_analyze]
        factual_upper = its_results.get("fsCFR_factual_upper", factual_mean)[:T_analyze]
        
        # Get counterfactual estimates
        cf_mean = its_results["fsCFR_counterfactual_mean"][:T_analyze]
        cf_lower = its_results.get("fsCFR_counterfactual_lower", cf_mean)[:T_analyze]
        cf_upper = its_results.get("fsCFR_counterfactual_upper", cf_mean)[:T_analyze]
        
        # Calculate metrics (logit scale). fsCFR provides point estimates only.
        metrics = {
            "mae_rt_logit": float(calculate_logit_mae(true_r_t, factual_mean)),
            "mae_rcf_logit": float(calculate_logit_mae(true_rcf_t, cf_mean)),
        }

        # Baseline and random effect MAE (logit scale)
        true_zeta = sim_data["true_zeta_0_t"][:T_analyze]
        true_eta = sim_data["true_eta_0_t"][:T_analyze]
        if "fsCFR_baseline_logit" in its_results:
            baseline_hat = its_results["fsCFR_baseline_logit"][:T_analyze]
            metrics["mae_baseline_logit"] = float(np.mean(np.abs(baseline_hat - true_zeta)))
        if "fsCFR_delta" in its_results:
            delta_hat = its_results["fsCFR_delta"][:T_analyze]
            metrics["mae_random_logit"] = float(np.mean(np.abs(delta_hat - true_eta)))
        
        # Add parameter metrics if interventions exist
        num_interventions = sim_data["num_interventions_true_K"]
        if num_interventions > 0:
            true_beta_abs = sim_data["true_beta_abs_0"]
            
            for k in range(num_interventions):
                if k < len(its_results.get("fsCFR_beta_abs_est", [])):
                    metrics[f"bias_gamma_{k+1}"] = float(
                        calculate_param_bias(
                            true_beta_abs[k], its_results["fsCFR_beta_abs_est"][k]
                        )
                    )
            if len(its_results.get("fsCFR_beta_abs_est", [])) > 0:
                est_beta_abs = its_results["fsCFR_beta_abs_est"]
                metrics["beta_abs_mae"] = float(np.mean(np.abs(est_beta_abs - true_beta_abs)))
        
        return ModelEvaluationResult(
            model_name="fsCFR",
            factual_estimates=PosteriorEstimates(factual_mean, factual_lower, factual_upper),
            counterfactual_estimates=PosteriorEstimates(cf_mean, cf_lower, cf_upper),
            metrics=metrics
        )
    
    def _extract_parameter_estimates(
        self,
        posterior_scfr: Dict[str, np.ndarray]
    ) -> Optional[Dict[str, PosteriorEstimates]]:
        """Extract parameter estimates from posterior samples."""
        param_estimates = {}
        
        # Beta parameters
        if "beta_abs" in posterior_scfr:
            beta_mean, beta_lower, beta_upper = get_posterior_estimates(
                posterior_scfr, "beta_abs", self.percentiles
            )
            param_estimates["beta_abs"] = PosteriorEstimates(
                np.atleast_1d(beta_mean),
                np.atleast_1d(beta_lower),
                np.atleast_1d(beta_upper)
            )
        
        # Lambda parameters
        if "lambda" in posterior_scfr:
            lambda_mean, lambda_lower, lambda_upper = get_posterior_estimates(
                posterior_scfr, "lambda", self.percentiles
            )
            param_estimates["lambda"] = PosteriorEstimates(
                np.atleast_1d(lambda_mean),
                np.atleast_1d(lambda_lower),
                np.atleast_1d(lambda_upper)
            )
        
        return param_estimates if param_estimates else None
    
    def _prepare_plot_data(
        self,
        result: ScenarioEvaluationResult,
        benchmarks_r_t: Optional[Dict[str, np.ndarray]] = None,
        benchmark_cis: Optional[Dict[str, np.ndarray]] = None,
        its_results: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Prepare data dictionary for plotting functions."""
        plot_data = {
            "true_r_t": result.true_factual_cfr,
            "true_rcf_0_t": result.true_counterfactual_cfr,
            "true_intervention_times_0_abs": result.intervention_times,
            "estimated_r_t_dict": {}
        }
        
        # Add sCFR estimates
        if "sCFR" in result.model_results:
            scfr_result = result.model_results["sCFR"]
            plot_data["estimated_r_t_dict"]["sCFR"] = {
                "mean": scfr_result.factual_estimates.mean,
                "lower": scfr_result.factual_estimates.lower,
                "upper": scfr_result.factual_estimates.upper,
            }
            if scfr_result.counterfactual_estimates:
                plot_data["estimated_r_t_dict"]["sCFR"].update({
                    "cf_mean": scfr_result.counterfactual_estimates.mean,
                    "cf_lower": scfr_result.counterfactual_estimates.lower,
                    "cf_upper": scfr_result.counterfactual_estimates.upper,
                })
        
        # Add benchmark estimates
        if benchmarks_r_t:
            for key in ["cCFR_model", "aCFR_model"]:
                if key in benchmarks_r_t:
                    mean_vals = benchmarks_r_t[key][:len(result.true_factual_cfr)]
                    lower_vals = benchmark_cis.get(f"{key}_lower", mean_vals)[:len(result.true_factual_cfr)] if benchmark_cis else mean_vals
                    upper_vals = benchmark_cis.get(f"{key}_upper", mean_vals)[:len(result.true_factual_cfr)] if benchmark_cis else mean_vals
                    plot_data["estimated_r_t_dict"][key] = {
                        "mean": mean_vals,
                        "lower": lower_vals,
                        "upper": upper_vals,
                    }
        
        # Add fsCFR estimates
        if its_results:
            plot_data["estimated_r_t_dict"]["fsCFR_model"] = {
                "factual_mean": its_results["fsCFR_factual_mean"][:len(result.true_factual_cfr)],
                "factual_lower": its_results.get("fsCFR_factual_lower", its_results["fsCFR_factual_mean"])[:len(result.true_factual_cfr)],
                "factual_upper": its_results.get("fsCFR_factual_upper", its_results["fsCFR_factual_mean"])[:len(result.true_factual_cfr)],
                "cf_mean": its_results["fsCFR_counterfactual_mean"][:len(result.true_counterfactual_cfr)],
                "cf_lower": its_results.get("fsCFR_counterfactual_lower", its_results["fsCFR_counterfactual_mean"])[:len(result.true_counterfactual_cfr)],
                "cf_upper": its_results.get("fsCFR_counterfactual_upper", its_results["fsCFR_counterfactual_mean"])[:len(result.true_counterfactual_cfr)],
            }
        
        return plot_data
    
    # =========================================================================
    # Visualization Methods
    # =========================================================================
    
    def plot_single_run(
        self,
        result: ScenarioEvaluationResult,
        mc_run_idx: int = 0,
        save_plots: bool = True,
        show_plots: bool = False
    ) -> plt.Figure:
        """
        Generate time series plot for a single Monte Carlo run.
        
        Args:
            result: ScenarioEvaluationResult containing evaluation results.
            mc_run_idx: Index of the Monte Carlo run for labeling.
            save_plots: Whether to save the plot to file.
            show_plots: Whether to display the plot.
        
        Returns:
            matplotlib Figure object.
        """
        try:
            # Use existing plotting function
            plot_cfr_timeseries_from_data(
                result.scenario_id,
                mc_run_idx,
                result.plot_data,
                self.output_dir
            )
            
            if show_plots:
                plt.show()
            
            return plt.gcf()
            
        except Exception as e:
            raise RuntimeError(f"Error plotting single run: {str(e)}") from e
    
    def evaluate_and_plot_single_run(
        self,
        sim_data: Dict[str, Any],
        posterior_scfr: Optional[Dict[str, np.ndarray]] = None,
        benchmarks_r_t: Optional[Dict[str, np.ndarray]] = None,
        benchmark_cis: Optional[Dict[str, np.ndarray]] = None,
        its_results: Optional[Dict[str, np.ndarray]] = None,
        mc_run_idx: int = 0,
        save_plots: bool = True,
        show_plots: bool = False
    ) -> Tuple[ScenarioEvaluationResult, plt.Figure]:
        """
        Combined method to evaluate and plot a single run.
        
        This is a convenience method that chains evaluation and plotting together.
        
        Args:
            sim_data: Dictionary containing simulation data.
            posterior_scfr: Posterior samples from sCFR model.
            benchmarks_r_t: Dictionary of benchmark CFR estimates.
            benchmark_cis: Dictionary of benchmark credible intervals.
            its_results: Results from fsCFR model.
            mc_run_idx: Index of Monte Carlo run.
            save_plots: Whether to save plots.
            show_plots: Whether to display plots.
        
        Returns:
            Tuple of (ScenarioEvaluationResult, Figure).
        """
        # Evaluate
        result = self.evaluate_single_run(
            sim_data, posterior_scfr, benchmarks_r_t, benchmark_cis, its_results
        )
        
        # Plot
        fig = self.plot_single_run(result, mc_run_idx, save_plots, show_plots)
        
        return result, fig
    
    def evaluate_aggregated(
        self,
        all_results: List[ScenarioEvaluationResult]
    ) -> AggregatedEvaluationResult:
        """
        Aggregate evaluation results across multiple runs.
        
        Args:
            all_results: List of ScenarioEvaluationResult objects.
        
        Returns:
            AggregatedEvaluationResult containing all aggregated data.
        """
        try:
            # Group results by scenario
            scenario_results: Dict[str, List[ScenarioEvaluationResult]] = {}
            for result in all_results:
                if result.scenario_id not in scenario_results:
                    scenario_results[result.scenario_id] = []
                scenario_results[result.scenario_id].append(result)
            
            # Create DataFrame of all metrics
            metrics_list = []
            for result in all_results:
                metrics_row = {
                    "scenario_id": result.scenario_id,
                    "run_seed": result.run_seed,
                }
                for model_name, model_result in result.model_results.items():
                    for metric_name, metric_value in model_result.metrics.items():
                        metrics_row[f"{metric_name}_{model_name}"] = metric_value
                metrics_list.append(metrics_row)
            
            results_df = pd.DataFrame(metrics_list)
            
            # Prepare aggregated plot data (use first run of each scenario)
            aggregated_plot_data = []
            for scenario_id, results in scenario_results.items():
                if results:
                    # Use the first result for plotting
                    first_result = results[0]
                    plot_dict = first_result.plot_data.copy()
                    plot_dict["scenario_id"] = scenario_id
                    aggregated_plot_data.append(plot_dict)
            
            return AggregatedEvaluationResult(
                results_df=results_df,
                scenario_results=scenario_results,
                aggregated_plot_data=aggregated_plot_data
            )
            
        except Exception as e:
            raise RuntimeError(f"Error aggregating results: {str(e)}") from e
    
    def plot_aggregated_summary(
        self,
        aggregated_result: AggregatedEvaluationResult,
        plot_type: Literal["factual", "counterfactual", "both"] = "both",
        save_plots: bool = True,
        show_plots: bool = False
    ) -> Dict[str, plt.Figure]:
        """
        Generate aggregated summary plots.
        
        Args:
            aggregated_result: AggregatedEvaluationResult containing aggregated data.
            plot_type: Type of plot to generate ("factual", "counterfactual", or "both").
            save_plots: Whether to save plots.
            show_plots: Whether to display plots.
        
        Returns:
            Dictionary mapping plot names to Figure objects.
        """
        figures = {}
        
        try:
            if plot_type in ["factual", "both"]:
                plot_aggregated_factual_summary(
                    aggregated_result.aggregated_plot_data,
                    self.output_dir
                )
                figures["factual"] = plt.gcf()
                if show_plots:
                    plt.show()
                else:
                    plt.close()
            
            if plot_type in ["counterfactual", "both"]:
                plot_aggregated_counterfactual_summary(
                    aggregated_result.aggregated_plot_data,
                    self.output_dir
                )
                figures["counterfactual"] = plt.gcf()
                if show_plots:
                    plt.show()
                else:
                    plt.close()
            
            return figures
            
        except Exception as e:
            raise RuntimeError(f"Error plotting aggregated summary: {str(e)}") from e
    
    def plot_metric_boxplots(
        self,
        aggregated_result: AggregatedEvaluationResult,
        plot_type: Literal["individual", "combined"] = "combined",
        save_plots: bool = True,
        show_plots: bool = False
    ) -> Dict[str, plt.Figure]:
        """
        Generate boxplots of evaluation metrics.
        
        Args:
            aggregated_result: AggregatedEvaluationResult containing metrics.
            plot_type: Type of boxplots ("individual" or "combined").
            save_plots: Whether to save plots.
            show_plots: Whether to display plots.
        
        Returns:
            Dictionary mapping plot names to Figure objects.
        """
        figures = {}
        
        try:
            if plot_type in ["individual", "both"]:
                plot_metric_summary_boxplots(
                    aggregated_result.results_df,
                    self.output_dir
                )
                figures["individual"] = plt.gcf()
                if show_plots:
                    plt.show()
                else:
                    plt.close()
            
            if plot_type in ["combined", "both"]:
                plot_combined_metrics_summary(
                    aggregated_result.results_df,
                    self.output_dir
                )
                figures["combined"] = plt.gcf()
                if show_plots:
                    plt.show()
                else:
                    plt.close()
            
            return figures
            
        except Exception as e:
            raise RuntimeError(f"Error plotting metric boxplots: {str(e)}") from e
    
    def evaluate_and_plot_aggregated(
        self,
        all_results: List[ScenarioEvaluationResult],
        plot_type: Literal["factual", "counterfactual", "both"] = "both",
        metric_plot_type: Literal["individual", "combined"] = "combined",
        save_plots: bool = True,
        show_plots: bool = False
    ) -> Tuple[AggregatedEvaluationResult, Dict[str, plt.Figure]]:
        """
        Combined method to evaluate and plot aggregated results.
        
        Args:
            all_results: List of ScenarioEvaluationResult objects.
            plot_type: Type of summary plots to generate.
            metric_plot_type: Type of metric boxplots to generate.
            save_plots: Whether to save plots.
            show_plots: Whether to display plots.
        
        Returns:
            Tuple of (AggregatedEvaluationResult, dictionary of Figures).
        """
        # Aggregate
        aggregated_result = self.evaluate_aggregated(all_results)
        
        # Generate plots
        figures = {}
        
        # Summary plots
        summary_figures = self.plot_aggregated_summary(
            aggregated_result, plot_type, save_plots, show_plots
        )
        figures.update(summary_figures)
        
        # Metric boxplots
        metric_figures = self.plot_metric_boxplots(
            aggregated_result, metric_plot_type, save_plots, show_plots
        )
        figures.update(metric_figures)
        
        return aggregated_result, figures
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_metric_summary(
        self,
        aggregated_result: AggregatedEvaluationResult,
        model_name: str,
        metric_name: str
    ) -> pd.Series:
        """
        Get summary statistics for a specific metric and model.
        
        Args:
            aggregated_result: AggregatedEvaluationResult.
            model_name: Name of the model (e.g., "sCFR", "fsCFR").
            metric_name: Name of the metric (e.g., "mae_rt", "mcic_rt").
        
        Returns:
            Pandas Series with summary statistics (mean, std, min, max, etc.).
        """
        col_name = f"{metric_name}_{model_name}"
        if col_name not in aggregated_result.results_df.columns:
            raise ValueError(f"Column {col_name} not found in results")
        
        return aggregated_result.results_df[col_name].describe()
    
    def export_metrics_to_csv(
        self,
        aggregated_result: AggregatedEvaluationResult,
        filename: str = "evaluation_metrics.csv"
    ) -> str:
        """
        Export aggregated metrics to CSV file.
        
        Args:
            aggregated_result: AggregatedEvaluationResult.
            filename: Name of the output file.
        
        Returns:
            Path to the saved file.
        """
        filepath = os.path.join(self.output_dir, filename)
        aggregated_result.results_df.to_csv(filepath, index=False)
        return filepath
    
    def compare_models(
        self,
        aggregated_result: AggregatedEvaluationResult,
        metric_name: str,
        models: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare models based on a specific metric.
        
        Args:
            aggregated_result: AggregatedEvaluationResult.
            metric_name: Name of the metric to compare.
            models: List of model names to compare. If None, compares all available models.
        
        Returns:
            DataFrame with comparison statistics.
        """
        if models is None:
            # Extract all model names from column names
            model_names = set()
            for col in aggregated_result.results_df.columns:
                if f"_{metric_name}_" in col or col.endswith(f"_{metric_name}"):
                    parts = col.split("_")
                    if len(parts) >= 2:
                        model_names.add("_".join(parts[2:]) if len(parts) > 2 else parts[-1])
            models = list(model_names)
        
        comparison_data = []
        for model in models:
            col_name = f"{metric_name}_{model}"
            if col_name in aggregated_result.results_df.columns:
                stats = aggregated_result.results_df[col_name].describe()
                comparison_data.append({
                    "model": model,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "median": stats["50%"],
                })
        
        return pd.DataFrame(comparison_data)


# =============================================================================
# Convenience Functions for Backward Compatibility
# =============================================================================

def evaluate_and_visualize_scenario(
    sim_data: Dict[str, Any],
    posterior_scfr: Optional[Dict[str, np.ndarray]] = None,
    benchmarks_r_t: Optional[Dict[str, np.ndarray]] = None,
    benchmark_cis: Optional[Dict[str, np.ndarray]] = None,
    its_results: Optional[Dict[str, np.ndarray]] = None,
    output_dir: str = "./simulation_outputs/plots",
    mc_run_idx: int = 0,
    save_plots: bool = True
) -> Tuple[ScenarioEvaluationResult, plt.Figure]:
    """
    Convenience function for evaluating and visualizing a single scenario.
    
    This function provides a simple interface that maintains backward compatibility
    with the original evaluation and plotting modules.
    
    Args:
        sim_data: Dictionary containing simulation data.
        posterior_scfr: Posterior samples from sCFR model.
        benchmarks_r_t: Dictionary of benchmark CFR estimates.
        benchmark_cis: Dictionary of benchmark credible intervals.
        its_results: Results from fsCFR model.
        output_dir: Directory to save plots.
        mc_run_idx: Index of Monte Carlo run.
        save_plots: Whether to save plots.
    
    Returns:
        Tuple of (ScenarioEvaluationResult, Figure).
    
    Example:
        >>> result, fig = evaluate_and_visualize_scenario(
        ...     sim_data, posterior_scfr, benchmarks_r_t, benchmark_cis, its_results
        ... )
    """
    evaluator = CFREvaluatorVisualizer(output_dir=output_dir)
    return evaluator.evaluate_and_plot_single_run(
        sim_data, posterior_scfr, benchmarks_r_t, benchmark_cis, its_results,
        mc_run_idx, save_plots
    )


def evaluate_and_visualize_aggregated(
    all_sim_data: List[Dict[str, Any]],
    all_posterior_scfr: List[Optional[Dict[str, np.ndarray]]],
    all_benchmarks_r_t: List[Optional[Dict[str, np.ndarray]]],
    all_benchmark_cis: List[Optional[Dict[str, np.ndarray]]],
    all_its_results: List[Optional[Dict[str, np.ndarray]]],
    output_dir: str = "./simulation_outputs/plots",
    save_plots: bool = True
) -> Tuple[AggregatedEvaluationResult, Dict[str, plt.Figure]]:
    """
    Convenience function for evaluating and visualizing aggregated results.
    
    This function processes multiple simulation runs and generates comprehensive
    summary plots and metric comparisons.
    
    Args:
        all_sim_data: List of simulation data dictionaries.
        all_posterior_scfr: List of posterior samples from sCFR model.
        all_benchmarks_r_t: List of benchmark CFR estimates.
        all_benchmark_cis: List of benchmark credible intervals.
        all_its_results: List of fsCFR results.
        output_dir: Directory to save plots.
        save_plots: Whether to save plots.
    
    Returns:
        Tuple of (AggregatedEvaluationResult, dictionary of Figures).
    
    Example:
        >>> agg_result, figures = evaluate_and_visualize_aggregated(
        ...     sim_data_list, posterior_list, bench_r_list, bench_ci_list, its_list
        ... )
    """
    evaluator = CFREvaluatorVisualizer(output_dir=output_dir)
    
    # Evaluate all runs
    all_results = []
    for i, sim_data in enumerate(all_sim_data):
        result = evaluator.evaluate_single_run(
            sim_data,
            all_posterior_scfr[i] if i < len(all_posterior_scfr) else None,
            all_benchmarks_r_t[i] if i < len(all_benchmarks_r_t) else None,
            all_benchmark_cis[i] if i < len(all_benchmark_cis) else None,
            all_its_results[i] if i < len(all_its_results) else None
        )
        all_results.append(result)
    
    # Aggregate and plot
    return evaluator.evaluate_and_plot_aggregated(all_results, save_plots=save_plots)


# =============================================================================
# Module-Level API for Direct Use
# =============================================================================

# Create a default evaluator instance
_default_evaluator = None


def get_default_evaluator() -> CFREvaluatorVisualizer:
    """Get or create the default evaluator instance."""
    global _default_evaluator
    if _default_evaluator is None:
        _default_evaluator = CFREvaluatorVisualizer()
    return _default_evaluator


def evaluate_single_run(
    sim_data: Dict[str, Any],
    posterior_scfr: Optional[Dict[str, np.ndarray]] = None,
    benchmarks_r_t: Optional[Dict[str, np.ndarray]] = None,
    benchmark_cis: Optional[Dict[str, np.ndarray]] = None,
    its_results: Optional[Dict[str, np.ndarray]] = None
) -> ScenarioEvaluationResult:
    """
    Evaluate a single simulation run using the default evaluator.
    
    Args:
        sim_data: Dictionary containing simulation data.
        posterior_scfr: Posterior samples from sCFR model.
        benchmarks_r_t: Dictionary of benchmark CFR estimates.
        benchmark_cis: Dictionary of benchmark credible intervals.
        its_results: Results from fsCFR model.
    
    Returns:
        ScenarioEvaluationResult containing evaluation results.
    """
    evaluator = get_default_evaluator()
    return evaluator.evaluate_single_run(
        sim_data, posterior_scfr, benchmarks_r_t, benchmark_cis, its_results
    )


def plot_single_run(
    result: ScenarioEvaluationResult,
    mc_run_idx: int = 0,
    save_plots: bool = True
) -> plt.Figure:
    """
    Plot a single simulation run using the default evaluator.
    
    Args:
        result: ScenarioEvaluationResult to plot.
        mc_run_idx: Index of Monte Carlo run.
        save_plots: Whether to save plots.
    
    Returns:
        matplotlib Figure object.
    """
    evaluator = get_default_evaluator()
    return evaluator.plot_single_run(result, mc_run_idx, save_plots)


# =============================================================================
# Main Entry Point for Testing
# =============================================================================

if __name__ == "__main__":
    print("evaluation module loaded successfully")
    print("This module provides unified evaluation and visualization for CFR analysis")
