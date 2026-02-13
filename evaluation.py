"""
Evaluation and visualization module for CFR estimation.

This module provides:
- Metric calculation (MAE, MCIW, MCIC, bias, coverage)
- Plotting functions for time series and summary visualizations
- Unified evaluator class for integrated workflow
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
    "fsCFR": "tab:orange",
}


# =============================================================================
# Metric Functions
# =============================================================================

def sanitize_metrics_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame by converting list-like values to scalars."""
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
    """Extract posterior estimates (mean, lower, upper) from samples."""
    samples = posterior_samples[param_name]
    mean = np.mean(samples, axis=0)
    lower = np.percentile(samples, percentiles[0], axis=0)
    upper = np.percentile(samples, percentiles[1], axis=0)
    return mean, lower, upper


def calculate_mae_rt(true_values: np.ndarray, estimated_values: np.ndarray) -> float:
    """Calculate Mean Absolute Error for CFR time series."""
    return float(np.mean(np.abs(true_values - estimated_values)))


def calculate_logit_mae(true_values: np.ndarray, estimated_values: np.ndarray) -> float:
    """Calculate MAE on logit scale."""
    eps = 1e-6
    true_clip = np.clip(true_values, eps, 1 - eps)
    est_clip = np.clip(estimated_values, eps, 1 - eps)
    return float(np.mean(np.abs(logit(est_clip) - logit(true_clip))))


def calculate_mciw_rt(lower_values: np.ndarray, upper_values: np.ndarray) -> float:
    """Calculate Mean Credible Interval Width."""
    return float(np.mean(upper_values - lower_values))


def calculate_mcic_rt(
    true_values: np.ndarray,
    lower_values: np.ndarray,
    upper_values: np.ndarray
) -> float:
    """Calculate Mean Credible Interval Coverage."""
    within_interval = (true_values >= lower_values) & (true_values <= upper_values)
    return float(np.mean(within_interval))


def calculate_param_bias(
    true_value: Union[float, np.ndarray],
    estimated_value: Union[float, np.ndarray]
) -> float:
    """Calculate parameter bias."""
    return float(np.mean(estimated_value) - np.mean(true_value))


def calculate_param_cri_width(
    lower_value: Union[float, np.ndarray],
    upper_value: Union[float, np.ndarray]
) -> float:
    """Calculate credible interval width for parameter."""
    return float(np.mean(upper_value) - np.mean(lower_value))


def calculate_param_cri_coverage(
    true_value: Union[float, np.ndarray],
    lower_value: Union[float, np.ndarray],
    upper_value: Union[float, np.ndarray]
) -> bool:
    """Check if true value is within credible interval."""
    return bool((np.mean(true_value) >= np.mean(lower_value)) and 
                (np.mean(true_value) <= np.mean(upper_value)))


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_cfr_timeseries_from_data(
    scenario_id: str,
    mc_run_idx: int,
    plot_data: Dict[str, Any],
    output_dir: str
) -> None:
    """Plot CFR time series for a single run."""
    os.makedirs(output_dir, exist_ok=True)
    
    true_r_t = plot_data["true_r_t"]
    true_rcf_t = plot_data["true_rcf_0_t"]
    intervention_times = plot_data.get("true_intervention_times_0_abs", np.array([]))
    estimated_r_t_dict = plot_data["estimated_r_t_dict"]
    
    T = len(true_r_t)
    t_array = np.arange(T)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Factual CFR
    ax1 = axes[0]
    ax1.plot(t_array, true_r_t, 'k-', linewidth=2, label='True CFR', alpha=0.7)
    
    if "sCFR" in estimated_r_t_dict:
        sCFR_data = estimated_r_t_dict["sCFR"]
        ax1.plot(t_array, sCFR_data["mean"], 'b-', linewidth=2, label='sCFR', alpha=0.8)
        ax1.fill_between(t_array, sCFR_data["lower"], sCFR_data["upper"], color='blue', alpha=0.2)
    
    if "cCFR_model" in estimated_r_t_dict:
        cCFR_data = estimated_r_t_dict["cCFR_model"]
        ax1.plot(t_array, cCFR_data["mean"], linestyle='--', color=METHOD_COLORS["cCFR"], linewidth=2.2, label='cCFR', alpha=0.8)
    
    if "aCFR_model" in estimated_r_t_dict:
        aCFR_data = estimated_r_t_dict["aCFR_model"]
        ax1.plot(t_array, aCFR_data["mean"], linestyle='--', color=METHOD_COLORS["aCFR"], linewidth=2.2, label='aCFR', alpha=0.8)
    
    if "fsCFR_model" in estimated_r_t_dict:
        its_data = estimated_r_t_dict["fsCFR_model"]
        ax1.plot(t_array, its_data["factual_mean"], linestyle='--', color=METHOD_COLORS["fsCFR"], linewidth=2.6, label='fsCFR', alpha=0.9)
    
    for t_int in intervention_times:
        if 0 <= t_int < T:
            ax1.axvline(x=t_int, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Case Fatality Rate')
    ax1.set_title(f'{scenario_id} - Run {mc_run_idx}: Factual CFR Estimates')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Counterfactual CFR
    ax2 = axes[1]
    ax2.plot(t_array, true_rcf_t, 'k-', linewidth=2, label='True Counterfactual CFR', alpha=0.7)
    
    if "sCFR" in estimated_r_t_dict and "cf_mean" in estimated_r_t_dict["sCFR"]:
        sCFR_cf = estimated_r_t_dict["sCFR"]
        ax2.plot(t_array, sCFR_cf["cf_mean"], 'b-', linewidth=2, label='sCFR Counterfactual', alpha=0.8)
        ax2.fill_between(t_array, sCFR_cf["cf_lower"], sCFR_cf["cf_upper"], color='blue', alpha=0.2)
    
    if "fsCFR_model" in estimated_r_t_dict:
        its_cf = estimated_r_t_dict["fsCFR_model"]
        ax2.plot(t_array, its_cf["cf_mean"], linestyle='--', color=METHOD_COLORS["fsCFR"], linewidth=2.6, label='fsCFR Counterfactual', alpha=0.9)
    
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


def plot_aggregated_factual_summary(aggregated_plot_data: list, output_dir: str) -> None:
    """Plot aggregated factual summary across all scenarios."""
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
        true_zeta_t = plot_dict.get("true_zeta_0_t", None)
        true_eta_t = plot_dict.get("true_eta_0_t", None)
        intervention_times = plot_dict.get("true_intervention_times_0_abs", np.array([]))
        estimated_r_t_dict = plot_dict["estimated_r_t_dict"]
        
        # Compute smooth True line without random effect
        # true_r_t = sigmoid(zeta + intervention + eta)
        # We want: sigmoid(zeta + intervention) = sigmoid(logit(true_r_t) - eta)
        if true_zeta_t is not None and true_eta_t is not None:
            true_r_t_smooth = sigmoid(logit(np.clip(true_r_t, 1e-6, 1-1e-6)) - true_eta_t)
        else:
            # Fallback: if true_zeta_0_t or true_eta_0_t not available, use true_r_t (will still have random effect)
            true_r_t_smooth = true_r_t
        
        T = len(true_r_t_smooth)
        t_array = np.arange(T)
        
        ax.plot(t_array, true_r_t_smooth, color='black', linewidth=3.5, label='True', alpha=0.7)
        
        if "sCFR" in estimated_r_t_dict:
            sCFR_data = estimated_r_t_dict["sCFR"]
            if "mean" in sCFR_data and len(sCFR_data["mean"]) == T:
                ax.plot(t_array, sCFR_data["mean"], color=METHOD_COLORS["sCFR"], linewidth=3.5, label='sCFR', alpha=0.9)
                if "lower" in sCFR_data and "upper" in sCFR_data and len(sCFR_data["lower"]) == T and len(sCFR_data["upper"]) == T:
                    ax.fill_between(t_array, sCFR_data["lower"], sCFR_data["upper"], color=METHOD_COLORS["sCFR"], alpha=0.2)
        
        if "cCFR_model" in estimated_r_t_dict:
            cCFR_data = estimated_r_t_dict["cCFR_model"]
            if "mean" in cCFR_data and len(cCFR_data["mean"]) == T:
                ax.plot(t_array, cCFR_data["mean"], linestyle='--', color=METHOD_COLORS["cCFR"], linewidth=3.0, label='cCFR', alpha=0.85)
        
        if "aCFR_model" in estimated_r_t_dict:
            aCFR_data = estimated_r_t_dict["aCFR_model"]
            if "mean" in aCFR_data and len(aCFR_data["mean"]) == T:
                ax.plot(t_array, aCFR_data["mean"], linestyle='--', color=METHOD_COLORS["aCFR"], linewidth=3.0, label='aCFR', alpha=0.85)
        
        if "fsCFR_model" in estimated_r_t_dict:
            its_data = estimated_r_t_dict["fsCFR_model"]
            if "factual_mean" in its_data and len(its_data["factual_mean"]) == T:
                ax.plot(t_array, its_data["factual_mean"], linestyle='--', color=METHOD_COLORS["fsCFR"], linewidth=3.5, label='fsCFR', alpha=0.95)
        
        for t_int in intervention_times:
            if 0 <= t_int < T:
                ax.axvline(x=t_int, color='black', linestyle='--', linewidth=2.0, alpha=0.85)
        
        ax.set_title(f'{scenario_id}', fontsize=18, fontweight='bold')
        ax.set_xlabel('Time', fontsize=16, fontweight='bold')
        ax.set_ylabel('CFR', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14, width=1.5)
        if cfr_codes.index(cfr_code) == 0 and int_codes.index(int_code) == 0:
            ax.legend(loc='best', fontsize=13, frameon=True, framealpha=0.9)
        else:
            ax.legend().remove()
        ax.grid(True, alpha=0.3, linewidth=1.0)

    for row_idx, cfr_code in enumerate(cfr_codes):
        row_label = config.cfr_types_params[cfr_code]["name"]
        axes[row_idx, 0].set_ylabel(f"{row_label}\nCFR", fontsize=16, fontweight='bold')
    for col_idx, int_code in enumerate(int_codes):
        col_label = config.intervention_types_params[int_code]["name"]
        axes[0, col_idx].set_title(f"{col_label}", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "aggregated_factual_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_path_pdf = os.path.join(output_dir, "aggregated_factual_summary.pdf")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    plt.close()


def plot_aggregated_counterfactual_summary(aggregated_plot_data: list, output_dir: str) -> None:
    """Plot aggregated counterfactual summary across all scenarios."""
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
        true_zeta_t = plot_dict.get("true_zeta_0_t", None)
        true_eta_t = plot_dict.get("true_eta_0_t", None)
        intervention_times = plot_dict.get("true_intervention_times_0_abs", np.array([]))
        estimated_r_t_dict = plot_dict["estimated_r_t_dict"]
        
        # Compute smooth True CF line without random effect
        # true_rcf_t = sigmoid(zeta + eta)
        # We want: sigmoid(zeta) = sigmoid(logit(true_rcf_t) - eta)
        if true_zeta_t is not None:
            # true_zeta_t is already on logit scale, so just apply sigmoid
            true_rcf_t_smooth = sigmoid(true_zeta_t)
        elif true_eta_t is not None:
            # Fallback: remove random effect from true_rcf_t
            true_rcf_t_smooth = sigmoid(logit(np.clip(true_rcf_t, 1e-6, 1-1e-6)) - true_eta_t)
        else:
            # Last resort: use true_rcf_t directly (will still have random effect)
            true_rcf_t_smooth = true_rcf_t
        
        T = len(true_rcf_t_smooth)
        t_array = np.arange(T)
        
        ax.plot(t_array, true_rcf_t_smooth, color='black', linewidth=3.5, label='True CF', alpha=0.7)
        
        if "sCFR" in estimated_r_t_dict and "cf_mean" in estimated_r_t_dict["sCFR"]:
            sCFR_cf = estimated_r_t_dict["sCFR"]
            if "cf_mean" in sCFR_cf and len(sCFR_cf["cf_mean"]) == T:
                ax.plot(t_array, sCFR_cf["cf_mean"], color=METHOD_COLORS["sCFR"], linewidth=3.5, label='sCFR CF', alpha=0.9)
                if "cf_lower" in sCFR_cf and "cf_upper" in sCFR_cf and len(sCFR_cf["cf_lower"]) == T and len(sCFR_cf["cf_upper"]) == T:
                    ax.fill_between(t_array, sCFR_cf["cf_lower"], sCFR_cf["cf_upper"], color=METHOD_COLORS["sCFR"], alpha=0.2)
        
        if "fsCFR_model" in estimated_r_t_dict:
            its_cf = estimated_r_t_dict["fsCFR_model"]
            if "cf_mean" in its_cf and len(its_cf["cf_mean"]) == T:
                ax.plot(t_array, its_cf["cf_mean"], linestyle='--', color=METHOD_COLORS["fsCFR"], linewidth=3.5, label='fsCFR CF', alpha=0.95)
        
        for t_int in intervention_times:
            if 0 <= t_int < T:
                ax.axvline(x=t_int, color='black', linestyle='--', linewidth=2.0, alpha=0.85)
        
        ax.set_title(f'{scenario_id}', fontsize=18, fontweight='bold')
        ax.set_xlabel('Time', fontsize=16, fontweight='bold')
        ax.set_ylabel('CFR', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14, width=1.5)
        if cfr_codes.index(cfr_code) == 0 and int_codes.index(int_code) == 0:
            ax.legend(loc='best', fontsize=13, frameon=True, framealpha=0.9)
        else:
            ax.legend().remove()
        ax.grid(True, alpha=0.3, linewidth=1.0)

    for row_idx, cfr_code in enumerate(cfr_codes):
        row_label = config.cfr_types_params[cfr_code]["name"]
        axes[row_idx, 0].set_ylabel(f"{row_label}\nCFR", fontsize=16, fontweight='bold')
    for col_idx, int_code in enumerate(int_codes):
        col_label = config.intervention_types_params[int_code]["name"]
        axes[0, col_idx].set_title(f"{col_label}", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "aggregated_counterfactual_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_path_pdf = os.path.join(output_dir, "aggregated_counterfactual_summary.pdf")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    plt.close()


def plot_metric_summary_boxplots(results_df: pd.DataFrame, output_dir: str) -> None:
    """Plot summary boxplots of evaluation metrics."""
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


def plot_combined_metrics_summary(results_df: pd.DataFrame, output_dir: str) -> None:
    """Plot combined metrics summary."""
    os.makedirs(output_dir, exist_ok=True)
    results_df = sanitize_metrics_dataframe(results_df.copy())
    if "error" in results_df.columns:
        results_df = results_df[results_df["error"].isin([None, "None"])]

    scenario_k_map = {s["id"]: s.get("num_interventions_K_true", 0) for s in config.SCENARIOS}
    beta_cols = [c for c in results_df.columns if c.startswith("beta_abs_") and ("_mae_" in c or "_coverage_" in c)]
    beta_df = results_df[["scenario_id"] + beta_cols].copy()
    for k in (1, 2):
        mae_key = f"beta_abs_step_{k}_mae_sCFR"
        cov_key = f"beta_abs_step_{k}_coverage_sCFR"
        if mae_key in beta_df.columns:
            beta_df[f"beta_abs_step_{k}_mae"] = beta_df[mae_key]
        if cov_key in beta_df.columns:
            beta_df[f"beta_abs_step_{k}_coverage"] = beta_df[cov_key]
        mae_key = f"beta_abs_slope_{k}_mae_sCFR"
        cov_key = f"beta_abs_slope_{k}_coverage_sCFR"
        if mae_key in beta_df.columns:
            beta_df[f"beta_abs_slope_{k}_mae"] = beta_df[mae_key]
        if cov_key in beta_df.columns:
            beta_df[f"beta_abs_slope_{k}_coverage"] = beta_df[cov_key]
    beta_df["K"] = beta_df["scenario_id"].map(scenario_k_map)
    beta_summary = beta_df.groupby(["scenario_id", "K"], dropna=False).mean(numeric_only=True).reset_index()
    beta_csv_path = os.path.join(output_dir, "beta_abs_metrics_summary.csv")
    beta_summary.to_csv(beta_csv_path, index=False)

    fig = plt.figure(figsize=(18, 10))
    # Increase bottom margin and adjust spacing for rotated x-axis labels in bottom subplots
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.5, wspace=0.25, 
                          bottom=0.12, top=0.95, left=0.08, right=0.98)

    scenario_ids = [s["id"] for s in config.SCENARIOS]
    x_base = np.arange(len(scenario_ids))

    def grouped_boxplot(ax, metric_cols, labels, colors, ylabel, title, rotate_xticks=False):
        offsets = np.linspace(-0.3, 0.3, len(metric_cols))
        width = 0.18 if len(metric_cols) > 2 else 0.25
        for idx in range(len(scenario_ids)):
            if idx % 2 == 1:
                ax.axvspan(idx - 0.5, idx + 0.5, color="#d9d9d9", alpha=0.7, zorder=0)
        for idx, (col, label, color, offset) in enumerate(zip(metric_cols, labels, colors, offsets)):
            data = [
                results_df.loc[results_df["scenario_id"] == scen, col].dropna().values
                if col in results_df.columns else np.array([])
                for scen in scenario_ids
            ]
            positions = x_base + offset
            bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True, showfliers=False, manage_ticks=False)
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
            for element in ["whiskers", "caps", "medians"]:
                for line in bp[element]:
                    line.set_color(color)
                    line.set_linewidth(1.5)
        ax.set_xticks(x_base)
        if rotate_xticks:
            # Rotate labels for bottom subplots to avoid overlap
            ax.set_xticklabels(scenario_ids, rotation=45, ha='right', fontsize=12, fontweight='bold')
            # Adjust bottom margin to accommodate rotated labels
            ax.tick_params(axis='x', which='major', pad=8)
        else:
            ax.set_xticklabels(scenario_ids, rotation=0, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.tick_params(axis='y', which='major', labelsize=14, width=1.5)
        ax.grid(True, alpha=0.3, axis="y", linewidth=1.0)
        ax.set_xlim(-0.7, len(scenario_ids) - 0.3)

    ax_main = fig.add_subplot(gs[0, :])
    grouped_boxplot(
        ax_main,
        metric_cols=["mae_rt_logit_sCFR", "mae_rt_logit_cCFR", "mae_rt_logit_aCFR", "mae_rt_logit_fsCFR"],
        labels=["sCFR", "cCFR", "aCFR", "fsCFR"],
        colors=[METHOD_COLORS["sCFR"], METHOD_COLORS["cCFR"], METHOD_COLORS["aCFR"], METHOD_COLORS["fsCFR"]],
        ylabel=r"Logit MAE ($r_F$)",
        title="Factual CFR MAE by Scenario"
    )
    ax_main.legend(
        handles=[plt.Line2D([0], [0], color=METHOD_COLORS[m], lw=6) for m in ["sCFR", "cCFR", "aCFR", "fsCFR"]],
        labels=["sCFR", "cCFR", "aCFR", "fsCFR"],
        loc="upper right", fontsize=14, frameon=True, framealpha=0.9
    )

    ax_cf = fig.add_subplot(gs[1, 0])
    grouped_boxplot(
        ax_cf,
        metric_cols=["mae_rcf_logit_sCFR", "mae_rcf_logit_fsCFR"],
        labels=["sCFR", "fsCFR"],
        colors=[METHOD_COLORS["sCFR"], METHOD_COLORS["fsCFR"]],
        ylabel=r"Logit MAE ($r_{CF}$)",
        title="Counterfactual CFR MAE",
        rotate_xticks=True
    )

    ax_base = fig.add_subplot(gs[1, 1])
    grouped_boxplot(
        ax_base,
        metric_cols=["mae_baseline_logit_sCFR", "mae_baseline_logit_fsCFR"],
        labels=["sCFR", "fsCFR"],
        colors=[METHOD_COLORS["sCFR"], METHOD_COLORS["fsCFR"]],
        ylabel="MAE (Baseline Logit)",
        title="Baseline Effect MAE",
        rotate_xticks=True
    )

    ax_rand = fig.add_subplot(gs[1, 2])
    grouped_boxplot(
        ax_rand,
        metric_cols=["mae_random_logit_sCFR", "mae_random_logit_fsCFR"],
        labels=["sCFR", "fsCFR"],
        colors=[METHOD_COLORS["sCFR"], METHOD_COLORS["fsCFR"]],
        ylabel="MAE (Random Effect)",
        title="Random Effect MAE",
        rotate_xticks=True
    )

    output_path = os.path.join(output_dir, "combined_metrics_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_path_pdf = os.path.join(output_dir, "combined_metrics_summary.pdf")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    plt.close()


# =============================================================================
# Data Classes
# =============================================================================

class EstimatorType(Enum):
    S_CFR = "sCFR"
    C_CFR = "cCFR"
    A_CFR = "aCFR"
    fsCFR = "fsCFR"


class MetricType(Enum):
    MAE = "mae"
    MCIW = "mciw"
    MCIC = "mcic"
    BIAS = "bias"
    WIDTH = "width"
    COVER = "cover"


class PlotType(Enum):
    TIME_SERIES = "timeseries"
    AGGREGATED_FACTUAL = "aggregated_factual"
    AGGREGATED_COUNTERFACTUAL = "aggregated_counterfactual"
    METRIC_BOXPLOT = "metric_boxplot"
    COMBINED_METRICS = "combined_metrics"


@dataclass
class PosteriorEstimates:
    """Container for posterior estimates with credible intervals."""
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    samples: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if not (self.mean.shape == self.lower.shape == self.upper.shape):
            raise ValueError(f"Shape mismatch: mean {self.mean.shape}, lower {self.lower.shape}, upper {self.upper.shape}")


@dataclass
class ModelEvaluationResult:
    """Evaluation result for a single model."""
    model_name: str
    factual_estimates: PosteriorEstimates
    counterfactual_estimates: Optional[PosteriorEstimates] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    parameter_estimates: Optional[Dict[str, PosteriorEstimates]] = None
    
    def get_metric(self, metric_name: str) -> float:
        return self.metrics.get(metric_name, np.nan)


@dataclass
class ScenarioEvaluationResult:
    """Evaluation result for a single scenario."""
    scenario_id: str
    run_seed: int
    true_factual_cfr: np.ndarray
    true_counterfactual_cfr: np.ndarray
    intervention_times: np.ndarray
    model_results: Dict[str, ModelEvaluationResult] = field(default_factory=dict)
    plot_data: Dict[str, Any] = field(default_factory=dict)
    true_zeta_0_t: Optional[np.ndarray] = None
    true_eta_0_t: Optional[np.ndarray] = None


@dataclass
class AggregatedEvaluationResult:
    """Aggregated evaluation results across multiple runs."""
    results_df: pd.DataFrame
    scenario_results: Dict[str, List[ScenarioEvaluationResult]] = field(default_factory=dict)
    aggregated_plot_data: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# Main Evaluator Class
# =============================================================================

class CFREvaluatorVisualizer:
    """Unified interface for evaluating CFR estimation methods and generating visualizations."""
    
    def __init__(
        self,
        output_dir: str = "./simulation_outputs/plots",
        percentiles: Tuple[float, float] = (2.5, 97.5),
        figsize: Tuple[int, int] = (12, 7),
        dpi: int = 300
    ):
        self.output_dir = output_dir
        self.percentiles = percentiles
        self.figsize = figsize
        self.dpi = dpi
        os.makedirs(self.output_dir, exist_ok=True)
        self._configure_plotting_style()
    
    def _configure_plotting_style(self) -> None:
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
    
    def evaluate_single_run(
        self,
        sim_data: Dict[str, Any],
        posterior_scfr: Optional[Dict[str, np.ndarray]] = None,
        benchmarks_r_t: Optional[Dict[str, np.ndarray]] = None,
        benchmark_cis: Optional[Dict[str, np.ndarray]] = None,
        its_results: Optional[Dict[str, np.ndarray]] = None
    ) -> ScenarioEvaluationResult:
        """Evaluate a single simulation run."""
        try:
            self._validate_simulation_data(sim_data)
            
            scenario_id = sim_data["scenario_id"]
            run_seed = sim_data["run_seed"]
            T_analyze = config.T_ANALYSIS_LENGTH
            
            true_r_t = sim_data["true_r_0_t"][:T_analyze]
            true_rcf_t = sim_data["true_rcf_0_t"][:T_analyze]
            intervention_times = sim_data.get("true_intervention_times_0_abs", np.array([]))
            
            result = ScenarioEvaluationResult(
                scenario_id=scenario_id,
                run_seed=run_seed,
                true_factual_cfr=true_r_t,
                true_counterfactual_cfr=true_rcf_t,
                intervention_times=intervention_times,
                true_zeta_0_t=sim_data.get("true_zeta_0_t", np.array([]))[:T_analyze],
                true_eta_0_t=sim_data.get("true_eta_0_t", np.array([]))[:T_analyze]
            )
            
            if posterior_scfr:
                result.model_results["sCFR"] = self._evaluate_scfr(posterior_scfr, true_r_t, true_rcf_t, T_analyze, sim_data)
            
            if benchmarks_r_t:
                if benchmark_cis is None:
                    benchmark_cis = {}
                result.model_results["cCFR"] = self._evaluate_benchmark("cCFR", benchmarks_r_t, benchmark_cis, true_r_t, T_analyze)
                result.model_results["aCFR"] = self._evaluate_benchmark("aCFR", benchmarks_r_t, benchmark_cis, true_r_t, T_analyze)
            
            if its_results:
                result.model_results["fsCFR"] = self._evaluate_its(its_results, true_r_t, true_rcf_t, T_analyze, sim_data)
            
            result.plot_data = self._prepare_plot_data(result, benchmarks_r_t, benchmark_cis, its_results)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error evaluating single run: {str(e)}") from e
    
    def _validate_simulation_data(self, sim_data: Dict[str, Any]) -> None:
        required_keys = ["scenario_id", "run_seed", "true_r_0_t", "true_rcf_0_t",
                        "true_intervention_times_0_abs", "num_interventions_true_K", "true_zeta_0_t", "true_eta_0_t"]
        missing_keys = [k for k in required_keys if k not in sim_data]
        if missing_keys:
            raise ValueError(f"Missing required simulation data keys: {missing_keys}")

    def _evaluate_scfr(self, posterior_scfr, true_r_t, true_rcf_t, T_analyze, sim_data):
        """Evaluate sCFR model."""
        r_t_key = "r_t" if "r_t" in posterior_scfr else "p"
        r_cf_key = "r_cf" if "r_cf" in posterior_scfr else "p_cf"
        r_t_mean, r_t_lower, r_t_upper = get_posterior_estimates(posterior_scfr, r_t_key, self.percentiles)
        rcf_t_mean, rcf_t_lower, rcf_t_upper = get_posterior_estimates(posterior_scfr, r_cf_key, self.percentiles)
        
        r_t_mean, r_t_lower, r_t_upper = r_t_mean[:T_analyze], r_t_lower[:T_analyze], r_t_upper[:T_analyze]
        rcf_t_mean, rcf_t_lower, rcf_t_upper = rcf_t_mean[:T_analyze], rcf_t_lower[:T_analyze], rcf_t_upper[:T_analyze]
        
        metrics = {
            "mae_rt_logit": float(calculate_logit_mae(true_r_t, r_t_mean)),
            "mciw_rt": float(calculate_mciw_rt(r_t_lower, r_t_upper)),
            "mcic_rt": float(calculate_mcic_rt(true_r_t, r_t_lower, r_t_upper)),
            "mae_rcf_logit": float(calculate_logit_mae(true_rcf_t, rcf_t_mean)),
            "mciw_rcf": float(calculate_mciw_rt(rcf_t_lower, rcf_t_upper)),
            "mcic_rcf": float(calculate_mcic_rt(true_rcf_t, rcf_t_lower, rcf_t_upper)),
        }

        true_zeta = sim_data["true_zeta_0_t"][:T_analyze]
        true_eta = sim_data["true_eta_0_t"][:T_analyze]
        if "baseline_logit" in posterior_scfr:
            baseline_mean = np.mean(posterior_scfr["baseline_logit"], axis=0)[:T_analyze]
            metrics["mae_baseline_logit"] = float(np.mean(np.abs(baseline_mean - true_zeta)))
        if "delta" in posterior_scfr:
            delta_mean = np.mean(posterior_scfr["delta"], axis=0)[:T_analyze]
            metrics["mae_random_logit"] = float(np.mean(np.abs(delta_mean - true_eta)))
        
        param_estimates = self._extract_parameter_estimates(posterior_scfr)

        if "beta_abs" in posterior_scfr:
            beta_mean = np.mean(posterior_scfr["beta_abs"], axis=0)
            beta_lower = np.percentile(posterior_scfr["beta_abs"], self.percentiles[0], axis=0)
            beta_upper = np.percentile(posterior_scfr["beta_abs"], self.percentiles[1], axis=0)
            true_beta_abs = sim_data.get("true_beta_abs_0", np.array([]))
            if len(true_beta_abs) > 0:
                metrics["beta_abs_mae"] = float(np.mean(np.abs(beta_mean - true_beta_abs)))
                coverage = (true_beta_abs >= beta_lower) & (true_beta_abs <= beta_upper)
                metrics["beta_abs_coverage"] = float(np.mean(coverage))

            for k in range(len(true_beta_abs)):
                metrics[f"beta_abs_step_{k+1}_mae"] = float(abs(beta_mean[k] - true_beta_abs[k]))
                metrics[f"beta_abs_step_{k+1}_coverage"] = float((true_beta_abs[k] >= beta_lower[k]) and (true_beta_abs[k] <= beta_upper[k]))

        if "beta_slope_abs" in posterior_scfr:
            slope_mean = np.mean(posterior_scfr["beta_slope_abs"], axis=0)
            slope_lower = np.percentile(posterior_scfr["beta_slope_abs"], self.percentiles[0], axis=0)
            slope_upper = np.percentile(posterior_scfr["beta_slope_abs"], self.percentiles[1], axis=0)
            true_beta_slope_abs = sim_data.get("true_beta_slope_abs_0", np.array([]))
            for k in range(len(true_beta_slope_abs)):
                metrics[f"beta_abs_slope_{k+1}_mae"] = float(abs(slope_mean[k] - true_beta_slope_abs[k]))
                metrics[f"beta_abs_slope_{k+1}_coverage"] = float((true_beta_slope_abs[k] >= slope_lower[k]) and (true_beta_slope_abs[k] <= slope_upper[k]))
        
        return ModelEvaluationResult(
            model_name="sCFR",
            factual_estimates=PosteriorEstimates(r_t_mean, r_t_lower, r_t_upper),
            counterfactual_estimates=PosteriorEstimates(rcf_t_mean, rcf_t_lower, rcf_t_upper),
            metrics=metrics,
            parameter_estimates=param_estimates
        )
    
    def _evaluate_benchmark(self, model_name, benchmarks_r_t, benchmark_cis, true_r_t, T_analyze):
        """Evaluate benchmark CFR method."""
        if model_name == "cCFR":
            r_t_key = "cCFR_model"
        elif model_name == "aCFR":
            r_t_key = "aCFR_model"
        else:
            raise ValueError(f"Unknown benchmark model: {model_name}")
        
        r_t_mean = benchmarks_r_t[r_t_key][:T_analyze]
        r_t_lower = benchmark_cis.get(f"{r_t_key}_lower", r_t_mean)[:T_analyze]
        r_t_upper = benchmark_cis.get(f"{r_t_key}_upper", r_t_mean)[:T_analyze]
        
        metrics = {"mae_rt_logit": float(calculate_logit_mae(true_r_t, r_t_mean))}
        
        return ModelEvaluationResult(
            model_name=model_name,
            factual_estimates=PosteriorEstimates(r_t_mean, r_t_lower, r_t_upper),
            metrics=metrics
        )
    
    def _evaluate_its(self, its_results, true_r_t, true_rcf_t, T_analyze, sim_data):
        """Evaluate fsCFR model."""
        factual_mean = its_results["fsCFR_factual_mean"][:T_analyze]
        factual_lower = its_results.get("fsCFR_factual_lower", factual_mean)[:T_analyze]
        factual_upper = its_results.get("fsCFR_factual_upper", factual_mean)[:T_analyze]
        
        cf_mean = its_results["fsCFR_counterfactual_mean"][:T_analyze]
        cf_lower = its_results.get("fsCFR_counterfactual_lower", cf_mean)[:T_analyze]
        cf_upper = its_results.get("fsCFR_counterfactual_upper", cf_mean)[:T_analyze]
        
        metrics = {
            "mae_rt_logit": float(calculate_logit_mae(true_r_t, factual_mean)),
            "mae_rcf_logit": float(calculate_logit_mae(true_rcf_t, cf_mean)),
        }

        true_zeta = sim_data["true_zeta_0_t"][:T_analyze]
        true_eta = sim_data["true_eta_0_t"][:T_analyze]
        if "fsCFR_baseline_logit" in its_results:
            baseline_hat = its_results["fsCFR_baseline_logit"][:T_analyze]
            metrics["mae_baseline_logit"] = float(np.mean(np.abs(baseline_hat - true_zeta)))
        if "fsCFR_delta" in its_results:
            delta_hat = its_results["fsCFR_delta"][:T_analyze]
            metrics["mae_random_logit"] = float(np.mean(np.abs(delta_hat - true_eta)))
        
        num_interventions = sim_data["num_interventions_true_K"]
        if num_interventions > 0:
            true_beta_abs = sim_data.get("true_beta_abs_0", np.array([]))
            est_beta_abs = its_results.get("fsCFR_beta_abs_est", np.array([]))
            for k in range(min(num_interventions, len(true_beta_abs), len(est_beta_abs))):
                metrics[f"beta_abs_step_{k+1}_mae"] = float(abs(est_beta_abs[k] - true_beta_abs[k]))
            if len(est_beta_abs) > 0 and len(true_beta_abs) > 0:
                metrics["beta_abs_mae"] = float(np.mean(np.abs(est_beta_abs - true_beta_abs)))

            true_beta_slope_abs = sim_data.get("true_beta_slope_abs_0", np.array([]))
            est_beta_slope_abs = its_results.get("fsCFR_beta_slope_abs_est", np.array([]))
            for k in range(min(len(true_beta_slope_abs), len(est_beta_slope_abs))):
                metrics[f"beta_abs_slope_{k+1}_mae"] = float(abs(est_beta_slope_abs[k] - true_beta_slope_abs[k]))
        
        return ModelEvaluationResult(
            model_name="fsCFR",
            factual_estimates=PosteriorEstimates(factual_mean, factual_lower, factual_upper),
            counterfactual_estimates=PosteriorEstimates(cf_mean, cf_lower, cf_upper),
            metrics=metrics
        )

    def _extract_parameter_estimates(self, posterior_scfr):
        """Extract parameter estimates from posterior samples."""
        param_estimates = {}
        
        if "beta_abs" in posterior_scfr:
            beta_mean, beta_lower, beta_upper = get_posterior_estimates(posterior_scfr, "beta_abs", self.percentiles)
            param_estimates["beta_abs"] = PosteriorEstimates(np.atleast_1d(beta_mean), np.atleast_1d(beta_lower), np.atleast_1d(beta_upper))
        
        if "lambda" in posterior_scfr:
            lambda_mean, lambda_lower, lambda_upper = get_posterior_estimates(posterior_scfr, "lambda", self.percentiles)
            param_estimates["lambda"] = PosteriorEstimates(np.atleast_1d(lambda_mean), np.atleast_1d(lambda_lower), np.atleast_1d(lambda_upper))
        
        return param_estimates if param_estimates else None
    
    def _prepare_plot_data(self, result, benchmarks_r_t, benchmark_cis, its_results):
        """Prepare data dictionary for plotting."""
        plot_data = {
            "true_r_t": result.true_factual_cfr,
            "true_rcf_0_t": result.true_counterfactual_cfr,
            "true_intervention_times_0_abs": result.intervention_times,
            "estimated_r_t_dict": {},
            "true_zeta_0_t": result.true_zeta_0_t,
            "true_eta_0_t": result.true_eta_0_t
        }
        
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
        
        if benchmarks_r_t:
            for key in ["cCFR_model", "aCFR_model"]:
                if key in benchmarks_r_t:
                    mean_vals = benchmarks_r_t[key][:len(result.true_factual_cfr)]
                    lower_vals = benchmark_cis.get(f"{key}_lower", mean_vals)[:len(result.true_factual_cfr)] if benchmark_cis else mean_vals
                    upper_vals = benchmark_cis.get(f"{key}_upper", mean_vals)[:len(result.true_factual_cfr)] if benchmark_cis else mean_vals
                    plot_data["estimated_r_t_dict"][key] = {"mean": mean_vals, "lower": lower_vals, "upper": upper_vals}
        
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
    
    def plot_single_run(self, result, mc_run_idx=0, save_plots=True, show_plots=False):
        """Generate time series plot for a single run."""
        try:
            plot_cfr_timeseries_from_data(result.scenario_id, mc_run_idx, result.plot_data, self.output_dir)
            if show_plots:
                plt.show()
            return plt.gcf()
        except Exception as e:
            raise RuntimeError(f"Error plotting single run: {str(e)}") from e


# =============================================================================
# Convenience Functions
# =============================================================================

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
    """Evaluate a single simulation run using the default evaluator."""
    evaluator = get_default_evaluator()
    return evaluator.evaluate_single_run(sim_data, posterior_scfr, benchmarks_r_t, benchmark_cis, its_results)


if __name__ == "__main__":
    print("evaluation module loaded successfully")
