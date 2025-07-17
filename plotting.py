# File: plotting.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import config
import evaluation # For get_posterior_estimates from saved summaries if needed

plt.style.use('seaborn-v0_8-paper') 
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 16,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 12, 'figure.dpi': 150,
    'savefig.dpi': 300, 'lines.linewidth': 1.5,
})
MODEL_COLORS = {
    'True Factual': 'black', 'True Counterfactual': 'dimgray',
    'sCFR': 'blue', 'cCFR': 'red', 'aCFR': 'green', 'ITS': 'purple'
}

def plot_cfr_timeseries_from_data(scenario_id, mc_run_idx, plot_data_dict, output_dir):
    """
    Generates and saves a time-series plot for a single specified MC run
    based on a dictionary of pre-loaded or pre-calculated data.

    Args:
        scenario_id (str): The ID of the scenario (e.g., "S01").
        mc_run_idx (int): The index of the Monte Carlo run being plotted.
        plot_data_dict (dict): A dictionary containing all necessary time-series data.
        output_dir (str): The directory where the output PDF will be saved.
    """
    T_analyze = len(plot_data_dict["true_r_t"])
    time_points = np.arange(T_analyze)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Get data dictionaries for cleaner plotting code
    est_dict = plot_data_dict.get("estimated_r_t_dict", {})
    sCFR_est = est_dict.get("sCFR", {})
    cCFR_est = est_dict.get("cCFR_cumulative", {})
    aCFR_est = est_dict.get("aCFR_cumulative", {})
    its_est = est_dict.get("ITS_MLE", {})

    # Plot ground truth
    ax.plot(time_points, plot_data_dict["true_r_t"], label="True Factual", color='black', linestyle='--')
    ax.plot(time_points, plot_data_dict["true_rcf_0_t"], label="True Counterfactual", color='dimgray', linestyle=':')
    
    # Plot sCFR model estimates
    ax.plot(time_points, sCFR_est.get("mean", []), label="sCFR Factual", color='blue')
    ax.fill_between(time_points, sCFR_est.get("lower", []), sCFR_est.get("upper", []), color='blue', alpha=0.2, label="sCFR 95% CrI")
    ax.plot(time_points, sCFR_est.get("cf_mean", []), label="sCFR Counterfactual", color='deepskyblue', linestyle='-.')
    ax.fill_between(time_points, sCFR_est.get("cf_lower", []), sCFR_est.get("cf_upper", []), color='deepskyblue', alpha=0.15)

    # Plot benchmark estimates
    ax.plot(time_points, cCFR_est.get("mean", []), label="cCFR", color='red', linestyle=':')
    ax.fill_between(time_points, cCFR_est.get("lower", []), cCFR_est.get("upper", []), color='red', alpha=0.15)
    
    ax.plot(time_points, aCFR_est.get("mean", []), label="aCFR", color='green', linestyle='-.' )
    ax.fill_between(time_points, aCFR_est.get("lower", []), aCFR_est.get("upper", []), color='green', alpha=0.15)
    
    ax.plot(time_points, its_est.get("factual_mean", []), label="ITS Factual", color='purple', linestyle='--')
    ax.fill_between(time_points, its_est.get("factual_lower", []), its_est.get("factual_upper", []), color='purple', alpha=0.15)

    # Add vertical lines for interventions
    intervention_times = plot_data_dict.get("true_intervention_times_0_abs", [])
    for i, t_int in enumerate(intervention_times):
        if t_int < T_analyze:
            ax.axvline(x=t_int, color='red', linestyle='--', alpha=0.8, 
                       label=f"Intervention Start" if i == 0 else "")
    
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Case Fatality Rate ($r_t$)")
    ax.set_title(f"CFR Estimation: Scenario {scenario_id} (MC Run {mc_run_idx + 1})")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    
    plot_filename = os.path.join(output_dir, f"cfr_timeseries_scen{scenario_id}_{mc_run_idx+1}_run.pdf")
    plt.savefig(plot_filename)
    plt.close(fig)


def plot_aggregated_factual_summary(aggregated_plot_data, output_dir):
    """
    Generates a summarized 4x3 grid plot for FACTUAL estimates.
    """
    if not aggregated_plot_data: return
    fig, axes = plt.subplots(4, 3, figsize=(18, 20), sharex=True, sharey=False)
    
    aggregated_plot_data.sort(key=lambda x: x["scenario_id"])
    row_titles = ["Constant", "Linear", "Sinusoidal", "Gaussian"]
    col_titles = ["K=0", "K=1", "K=2"]

    for i, plot_data_dict in enumerate(aggregated_plot_data):
        if i >= 12: break
        row_idx, col_idx = i // 3, i % 3
        ax = axes[row_idx, col_idx]
        T_analyze = len(plot_data_dict["true_r_t"])
        time_points = np.arange(T_analyze)

        est_dict = plot_data_dict["estimated_r_t_dict"]
        sCFR_est = est_dict.get("sCFR", {}) 
        cCFR_est = est_dict.get("cCFR_cumulative", {})
        aCFR_est = est_dict.get("aCFR_cumulative", {})
        its_est = est_dict.get("ITS_MLE", {})

        ax.plot(time_points, plot_data_dict["true_r_t"], label="True Factual", color=MODEL_COLORS['True Factual'], linestyle='--')
        
        ax.plot(time_points, sCFR_est.get("mean", []), label="sCFR", color=MODEL_COLORS['sCFR'])
        ax.fill_between(time_points, sCFR_est.get("lower", []), sCFR_est.get("upper", []), color=MODEL_COLORS['sCFR'], alpha=0.2)
        
        ax.plot(time_points, cCFR_est.get("mean", []), label="cCFR", color=MODEL_COLORS['cCFR'], linestyle=':')
        ax.fill_between(time_points, cCFR_est.get("lower", []), cCFR_est.get("upper", []), color=MODEL_COLORS['cCFR'], alpha=0.15)
        
        ax.plot(time_points, aCFR_est.get("mean", []), label="aCFR", color=MODEL_COLORS['aCFR'], linestyle='-.')
        ax.fill_between(time_points, aCFR_est.get("lower", []), aCFR_est.get("upper", []), color=MODEL_COLORS['aCFR'], alpha=0.15)
        
        ax.plot(time_points, its_est.get("factual_mean", []), label="ITS", color=MODEL_COLORS['ITS'], linestyle='--')
        ax.fill_between(time_points, its_est.get("factual_lower", []), its_est.get("factual_upper", []), color=MODEL_COLORS['ITS'], alpha=0.15)

        intervention_times = plot_data_dict.get("true_intervention_times_0_abs", [])
        for t_int in intervention_times:
            if t_int < T_analyze:
                ax.axvline(x=t_int, color='red', linestyle='--', alpha=0.7)
        
        plotted_vals = np.concatenate([
            plot_data_dict["true_r_t"], sCFR_est.get("upper", []), 
            its_est.get("factual_upper", [])
        ])
        min_val, max_val = np.nanmin(plotted_vals), np.nanmax(plotted_vals)
        y_padding = (max_val - min_val) * 0.10 if (max_val - min_val) > 0 else 0.01
        ax.set_ylim(0, min(0.3, max_val + y_padding))

    # --- Set row and column titles ---
    for i, title in enumerate(row_titles): axes[i, 0].set_ylabel(title, fontsize=16, labelpad=15)
    for j, title in enumerate(col_titles): axes[0, j].set_title(title, fontsize=18, pad=10)

    # --- Generate the custom legend ---
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    # fig.supylabel("Baseline CFR Pattern", x=0.03, y=0.5, fontsize=18)
    # fig.supxlabel("Number of Interventions", x=0.5, y=0.97, fontsize=18)
    
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.03))
    fig.suptitle("Aggregated Factual CFR Estimation Across All Scenarios", fontsize=24, y=0.98)
    plt.tight_layout(rect=[0.05, 0.07, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "aggregate_summary_plot_F.pdf"))
    plt.close(fig)

def plot_aggregated_counterfactual_summary(aggregated_plot_data, output_dir):
    """
    Generates a summarized 4x3 grid plot for COUNTERFACTUAL estimates.
    """
    if not aggregated_plot_data: return
    fig, axes = plt.subplots(4, 3, figsize=(18, 20), sharex=True, sharey=False)

    aggregated_plot_data.sort(key=lambda x: x["scenario_id"])
    row_titles = ["Constant", "Linear", "Sinusoidal", "Gaussian"]
    col_titles = ["K=0", "K=1", "K=2"]

    for i, plot_data_dict in enumerate(aggregated_plot_data):
        if i >= 12: break
        row_idx, col_idx = i // 3, i % 3
        ax = axes[row_idx, col_idx]
        T_analyze = len(plot_data_dict["true_rcf_0_t"])
        time_points = np.arange(T_analyze)

        est_dict = plot_data_dict["estimated_r_t_dict"]
        sCFR_est = est_dict.get("sCFR", {}) 
        its_est = est_dict.get("ITS_MLE", {})

        ax.plot(time_points, plot_data_dict["true_rcf_0_t"], label="True Counterfactual", color=MODEL_COLORS['True Counterfactual'], linestyle=':')
        
        ax.plot(time_points, sCFR_est.get("cf_mean", []), label="sCFR", color=MODEL_COLORS['sCFR'])
        ax.fill_between(time_points, sCFR_est.get("cf_lower", []), sCFR_est.get("cf_upper", []), color=MODEL_COLORS['sCFR'], alpha=0.2)
        
        ax.plot(time_points, its_est.get("cf_mean", []), label="ITS", color=MODEL_COLORS['ITS'], linestyle='--')
        ax.fill_between(time_points, its_est.get("cf_lower", []), its_est.get("cf_upper", []), color=MODEL_COLORS['ITS'], alpha=0.15)

        intervention_times = plot_data_dict.get("true_intervention_times_0_abs", [])
        for t_int in intervention_times:
            if t_int < T_analyze:
                ax.axvline(x=t_int, color='red', linestyle='--', alpha=0.7)
        
        plotted_vals = np.concatenate([
            plot_data_dict["true_rcf_0_t"], sCFR_est.get("cf_upper", []), its_est.get("cf_upper", [])
        ])
        min_val, max_val = np.nanmin(plotted_vals), np.nanmax(plotted_vals)
        y_padding = (max_val - min_val) * 0.10 if (max_val - min_val) > 0 else 0.01
        ax.set_ylim(0, min(0.3, max_val + y_padding))

    # --- Set row and column titles ---
    for i, title in enumerate(row_titles): axes[i, 0].set_ylabel(title, fontsize=16, labelpad=15)
    for j, title in enumerate(col_titles): axes[0, j].set_title(title, fontsize=18, pad=10)

    # fig.supylabel("Baseline CFR Pattern", x=0.03, y=0.5, fontsize=18)
    # fig.supxlabel("Number of Interventions", x=0.5, y=0.97, fontsize=18)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.03))
    fig.suptitle("Aggregated Counterfactual CFR Estimation Across All Scenarios", fontsize=24, y=0.98)
    plt.tight_layout(rect=[0.05, 0.07, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "aggregate_summary_plot_CF.pdf"))
    plt.close(fig)

    
# def plot_aggregated_scenarios_summary(aggregated_plot_data, output_dir):
#     """
#     Generates a summarized 4x3 grid plot showing the average performance across all MC runs.
#     This version includes all requested enhancements to layout, legend, and y-axis scaling.

#     Args:
#         aggregated_plot_data (list): A list of dictionaries containing average curves and CIs.
#         output_dir (str): The directory to save the plot.
#     """
#     if not aggregated_plot_data:
#         print("Warning: No aggregated data provided for summary plot.")
#         return

#     fig, axes = plt.subplots(4, 3, figsize=(18, 22), sharex=True, sharey=False)
    
#     aggregated_plot_data.sort(key=lambda x: x["scenario_id"])

#     row_titles = ["Constant", "Linear", "Sinusoidal", "Gaussian"]
#     col_titles = ["K=0", "K=1", "K=2"]

#     for i, plot_data_dict in enumerate(aggregated_plot_data):
#         if i >= 12: break
#         row_idx, col_idx = i // 3, i % 3
#         ax = axes[row_idx, col_idx]
        
#         T_analyze = len(plot_data_dict["true_r_t"])
#         time_points = np.arange(T_analyze)
        
#         est_dict = plot_data_dict["estimated_r_t_dict"]
#         sCFR_est = est_dict.get("sCFR", {})
#         cCFR_est = est_dict.get("cCFR_cumulative", {})
#         aCFR_est = est_dict.get("aCFR_cumulative", {})
#         its_est = est_dict.get("ITS_MLE", {})

#         # Plot all curves
#         ax.plot(time_points, plot_data_dict["true_r_t"], label="True Factual", color='black', linestyle='--')
#         ax.plot(time_points, plot_data_dict["true_rcf_0_t"], label="True Counterfactual", color='dimgray', linestyle=':')
        
#         ax.plot(time_points, sCFR_est.get("mean", []), label="sCFR_F", color='blue')
#         ax.fill_between(time_points, sCFR_est.get("lower", []), sCFR_est.get("upper", []), color='blue', alpha=0.2, label="sCFR_F 95% CrI")
#         ax.plot(time_points, sCFR_est.get("cf_mean", []), label="sCFR_CF", color='deepskyblue', linestyle='-.')
#         ax.fill_between(time_points, sCFR_est.get("cf_lower", []), sCFR_est.get("cf_upper", []), color='deepskyblue', alpha=0.15, label="sCFR_CF 95% CrI")
        
#         ax.plot(time_points, cCFR_est.get("mean", []), label="cCFR", color='red', linestyle=':')
#         ax.fill_between(time_points, cCFR_est.get("lower", []), cCFR_est.get("upper", []), color='red', alpha=0.15, label="cCFR 95% CrI")
        
#         ax.plot(time_points, aCFR_est.get("mean", []), label="aCFR", color='green', linestyle='-.' )
#         ax.fill_between(time_points, aCFR_est.get("lower", []), aCFR_est.get("upper", []), color='green', alpha=0.15, label="aCFR 95% CrI")
        
#         ax.plot(time_points, its_est.get("factual_mean", []), label="ITS", color='purple', linestyle=(0, (3, 1, 1, 1)))
#         ax.fill_between(time_points, its_est.get("factual_lower", []), its_est.get("factual_upper", []), color='purple', alpha=0.15, label="ITS 95% CI")
#         ax.plot(time_points, its_est.get("cf_mean", []), label="ITS_CF", color='magenta', linestyle=':')
#         ax.fill_between(time_points, its_est.get("cf_lower", []), its_est.get("cf_upper", []), color='magenta', alpha=0.15, label="ITS_CF 95% CI")
        
#         intervention_times = plot_data_dict.get("true_intervention_times_0_abs", [])
#         for t_int in intervention_times:
#             if t_int < T_analyze:
#                 ax.axvline(x=t_int, color='red', linestyle='--', alpha=0.7)
    
#         ax.grid(True, linestyle=':', alpha=0.6)

#         # # Plot all curves with their specific labels for the legend
#         # ax.plot(time_points, plot_data_dict["true_r_t"], label="True Factual", color='black', linestyle='--')
#         # ax.plot(time_points, plot_data_dict["true_rcf_0_t"], label="True Counterfactual", color='dimgray', linestyle=':')
#         # ax.plot(time_points, sCFR_est.get("mean", []), label="sCFR_F", color='blue')
#         # ax.fill_between(time_points, sCFR_est.get("lower", []), sCFR_est.get("upper", []), color='blue', alpha=0.2, label="sCFR_F 95% CrI")
#         # ax.plot(time_points, sCFR_est.get("cf_mean", []), label="sCFR_CF", color='deepskyblue', linestyle='-.')
#         # ax.fill_between(time_points, sCFR_est.get("cf_lower", []), sCFR_est.get("cf_upper", []), color='deepskyblue', alpha=0.15, label="sCFR_CF 95% CrI")
        
#         # ax.plot(time_points, cCFR_est.get("mean", []), label="cCFR", color='red', linestyle=':')
#         # ax.fill_between(time_points, cCFR_est.get("lower", []), cCFR_est.get("upper", []), color='red', alpha=0.15, label="cCFR 95% CrI")
        
#         # ax.plot(time_points, aCFR_est.get("mean", []), label="aCFR", color='green', linestyle='-.' )
#         # ax.fill_between(time_points, aCFR_est.get("lower", []), aCFR_est.get("upper", []), color='green', alpha=0.15, label="aCFR 95% CrI")
        
#         # # ax.plot(time_points, its_est.get("factual_mean", []), label="ITS", color='purple', linestyle=(0, (3, 1, 1, 1))) # dashdotdot
#         # # ax.fill_between(time_points, its_est.get("factual_lower", []), its_est.get("factual_upper", []), color='purple', alpha=0.15, label="ITS 95% CI")

#         # # ITS factual estimate and its CI
#         # ax.plot(time_points, its_est.get("factual_mean", []), label="ITS (Factual)", color='purple', linestyle='--')
#         # ax.fill_between(time_points, its_est.get("factual_lower", []), its_est.get("factual_upper", []), color='purple', alpha=0.15, label="ITS 95% CI (Factual)")
        
#         # # ITS counterfactual estimate and its CI
#         # ax.plot(time_points, its_est.get("cf_mean", []), label="ITS (Counterfactual)", color='magenta', linestyle=':')
#         # ax.fill_between(time_points, its_est.get("cf_lower", []), its_est.get("cf_upper", []), color='magenta', alpha=0.15, label="ITS 95% CI (CF)")
        
#         # ax.grid(True, linestyle=':', alpha=0.6)

#         # Dynamic Y-axis zooming
#         plotted_vals = np.concatenate([
#             plot_data_dict["true_r_t"], sCFR_est.get("upper", []), aCFR_est.get("upper", []), its_est.get("factual_upper", [])
#         ])
#         min_val, max_val = np.nanmin(plotted_vals), np.nanmax(plotted_vals)
#         y_padding = (max_val - min_val) * 0.10 if (max_val - min_val) > 0 else 0.01
#         ax.set_ylim(0, min(0.3, max_val + y_padding))
        
#         # plotted_vals = np.concatenate([
#         #     plot_data_dict["true_r_t"], sCFR_est.get("upper", []), aCFR_est.get("upper", [])
#         # ])
#         # min_val, max_val = np.nanmin(plotted_vals), np.nanmax(plotted_vals)
#         # y_padding = (max_val - min_val) * 0.10 if (max_val - min_val) > 0 else 0.01
#         # ax.set_ylim(0, max_val + y_padding)

#     # --- Set row and column titles ---
#     for i, title in enumerate(row_titles): axes[i, 0].set_ylabel(title, fontsize=16, labelpad=15)
#     for j, title in enumerate(col_titles): axes[0, j].set_title(title, fontsize=18, pad=10)

#     # --- Generate the custom legend ---
#     handles, labels = axes[0, 0].get_legend_handles_labels()
#     label_to_handle = dict(zip(labels, handles))
    
#     row1_labels = ["True Factual", "sCFR_F", "sCFR_CF", "cCFR", "aCFR", "ITS_F", "ITS_CF"]
#     row2_labels = ["True Counterfactual", "sCFR_F 95% CrI", "sCFR_CF 95% CrI", "cCFR 95% CrI", "aCFR 95% CrI", "ITS_F 95% CI", "ITS_CF 95% CI"]
    
#     final_labels = row1_labels + row2_labels
#     final_handles = [label_to_handle.get(lbl, plt.Rectangle((0,0),1,1, fill=False, edgecolor='none', visible=False)) for lbl in final_labels]

#     fig.legend(final_handles, final_labels, loc='lower center', ncol=len(row1_labels), bbox_to_anchor=(0.5, 0.03), fontsize=12, frameon=True)
    
#     fig.supylabel("Baseline CFR Pattern", x=0.03, y=0.5, fontsize=18)
#     fig.supxlabel("Number of Interventions", x=0.5, y=0.97, fontsize=18)
#     fig.suptitle("Aggregated CFR Estimation Across All Scenarios", fontsize=24, y=1.0)
    
#     fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.92, hspace=0.25, wspace=0.05)
    
#     plot_filename = os.path.join(output_dir, "all_scenarios_aggregated_summary.pdf")
#     plt.savefig(plot_filename)
#     plt.close(fig)

#     # # --- Set row and column titles ---
#     # for i, title in enumerate(row_titles):
#     #     axes[i, 0].set_ylabel(title, fontsize=16, labelpad=15)
#     # for j, title in enumerate(col_titles):
#     #     axes[0, j].set_title(title, fontsize=18, pad=10)

#     # # --- FIX: Generate the custom two-row legend with correct order ---
#     # handles, labels = axes[0, 0].get_legend_handles_labels()
#     # label_to_handle = dict(zip(labels, handles))
    
#     # row1_labels = ["True Factual", "sCFR_F", "sCFR_CF", "cCFR", "aCFR"]
#     # row2_labels = ["True Counterfactual", "sCFR_F 95% CrI", "sCFR_CF 95% CrI", "cCFR 95% CrI", "aCFR 95% CrI"]
    
#     # # Add ITS to the legend if it was plotted
#     # if "ITS" in " ".join(label_to_handle.keys()):
#     #     row1_labels.append("ITS")
#     #     row2_labels.append("ITS 95% CI")
    
#     # final_labels = row1_labels + row2_labels
#     # final_handles = [label_to_handle.get(lbl, plt.Rectangle((0,0),1,1, fill=False, edgecolor='none', visible=False)) for lbl in final_labels]

#     # # Create the legend at the bottom with the appropriate number of columns
#     # ncol = len(row1_labels)
#     # fig.legend(final_handles, final_labels, loc='lower center', ncol=ncol, bbox_to_anchor=(0.5, 0.03), fontsize=14, frameon=True, framealpha=0.95)
    
#     # # --- FIX: Use super-titles and manual spacing for a clean layout ---
#     # fig.supylabel("Baseline CFR Pattern", x=0.03, y=0.5, fontsize=18)
#     # fig.suptitle("Aggregated CFR Estimation Across All 12 Scenarios", fontsize=24, y=0.99)
#     # fig.text(0.5, 0.95, "Number of Interventions", ha='center', va='center', fontsize=18)

#     # # Use subplots_adjust to manually set spacing and prevent warnings
#     # fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.91, hspace=0.25, wspace=0.05)
    
#     # plot_filename = os.path.join(output_dir, "all_scenarios_aggregated_summary.pdf")
#     # plt.savefig(plot_filename)
#     # plt.close(fig)

# def plot_aggregated_scenarios_summary(aggregated_plot_data, output_dir):
#     """
#     Generates a summarized 4x3 grid plot showing the average performance across all MC runs.
#     This version includes all requested enhancements to layout, y-axis scaling, and legend.

#     Args:
#         aggregated_plot_data (list): A list of dictionaries containing average curves and CIs.
#         output_dir (str): The directory to save the plot.
#     """
#     if not aggregated_plot_data:
#         print("Warning: No aggregated data provided for summary plot.")
#         return

#     fig, axes = plt.subplots(4, 3, figsize=(18, 22), sharex=True, sharey=True,
#                              gridspec_kw={'hspace': 0.1, 'wspace': 0.05})
    
#     aggregated_plot_data.sort(key=lambda x: x["scenario_id"])

#     # Define row and column titles for the grid
#     row_titles = ["Constant", "Linear", "Sinusoidal", "Gaussian"]
#     col_titles = ["K=0", "K=1", "K=2"]

#     for i, plot_data_dict in enumerate(aggregated_plot_data):
#         if i >= 12: break
#         row_idx, col_idx = i // 3, i % 3
#         ax = axes[row_idx, col_idx]
        
#         T_analyze = len(plot_data_dict["true_r_t"])
#         time_points = np.arange(T_analyze)

#         # Get data dictionaries for cleaner plotting code
#         est_dict = plot_data_dict["estimated_r_t_dict"]
#         sCFR_est = est_dict.get("sCFR", {})
#         cCFR_est = est_dict.get("cCFR_cumulative", {})
#         aCFR_est = est_dict.get("aCFR_cumulative", {})
#         its_est = plot_data_dict["estimated_r_t_dict"].get("ITS_NLS", {})

#         # Plot all curves with their specific labels for the legend
#         ax.plot(time_points, plot_data_dict["true_r_t"], label="True Factual", color='black', linestyle='--')
#         ax.plot(time_points, plot_data_dict["true_rcf_0_t"], label="True Counterfactual", color='dimgray', linestyle=':')
#         ax.plot(time_points, sCFR_est.get("mean", []), label="sCFR_F", color='blue')
#         ax.fill_between(time_points, sCFR_est.get("lower", []), sCFR_est.get("upper", []), color='blue', alpha=0.2, label="sCFR_F 95% CrI")
#         ax.plot(time_points, sCFR_est.get("cf_mean", []), label="sCFR_CF", color='deepskyblue', linestyle='-.')
#         ax.fill_between(time_points, sCFR_est.get("cf_lower", []), sCFR_est.get("cf_upper", []), color='deepskyblue', alpha=0.15, label="sCFR_CF 95% CrI")

#         ax.plot(time_points, its_est.get("factual_mean", []), label="ITS (Factual)", color='purple', linestyle='--')
#         ax.fill_between(time_points, its_est.get("factual_lower", []), its_est.get("factual_upper", []), color='purple', alpha=0.15, label="ITS 95% CI (Factual)")
#         ax.plot(time_points, its_est.get("cf_mean", []), label="ITS (Counterfactual)", color='magenta', linestyle=':')
#         ax.fill_between(time_points, its_est.get("cf_lower", []), its_est.get("cf_upper", []), color='magenta', alpha=0.15, label="ITS 95% CI (CF)")
        
#         ax.plot(time_points, cCFR_est.get("mean", []), label="cCFR", color='red', linestyle=':')
#         if "lower" in cCFR_est:
#             ax.fill_between(time_points, cCFR_est.get("lower", []), cCFR_est.get("upper", []), color='red', alpha=0.15, label="cCFR 95% CrI")
        
#         ax.plot(time_points, aCFR_est.get("mean", []), label="aCFR", color='green', linestyle='-.' )
#         if "lower" in aCFR_est:
#             ax.fill_between(time_points, aCFR_est.get("lower", []), aCFR_est.get("upper", []), color='green', alpha=0.15, label="aCFR 95% CrI")
        
#         ax.grid(True, linestyle=':', alpha=0.6)
        
#         # Dynamic Y-axis zooming for each subplot
#         plotted_vals = np.concatenate([
#             plot_data_dict["true_r_t"], sCFR_est.get("lower", []), sCFR_est.get("upper", [])
#         ])
#         min_val, max_val = np.nanmin(plotted_vals), np.nanmax(plotted_vals)
#         y_padding = (max_val - min_val) * 0.10 if (max_val - min_val) > 0 else 0.01
#         ax.set_ylim(max(0, min_val - y_padding), max_val + y_padding * 1.5) # Add a bit more top padding

#     # --- Set row and column titles ---
#     for i, title in enumerate(row_titles):
#         axes[i, 0].set_ylabel(title, fontsize=16, labelpad=15)
#     for j, title in enumerate(col_titles):
#         axes[0, j].set_title(title, fontsize=18, pad=10)

#     # --- Generate the custom two-row legend ---
#     handles, labels = axes[0, 0].get_legend_handles_labels()
#     label_to_handle = dict(zip(labels, handles))
    
#     # # Define the exact order and content for the two rows as you requested
#     # row1_labels = ["True Factual", "sCFR_F", "sCFR_CF", "cCFR", "aCFR"]
#     # row2_labels = ["True Counterfactual", "sCFR_F 95% CrI", "sCFR_CF 95% CrI", "cCFR 95% CrI", "aCFR 95% CrI"]

#     # Define the exact order and content for the two rows as you requested
#     row1_labels = ["True Factual", "sCFR_F", "sCFR_CF", "cCFR", "aCFR", "ITS (Factual)"]
#     row2_labels = ["True Counterfactual", "sCFR_F 95% CrI", "sCFR_CF 95% CrI", "cCFR 95% CrI", "aCFR 95% CrI", "ITS 95% CI (Factual)"]

    
#     # Build the final lists for the legend function in the correct order
#     final_labels = np.array([row1_labels,row2_labels]).T.reshape(-1,1).flatten().tolist()
#     final_handles = [label_to_handle.get(lbl, plt.Rectangle((0,0),1,1, fill=False, edgecolor='none', visible=False)) for lbl in final_labels]

#     print(final_labels)

#     # Create the legend at the bottom with 5 columns to force two rows
#     fig.legend(final_handles, final_labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.03), fontsize=14, frameon=True, framealpha=0.95)
    
#     fig.supylabel("Baseline CFR Pattern", x=0.01, y=0.5, fontsize=18)
#     fig.supxlabel("Number of Interventions", x=0.5, y=0.95, fontsize=18)
    
#     fig.suptitle("Aggregated CFR Estimation Across All 12 Scenarios", fontsize=24, y=.99)
    
#     fig.subplots_adjust(left=0.08, right=0.98, bottom=0.1, top=0.92, hspace=0.25, wspace=0.05)
    
#     plot_filename = os.path.join(output_dir, "all_scenarios_aggregated_summary.pdf")
#     plt.savefig(plot_filename, bbox_inches='tight')
#     plt.close(fig)
    
    # """
    # Generates a summarized 4x3 grid plot showing the average performance across all MC runs.
    # This version includes all requested enhancements to layout and legend.
    # """
    # if not aggregated_plot_data:
    #     print("Warning: No aggregated data provided for summary plot.")
    #     return

    # fig, axes = plt.subplots(4, 3, figsize=(18, 22), sharex=True, sharey='row',
    #                          gridspec_kw={'hspace': 0.1, 'wspace': 0.05})
    
    # aggregated_plot_data.sort(key=lambda x: x["scenario_id"])

    # row_titles = ["Constant", "Linear", "Sinusoidal", "Gaussian"]
    # col_titles = ["K=0", "K=1", "K=2"]

    # for i, plot_data_dict in enumerate(aggregated_plot_data):
    #     if i >= 12: break
    #     row_idx, col_idx = i // 3, i % 3
    #     ax = axes[row_idx, col_idx]
        
    #     T_analyze = len(plot_data_dict["true_r_t"])
    #     time_points = np.arange(T_analyze)

    #     # --- Plot all curves on the subplot ---
    #     est_dict = plot_data_dict["estimated_r_t_dict"]
    #     sCFR_est = est_dict.get("sCFR", {})
    #     cCFR_est = est_dict.get("cCFR_cumulative", {})
    #     aCFR_est = est_dict.get("aCFR_cumulative", {})

    #     # True curves
    #     ax.plot(time_points, plot_data_dict["true_r_t"], label="True Factual", color='black', linestyle='--')
    #     ax.plot(time_points, plot_data_dict["true_rcf_0_t"], label="True Counterfactual", color='dimgray', linestyle=':')
        
    #     # sCFR Factual
    #     ax.plot(time_points, sCFR_est.get("mean", []), label="sCFR_F", color='blue')
    #     ax.fill_between(time_points, sCFR_est.get("lower", []), sCFR_est.get("upper", []), color='blue', alpha=0.2, label="sCFR_F 95% CrI")
        
    #     # sCFR Counterfactual
    #     ax.plot(time_points, sCFR_est.get("cf_mean", []), label="sCFR_CF", color='deepskyblue', linestyle='-.')
    #     ax.fill_between(time_points, sCFR_est.get("cf_lower", []), sCFR_est.get("cf_upper", []), color='deepskyblue', alpha=0.15, label="sCFR_CF 95% CrI")

    #     # Benchmarks
    #     ax.plot(time_points, cCFR_est.get("mean", []), label="cCFR", color='red', linestyle=':')
    #     if "lower" in cCFR_est:
    #         ax.fill_between(time_points, cCFR_est.get("lower", []), cCFR_est.get("upper", []), color='red', alpha=0.15, label="cCFR 95% CrI")
        
    #     ax.plot(time_points, aCFR_est.get("mean", []), label="aCFR", color='green', linestyle='-.' )
    #     if "lower" in aCFR_est:
    #         ax.fill_between(time_points, aCFR_est.get("lower", []), aCFR_est.get("upper", []), color='green', alpha=0.15, label="aCFR 95% CrI")
        
    #     ax.grid(True, linestyle=':', alpha=0.7)
        
    #     # --- Y-axis zooming logic ---
    #     plotted_vals = np.concatenate([
    #         plot_data_dict["true_r_t"], sCFR_est.get("lower", []), sCFR_est.get("upper", [])
    #     ])
    #     min_val, max_val = np.nanmin(plotted_vals), np.nanmax(plotted_vals)
    #     y_padding = (max_val - min_val) * 0.10 if (max_val - min_val) > 0 else 0.01
    #     ax.set_ylim(max(0, min_val - y_padding), max_val + y_padding * 1.5) # Add a bit more top padding

    # # --- Set row and column titles ---
    # for i, title in enumerate(row_titles):
    #     axes[i, 0].set_ylabel(title, fontsize=16, labelpad=10)
    # for j, title in enumerate(col_titles):
    #     axes[0, j].set_title(title, fontsize=18, pad=10)

    # # --- Generate the custom two-row legend ---
    # handles, labels = axes[0, 0].get_legend_handles_labels()
    # label_to_handle = dict(zip(labels, handles))
    
    # # Define the exact order and content for the two rows
    # row1_labels = ["True Factual", "sCFR_F", "sCFR_CF", "cCFR", "aCFR"]
    # row2_labels = ["True Counterfactual", "sCFR_F 95% CrI", "sCFR_CF 95% CrI", "cCFR 95% CrI", "aCFR 95% CrI"]

    # # Build the final lists for the legend function, ensuring order is correct
    # final_labels = row1_labels + row2_labels
    # final_handles = [label_to_handle.get(lbl, plt.Rectangle((0,0),1,1, fill=False, edgecolor='none', visible=False)) for lbl in final_labels]

    # # Create the legend at the bottom with 5 columns (to force two rows)
    # fig.legend(final_handles, final_labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.01), fontsize=14)
    
    # fig.supylabel("Baseline CFR Pattern", x=0.01, y=0.5, fontsize=18)
    # fig.supxlabel("Number of Interventions", x=0.5, y=0.96, fontsize=18)
    
    # plt.tight_layout(rect=[0.05, 0.08, 1, 0.94])
    
    # plot_filename = os.path.join(output_dir, "all_scenarios_aggregated_summary.pdf")
    # plt.savefig(plot_filename)
    # plt.close()

def plot_all_scenarios_summary(all_first_run_plot_data, output_dir):
    """
    Generates a summarized big time series plot (4 rows x 3 columns) for all 12 scenarios.
    all_first_run_plot_data is a list of plot_data_dict for the first run of each scenario.
    """
    if not all_first_run_plot_data:
        print("No data for summary plot.")
        return

    fig, axes = plt.subplots(4, 3, figsize=(18, 20), sharex=True, sharey=True) # Adjust figsize
    axes = axes.flatten()

    # Sort plot data by scenario ID to match S01-S12 order
    all_first_run_plot_data.sort(key=lambda x: x["scenario_id"])

    for i, plot_data_dict in enumerate(all_first_run_plot_data):
        if i >= 12: break # Max 12 plots
        ax = axes[i]
        scenario_id = plot_data_dict["scenario_id"]
        T = len(plot_data_dict["true_r_t"])
        time_points = np.arange(T)

        ax.plot(time_points, plot_data_dict["true_r_t"], label="True Fact. $r_t$", color='black', linestyle='--')
        if "true_rcf_0_t" in plot_data_dict:
             ax.plot(time_points, plot_data_dict["true_rcf_0_t"], label="True Count. $r_t$", color='grey', linestyle=':')
        
        est_dict = plot_data_dict["estimated_r_t_dict"]
        
        if "sCFR" in est_dict:
            prop_est = est_dict["sCFR"]
            ax.plot(time_points, prop_est["mean"], label=r"sCFR Fact. $\hat{r}_t$", color='blue')
            ax.fill_between(time_points, prop_est["lower"], prop_est["upper"], color='blue', alpha=0.2)
            if "cf_mean" in prop_est:
                 ax.plot(time_points, prop_est["cf_mean"], label=r"sCFR Count. $\hat{r}_t$", color='cyan', linestyle='-.')
        
        # Plot Benchmarks with their CIs
        if "cCFR_cumulative" in est_dict:
            cCFR_est = est_dict["cCFR_cumulative"]
            ax.plot(time_points, cCFR_est["mean"], label="cCFR (Cumulative)", color='red', linestyle=':')
            if "lower" in cCFR_est: # Check if CIs exist
                ax.fill_between(time_points, cCFR_est["lower"], cCFR_est["upper"], color='red', alpha=0.15, label="cCFR 95% CrI (Beta-Binom)")
        
        if "aCFR_cumulative" in est_dict:
            aCFR_est = est_dict["aCFR_cumulative"]
            ax.plot(time_points, aCFR_est["mean"], label="aCFR (Nishiura Adj.)", color='green', linestyle='-.' )
            if "lower" in aCFR_est:
                ax.fill_between(time_points, aCFR_est["lower"], aCFR_est["upper"], color='green', alpha=0.15, label="aCFR 95% CrI (Beta-Binom)")

        intervention_times = plot_data_dict.get("true_intervention_times_0_abs", [])
        for t_int in intervention_times:
            if t_int < len(plot_data_dict["true_r_t"]):
                ax.axvline(x=t_int, color='red', linestyle='--', alpha=0.7)

        ax.set_title(f"Scenario {scenario_id}", fontsize=10)
        ax.tick_params(axis='x', rotation=30)
        if i % 3 == 0: # Leftmost column
            ax.set_ylabel("CFR ($r_t$)", fontsize=9)
        if i // 3 == 3: # Bottom row
            ax.set_xlabel("Time (days)", fontsize=9)
        
        ax.grid(True, linestyle=':', alpha=0.7)

    # Common Y-axis limit if sharey=True
    max_y_val = 0
    for p_data in all_first_run_plot_data:
        max_y_val = max(max_y_val, np.max(p_data["true_r_t"]))
        if "proposed" in p_data["estimated_r_t_dict"]:
            max_y_val = max(max_y_val, np.max(p_data["estimated_r_t_dict"]["proposed"]["upper"]))
    
    if max_y_val > 0:
        for ax in axes:
            ax.set_ylim(0, min(0.2, max_y_val * 1.2)) # Cap at 0.2 for typical CFRs or 1.2x max_y_val
    
    # Add a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), 5), bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout() # Adjust rect to make space for legend
    fig.suptitle("CFR Estimation Across All 12 Scenarios (First MC Run)", fontsize=16, y=0.99)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "all_scenarios_summary_timeseries.pdf"))
    plt.close()


def plot_metric_summary_boxplots(results_df, output_dir):
    """ (As previously defined, ensure benchmark names match those in collect_all_metrics) """
    # Metrics for r_t
    rt_metrics = ["mae_rt", "mciw_rt", "mcic_rt"]
    rt_models_to_plot = {
        "sCFR": "_sCFR", 
        "ITS": "_its",
        "aCFR": "_aCFR_cumulative",
        "cCFR": "_cCFR_cumulative"
    }
    
    for metric in rt_metrics:
        plot_df_list = []
        for scen_id in results_df["scenario_id"].unique():
            scen_df = results_df[results_df["scenario_id"] == scen_id]
            for model_name, model_suffix in rt_models_to_plot.items():
                col_name = f"{metric}{model_suffix}"
                if col_name in scen_df.columns:
                    plot_df_list.append(pd.DataFrame({
                        "scenario_id": scen_id, "method": model_name, "value": scen_df[col_name]
                    }))
        
        if not plot_df_list: continue
        plot_df_rt = pd.concat(plot_df_list)
        plot_df_rt.dropna(subset=['value'], inplace=True)

        if not plot_df_rt.empty:
            plt.figure(figsize=(15, 7))
            sns.boxplot(x="scenario_id", y="value", hue="method", data=plot_df_rt,
                        order=[s["id"] for s in config.SCENARIOS])
            plt.title(f"Factual r_t Summary: {metric.upper().replace('RT_','')}")
            plt.ylabel(metric.upper().replace('RT_',''))
            plt.xlabel("Scenario ID")
            plt.xticks(rotation=45, ha='right')
            plt.legend(title="Method")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"summary_boxplot_{metric}_rt.pdf"))
            plt.close()

    # --- Counterfactual rcf_t metrics: Compare sCFR and ITS ---
    rcf_metrics = ["mae_rcf", "mciw_rcf", "mcic_rcf"]
    rcf_models_to_plot = { "sCFR": "_sCFR", "ITS": "_its" }

    for metric in rcf_metrics:
        plot_df_list = []
        for scen_id in results_df["scenario_id"].unique():
            scen_df = results_df[results_df["scenario_id"] == scen_id]
            for model_name, model_suffix in rcf_models_to_plot.items():
                col_name = f"{metric}{model_suffix}"
                if col_name in scen_df.columns:
                    plot_df_list.append(pd.DataFrame({
                        "scenario_id": scen_id, "method": model_name, "value": scen_df[col_name]
                    }))
        
        if not plot_df_list: continue
        plot_df_rcf = pd.concat(plot_df_list)
        plot_df_rcf.dropna(subset=['value'], inplace=True)
        
        if not plot_df_rcf.empty:
            plt.figure(figsize=(15, 7))
            sns.boxplot(x="scenario_id", y="value", hue="method", data=plot_df_rcf,
                        order=[s["id"] for s in config.SCENARIOS])
            plt.title(f"Counterfactual r_t Summary: {metric.upper().replace('RCF_','')}")
            plt.ylabel(metric.upper().replace('RCF_',''))
            plt.xlabel("Scenario ID")
            plt.xticks(rotation=45, ha='right')
            plt.legend(title="Method")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"summary_boxplot_{metric}_rcf.pdf"))
            plt.close()

    # Similar logic for param_metrics as before
    param_metrics = ["bias", "width", "cover"]
    max_interventions = max(s["num_interventions_K_true"] for s in config.SCENARIOS)
            
    for k in range(1, max_interventions + 1):
        for param_prefix_base in ["beta_abs", "lambda"]:
            param_prefix = f"{param_prefix_base}_{k}"
            for metric_suffix in param_metrics:
                metric_col = f"{metric_suffix}_{param_prefix}"
                if metric_col in results_df.columns:
                    relevant_scenarios = [s["id"] for s in config.SCENARIOS if s["num_interventions_K_true"] >= k]
                    if not relevant_scenarios: continue
                    
                    plot_df_param = results_df[results_df["scenario_id"].isin(relevant_scenarios)][["scenario_id", metric_col]].copy()
                    plot_df_param.rename(columns={metric_col: "value"}, inplace=True)
                    plot_df_param.dropna(subset=['value'], inplace=True)
                    
                    if not plot_df_param.empty:
                        plt.figure(figsize=(10, 6))
                        sns.boxplot(x="scenario_id", y="value", data=plot_df_param, order=relevant_scenarios)
                        plt.title(f"Summary: {metric_suffix.capitalize()} for {param_prefix.replace('_', ' ').title()}")
                        plt.ylabel(metric_suffix.capitalize())
                        plt.xlabel("Scenario ID")
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f"summary_boxplot_{metric_col}.pdf"))
                        plt.close()

    param_metrics = ["bias", "width", "cover"]
    max_interventions = max(s["num_interventions_K_true"] for s in config.SCENARIOS)
    
    # Define which models to compare for parameters
    models_to_compare = {"sCFR": "_sCFR", "ITS": "_its"}
    
    for k in range(1, max_interventions + 1):
        # Loop through each parameter type: beta_abs and lambda
        for param_base in ["beta_abs", "lambda"]:
            for metric_suffix in param_metrics:
                plot_df_list = []
                # Get scenarios that actually have this intervention
                scen_list_for_plot = [s["id"] for s in config.SCENARIOS if s["num_interventions_K_true"] >= k]
                
                for scen_id in scen_list_for_plot:
                    scen_df = results_df[results_df["scenario_id"] == scen_id]
                    if scen_df.empty: continue
                    
                    # Melt data for both sCFR and ITS
                    for model_name, model_suffix_df in models_to_compare.items():
                        # For sCFR, the parameter is beta_abs. For ITS, it's also beta_abs.
                        param_base_df = "beta_abs" if model_name == "sCFR" and param_base == "beta_abs" else param_base
                        col_name = f"{metric_suffix}_{param_base_df}_{k}{model_suffix_df}"
                        
                        if col_name in scen_df.columns:
                            plot_df_list.append(pd.DataFrame({
                                "scenario_id": scen_id, 
                                "method": model_name, 
                                "value": scen_df[col_name]
                            }))
                
                if not plot_df_list: continue
                plot_df_param = pd.concat(plot_df_list)
                plot_df_param.dropna(subset=['value'], inplace=True)

                if not plot_df_param.empty:
                    plt.figure(figsize=(12, 6))
                    sns.boxplot(x="scenario_id", y="value", hue="method", data=plot_df_param, order=scen_list_for_plot)
                    
                    param_display_name = f"$\\beta_{{abs,{k}}}$" if param_base == "beta_abs" else f"$\\lambda_{{{k}}}$"
                    plt.title(f"Parameter Summary: {metric_suffix.capitalize()} for {param_display_name}")
                    plt.ylabel(metric_suffix.capitalize())
                    plt.xlabel("Scenario ID")
                    plt.xticks(rotation=45, ha='right')
                    plt.legend(title="Method")
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"summary_boxplot_{metric_suffix}_{param_base}_{k}.pdf"))
                    plt.close()