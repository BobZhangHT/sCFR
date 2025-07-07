# File: tables.py
# Description: Functions for generating and saving summary tables in LaTeX and CSV format.

import pandas as pd
import numpy as np
import os
import config

def format_mean_std(mean_val, std_val, is_coverage=False):
    """Formats a mean and standard deviation into a 'mean (std)' string."""
    if pd.isna(mean_val) or pd.isna(std_val):
        return "---"
    if is_coverage:
        return f"{mean_val:.2f} ({std_val:.2f})"
    return f"{mean_val:.4f} ({std_val:.4f})"

def generate_rt_metrics_table(results_df_summary, output_dir):
    """
    Generates a summary table for r_t metrics.
    """
    table_rows = []
    # Define all methods and their corresponding suffixes in the results data
    methods = {
        "sCFR": "_sCFR", 
        "cCFR": "_cCFR_cumulative", 
        "aCFR": "_aCFR_cumulative",
        "ITS": "_its"
    }

    for scen_conf in config.SCENARIOS:
        scen_id = scen_conf["id"]
        row_data = {"Scenario ID": scen_id}
        scen_res = results_df_summary[results_df_summary["scenario_id"] == scen_id]
        if scen_res.empty: continue

        for table_prefix, data_suffix in methods.items():
            for metric_base in ["mae_rt", "mciw_rt", "mcic_rt", "mae_rcf", "mciw_rcf", "mcic_rcf"]:
                metric_name_display = metric_base.upper().replace("_", "_")
                table_col_name = f"{table_prefix} {metric_name_display}"
                
                if table_prefix in ["cCFR", "aCFR"] and "rcf" in metric_base:
                    continue

                mean_col = f"{metric_base}{data_suffix}_mean"
                std_col = f"{metric_base}{data_suffix}_std"
                
                row_data[table_col_name] = format_mean_std(
                    scen_res[mean_col].iloc[0] if mean_col in scen_res else np.nan,
                    scen_res[std_col].iloc[0] if std_col in scen_res else np.nan,
                    is_coverage=("mcic" in metric_base)
                )
        table_rows.append(row_data)
        
    if not table_rows: return
    df_table = pd.DataFrame(table_rows)
    
    # Define the desired column order, excluding sCFR-O
    cols_ordered = [ "Scenario ID", 
                    "sCFR MAE_RT", "sCFR MCIW_RT", "sCFR MCIC_RT", 
                    "sCFR MAE_RCF", "sCFR MCIW_RCF", "sCFR MCIC_RCF",
                    "ITS MAE_RT", "ITS MCIW_RT", "ITS MCIC_RT",
                    "ITS MAE_RCF", "ITS MCIW_RCF", "ITS MCIC_RCF",
                    "cCFR MAE_RT", "cCFR MCIW_RT", "cCFR MCIC_RT", 
                    "aCFR MAE_RT", "aCFR MCIW_RT", "aCFR MCIC_RT" ]
    
    cols_to_use = [col for col in cols_ordered if col in df_table.columns]
    df_table_reordered = df_table[cols_to_use]
    
    # Save as CSV
    csv_filepath = os.path.join(output_dir, "summary_table_rt_metrics.csv")
    df_table_reordered.to_csv(csv_filepath, index=False, na_rep="N/A")
    print(f"Summary table for r_t metrics saved to {csv_filepath}")
    
    # Save as LaTeX
    latex_str = df_table_reordered.to_latex(index=False, escape=False, na_rep="---", column_format='l' + 'r' * (len(df_table_reordered.columns)-1))
    tex_filepath = os.path.join(output_dir, "summary_table_rt_metrics.tex")
    with open(tex_filepath, "w") as f: f.write(latex_str)
    print(f"LaTeX table for r_t metrics saved to {tex_filepath}")

def generate_param_metrics_table(results_df_summary, output_dir):
    """
    Generates and saves separate summary tables for the parameter estimation metrics
    of sCFR, sCFR-O, and the ITS benchmark.
    """
    max_interventions = max(s["num_interventions_K_true"] for s in config.SCENARIOS)
    
    # Define the models and their suffixes in the results data
    models_to_compare = {
        "sCFR": "_sCFR",
        "ITS": "_its"
    }

    # --- Loop through each model to generate a separate table ---
    for model_name, model_suffix in models_to_compare.items():
        table_rows = []
        # Only include scenarios that have interventions
        for scen_conf in config.SCENARIOS:
            if scen_conf["intervention_type_code"] == "I0":
                continue
            
            scen_id = scen_conf["id"]
            row_data = {"Scenario ID": scen_id}
            scen_res = results_df_summary[results_df_summary["scenario_id"] == scen_id]
            if scen_res.empty:
                continue

            num_current_interventions = scen_conf["num_interventions_K_true"]

            for k in range(1, max_interventions + 1):
                # Add data for this intervention if the scenario has it
                if k <= num_current_interventions:
                    for param_base in ["gamma", "lambda"]:
                        for metric_suffix in ["bias", "width", "cover"]:
                            # Construct the source column name from the results dataframe
                            mean_col = f"{metric_suffix}_{param_base}_{k}{model_suffix}_mean"
                            std_col = f"{metric_suffix}_{param_base}_{k}{model_suffix}_std"
                            
                            # Construct the final table header
                            metric_display = metric_suffix.capitalize().replace("Cover", "Coverage")
                            param_latex_name = f"${param_base}_{{{k}}}$"
                            table_col_name = f"{param_latex_name} {metric_display}"
                            
                            row_data[table_col_name] = format_mean_std(
                                scen_res[mean_col].iloc[0] if mean_col in scen_res else np.nan,
                                scen_res[std_col].iloc[0] if std_col in scen_res else np.nan,
                                is_coverage=(metric_suffix == "cover")
                            )
                else: # Pad with N/A for interventions not present in the scenario
                    for param_base in ["gamma", "lambda"]:
                        for metric_suffix in ["bias", "width", "cover"]:
                            metric_display = metric_suffix.capitalize().replace("Cover", "Coverage")
                            param_latex_name = f"${param_base}_{{{k}}}$"
                            row_data[f"{param_latex_name} {metric_display}"] = "N/A"
            table_rows.append(row_data)

        if not table_rows:
            print(f"No data available to generate parameter metrics table for {model_name}.")
            continue

        df_table = pd.DataFrame(table_rows)

        # Define column order for the table
        cols_ordered = ["Scenario ID"]
        for k in range(1, max_interventions + 1):
            for param_base in ["gamma", "lambda"]:
                param_latex = f"${param_base}_{{{k}}}$"
                for metric_suffix in ["Bias", "Width", "Coverage"]:
                    col_name = f"{param_latex} {metric_suffix}"
                    if col_name in df_table.columns:
                        cols_ordered.append(col_name)
        
        df_table_reordered = df_table[cols_ordered]
        
        # --- Save CSV with sanitized column names ---
        df_table_csv = df_table_reordered.copy()
        df_table_csv.columns = [col.replace('$\\gamma_{', 'gamma_').replace('$\\lambda_{', 'lambda_').replace('}$', '') for col in df_table_csv.columns]
        csv_filepath = os.path.join(output_dir, f"summary_table_param_metrics_{model_name}.csv")
        df_table_csv.to_csv(csv_filepath, index=False, na_rep="N/A")
        print(f"Summary table for {model_name} parameter metrics saved to {csv_filepath}")

        # --- Save as LaTeX ---
        latex_str = df_table_reordered.to_latex(index=False, escape=False, na_rep="---", 
                                         column_format='l' + 'r' * (len(df_table_reordered.columns)-1))
        tex_filepath = os.path.join(output_dir, f"summary_table_param_metrics_{model_name}.tex")
        with open(tex_filepath, "w") as f:
            f.write(latex_str)
        print(f"LaTeX table for {model_name} parameter metrics saved to {tex_filepath}")


# def generate_param_metrics_table(results_df_summary, output_dir):
#     """
#     Generates and saves a summary table for intervention parameter estimation metrics.

#     Args:
#         results_df_summary (pd.DataFrame): DataFrame with aggregated (mean and std) metrics.
#         output_dir (str): The directory where the output files will be saved.
#     """
#     table_rows = []
#     max_interventions = max(s["num_interventions_K_true"] for s in config.SCENARIOS)

#     for scen_conf in config.SCENARIOS:
#         scen_id = scen_conf["id"]
#         if scen_conf["intervention_type_code"] == "I0": continue
        
#         row_data = {"Scenario ID": scen_id}
#         scen_res = results_df_summary[results_df_summary["scenario_id"] == scen_id]
#         if scen_res.empty: continue

#         for k in range(1, max_interventions + 1):
#             # ** FIX APPLIED HERE: Look for "gamma" instead of "beta_abs" **
#             for param_base_name in ["gamma", "lambda"]:
#                 param_latex_name = f"$\\gamma_{{{k}}}$" if param_base_name == "gamma" else f"$\\lambda_{{{k}}}$"
#                 for metric_suffix in ["bias", "width", "cover"]:
#                     metric_latex_name = metric_suffix.capitalize().replace("Cover", "Coverage")
#                     col_name_latex = f"{param_latex_name} {metric_latex_name}"
                    
#                     if k <= scen_conf["num_interventions_K_true"]:
#                         mean_col = f"{metric_suffix}_{param_base_name}_{k}_mean"
#                         std_col = f"{metric_suffix}_{param_base_name}_{k}_std"
#                         row_data[col_name_latex] = format_mean_std(
#                             scen_res[mean_col].iloc[0] if mean_col in scen_res else np.nan,
#                             scen_res[std_col].iloc[0] if std_col in scen_res else np.nan,
#                             is_coverage=("cover" in metric_suffix))
#                     else:
#                         row_data[col_name_latex] = "N/A"
#         table_rows.append(row_data)

#     if not table_rows: return
#     df_table = pd.DataFrame(table_rows)
    
#     if not df_table.empty:
#         # Define the desired column order dynamically
#         cols_ordered = ["Scenario ID"]
#         for k in range(1, max_interventions + 1):
#             for param_base in ["gamma", "lambda"]:
#                 param_latex = f"$\\gamma_{{{k}}}$" if param_base == "gamma" else f"$\\lambda_{{{k}}}$"
#                 for metric_suffix in ["Bias", "Width", "Coverage"]:
#                     cols_ordered.append(f"{param_latex} {metric_suffix}")
            
#             cols_to_use = [col for col in cols_ordered if col in df_table.columns]
#             df_table_reordered = df_table[cols_to_use]

#         # Save as CSV with sanitized column names
#         df_table_csv = df_table_reordered.copy()
#         df_table_csv.columns = [col.replace('$\\beta_{', 'beta_').replace(',abs}}$', '_abs').replace('$\\lambda_{', 'lambda_').replace('}$', '') for col in df_table_csv.columns]
#         csv_filepath = os.path.join(output_dir, "summary_table_param_metrics.csv")
#         df_table_csv.to_csv(csv_filepath, index=False, na_rep="N/A")
#         print(f"Summary table for parameter metrics saved to {csv_filepath}")

#         # Save as LaTeX
#         latex_str = df_table_reordered.to_latex(index=False, escape=False, na_rep="---", 
#                                      column_format='l' + 'r' * (len(df_table_reordered.columns)-1))
#         tex_filepath = os.path.join(output_dir, "summary_table_param_metrics.tex")
#         with open(tex_filepath, "w") as f: f.write(latex_str)
#         print(f"LaTeX table for parameter metrics saved to {tex_filepath}")
#     else:
#         print("No data available to generate parameter metrics table.")