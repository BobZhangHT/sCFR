#!/usr/bin/env python3
"""
UK COVID-19 Case Fatality Rate Analysis.

This script analyzes time-varying factual and counterfactual CFR for the UK
using the Bayesian sCFR model and benchmark methods.

Usage:
    python real_data_analysis.py          # sCFR only
    python real_data_analysis.py --full   # Include all benchmarks

Outputs saved to ./real_data_outputs/
"""

import os
import warnings
import argparse
import pandas as pd
import numpy as np
import jax
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import config
import data_generation
import methods

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Parse arguments
parser = argparse.ArgumentParser(description='UK COVID-19 CFR Analysis')
parser.add_argument('--full', action='store_true', help='Include all benchmark models')
args = parser.parse_args()

PLOT_FULL = args.full

# Output directory
OUTPUT_DIR = "./real_data_outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Outputs will be saved to: {OUTPUT_DIR}")
print(f"Full comparison mode: {PLOT_FULL}")

# =============================================================================
# Load and Prepare UK Data
# =============================================================================

print("Loading WHO COVID-19 data...")
df_who = pd.read_csv("./WHO-COVID-19-global-daily-data.csv", encoding='unicode_escape')
df_uk = df_who[df_who['Country'] == "United Kingdom of Great Britain and Northern Ireland"].copy()

df_uk['Date_reported'] = pd.to_datetime(df_uk['Date_reported'])
start_date = '2020-03-01'
end_date = '2021-12-31'
df_period = df_uk[(df_uk['Date_reported'] >= start_date) & (df_uk['Date_reported'] <= end_date)].copy()

dates = df_period['Date_reported'].values
ct_raw = np.nan_to_num(df_period['New_cases'].values)
dt_raw = np.nan_to_num(df_period['New_deaths'].values)

ct = pd.Series(ct_raw).rolling(window=7, min_periods=1).mean().values.astype(int)
dt = pd.Series(dt_raw).rolling(window=7, min_periods=1).mean().values.astype(int)

N_obs = len(ct)
print(f"Data loaded for the period: {start_date} to {end_date} ({N_obs} days)")

# =============================================================================
# Plot Cases and Deaths
# =============================================================================

print("Generating cases and deaths plot...")
fig, ax1 = plt.subplots(figsize=(12, 7))
plt.style.use('seaborn-v0_8-paper')

ax1.plot(dates, ct, color='darkblue', label='Confirmed Cases (7-day avg)', linewidth=1.5)
ax1.set_ylabel('Confirmed Cases', color='darkblue', fontsize=16)
ax1.set_xlabel('Date', fontsize=16)
ax1.tick_params(axis='y', labelcolor='darkblue', labelsize=14)
ax1.tick_params(axis='x', labelsize=14)

ax2 = ax1.twinx()
ax2.plot(dates, dt, color='darkred', label='Deaths (7-day avg)', linewidth=1.5)
ax2.set_ylabel('Deaths', color='darkred', fontsize=16)
ax2.tick_params(axis='y', labelcolor='darkred', labelsize=14)

wave_periods = {
    'Wave 1': ('2020-03-01', '2020-07-01'),
    'Wave 2 ': ('2020-09-01', '2021-05-01'),
    'Wave 3 ': ('2021-06-01', '2021-12-01')
}

for wave, (start, end) in wave_periods.items():
    ax1.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='gray', alpha=0.15, zorder=0)

fig.suptitle('UK Daily Confirmed Cases and Deaths', fontsize=20, weight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.99])
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=14)

plt.savefig(os.path.join(OUTPUT_DIR, "uk_cases_and_deaths.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, "uk_cases_and_deaths.png"), dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved: uk_cases_and_deaths.pdf and .png")

# =============================================================================
# Define Interventions
# =============================================================================

print("\nSetting up intervention times...")
intervention_dates = [
    '2020-03-23', '2020-06-15',  # Lockdown 1
    '2020-11-05', '2020-12-02',  # Lockdown 2
    '2021-01-06', '2021-04-12'   # Lockdown 3
]
intervention_times_abs = [(pd.to_datetime(d) - pd.to_datetime(start_date)).days for d in intervention_dates]
intervention_signs = np.array([-1, 1, -1, 1, -1, 1])
K_interventions = len(intervention_times_abs)

print(f"Interventions defined at days: {intervention_times_abs}")

# =============================================================================
# Prepare Model Data
# =============================================================================

N_SPLINE_KNOTS_J = 20
f_s = data_generation.generate_delay_distribution(N_obs, config.F_MEAN, config.F_SHAPE)
Q_matrix = methods.construct_Q_matrix(ct, f_s, N_obs)
Bm_matrix = data_generation.generate_bspline_basis(N_obs, N_SPLINE_KNOTS_J, config.SPLINE_ORDER)
Z_input_matrix = data_generation.generate_intervention_input_matrix(
    np.arange(N_obs), intervention_times_abs, K_interventions
)

sim_data_dict = {
    "c_t": ct, "d_t": dt, "f_s_true": f_s, "Q_true": Q_matrix, 
    "Bm_true": Bm_matrix, "Z_input_true": Z_input_matrix,
    "beta_signs_true": intervention_signs, "N_obs": N_obs,
    "K_spline_obs": N_SPLINE_KNOTS_J,
    "num_interventions_true_K": K_interventions,
    "true_intervention_times_0_abs": intervention_times_abs
}

# =============================================================================
# Fit sCFR Model
# =============================================================================

print("\nFitting the proposed sCFR model...")
jax_key = jax.random.PRNGKey(config.GLOBAL_BASE_SEED)
posterior_samples, _ = methods.fit_proposed_model(sim_data_dict, jax_key)
print("sCFR model fitting complete.")

if PLOT_FULL:
    print("Running benchmark models...")
    benchmark_r_t_estimates = {
        "cCFR_model": methods.cCFR_model(sim_data_dict["d_t"], sim_data_dict["c_t"], cumulative=True),
        "aCFR_model": methods.aCFR_model(sim_data_dict["d_t"], sim_data_dict["c_t"], sim_data_dict["f_s_true"])
    }
    its_results = methods.fsCFR_model(
        d_t=sim_data_dict["d_t"], c_t=sim_data_dict["c_t"], 
        f_s=sim_data_dict["f_s_true"],
        Bm=sim_data_dict["Bm_true"],
        intervention_times_abs=sim_data_dict["true_intervention_times_0_abs"],
        intervention_signs=sim_data_dict["beta_signs_true"]
    )
    benchmark_outputs = {**benchmark_r_t_estimates, **its_results}
    print("Benchmark models complete.")
else:
    benchmark_outputs = {}
    print("Skipping benchmark models (PLOT_FULL=False).")

# =============================================================================
# Plot CFR Estimates
# =============================================================================

print("\nGenerating CFR plot...")
fig, ax = plt.subplots(figsize=(12, 7))
plt.style.use('seaborn-v0_8-paper')

r_t_key = 'r_t' if 'r_t' in posterior_samples else 'p'
r_cf_key = 'r_cf' if 'r_cf' in posterior_samples else 'p_cf'
p_mean = np.mean(posterior_samples[r_t_key], axis=0)
p_lower, p_upper = np.percentile(posterior_samples[r_t_key], [2.5, 97.5], axis=0)
p_cf_mean = np.mean(posterior_samples[r_cf_key], axis=0)

# Plot sCFR
ax.plot(dates, p_mean, label='sCFR (Factual)', color='tab:blue', linewidth=2.0, zorder=5)
ax.fill_between(dates, p_lower, p_upper, color='tab:blue', alpha=0.2, label='sCFR 95% CrI')
ax.plot(dates, p_cf_mean, label='sCFR (Counterfactual)', color='tab:cyan', linestyle='--', linewidth=2.0, zorder=4)

if PLOT_FULL:
    ax.plot(dates, benchmark_outputs['cCFR_model'], label='cCFR', color='tab:green', linestyle=':', linewidth=2.0)
    ax.plot(dates, benchmark_outputs['aCFR_model'], label='aCFR', color='tab:red', linestyle='-.', linewidth=2.0)
    ax.plot(dates, benchmark_outputs['fsCFR_factual_mean'], label='fsCFR (Factual)', color='tab:orange', linestyle='--', linewidth=2.0)
    ax.plot(dates, benchmark_outputs['fsCFR_counterfactual_mean'], label='fsCFR (Counterfactual)', color='tab:brown', linestyle=':', linewidth=2.0)

# Add intervention annotations
intervention_labels = [
    'Lockdown 1 Start', 'Lockdown 1 Lift',
    'Lockdown 2 Start', 'Lockdown 2 Lift',
    'Lockdown 3 Start', 'Lockdown 3 Lift'
]

y_max = ax.get_ylim()[1]
annotation_y_levels = [y_max * 0.85, y_max * 0.7, y_max * 0.55]
text_y_offsets = [-25, 25, -25, 25, -25, 25]

for i, t_int in enumerate(intervention_times_abs):
    date = dates[t_int]
    label = intervention_labels[i]
    y_level = annotation_y_levels[i // 2] + 0.05
    y_offset = text_y_offsets[i]
    
    ax.axvline(x=date, color='black', linestyle='--', alpha=0.7)
    ax.annotate(label, xy=(date, y_level), xytext=(5, y_offset), 
                textcoords='offset points', ha='left', va='center', fontsize=11, 
                arrowprops=dict(arrowstyle='-[, widthB=0.5, lengthB=0.2', lw=1.0))

ax.set_title(
    'Comparison of CFR Estimation Methods for the UK' if PLOT_FULL else 'sCFR Estimates for the UK',
    fontsize=20, weight='bold'
)
ax.set_xlabel('Date', fontsize=16)
ax.set_ylabel('Case Fatality Rate', fontsize=16)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.legend(loc='upper right', fontsize=11)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()

plot_suffix = "uk_cfr_comparison" if PLOT_FULL else "uk_scfr_only"
plt.savefig(os.path.join(OUTPUT_DIR, f"{plot_suffix}.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, f"{plot_suffix}.png"), dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {plot_suffix}.pdf and .png")

# =============================================================================
# Save Results
# =============================================================================

print("\nSaving results to CSV...")
results_df = pd.DataFrame({
    'date': dates,
    'new_cases': ct,
    'new_deaths': dt,
    'scfr_factual': p_mean,
    'scfr_factual_lower': p_lower,
    'scfr_factual_upper': p_upper,
    'scfr_counterfactual': p_cf_mean,
})
if PLOT_FULL:
    results_df['ccfr'] = benchmark_outputs['cCFR_model']
    results_df['acfr'] = benchmark_outputs['aCFR_model']
    results_df['fscfr_factual'] = benchmark_outputs['fsCFR_factual_mean']
    results_df['fscfr_counterfactual'] = benchmark_outputs['fsCFR_counterfactual_mean']
results_df.to_csv(os.path.join(OUTPUT_DIR, "uk_cfr_results.csv"), index=False)
print("  Saved: uk_cfr_results.csv")

# =============================================================================
# Beta Estimates Table (sCFR and fsCFR)
# =============================================================================

def _fmt_num(x, ndec=3):
    """Format number for display; None/NaN -> '---'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    return f"{float(x):.{ndec}f}"

def _fmt_interval(mean, lo, hi, ndec=3):
    """Format mean [lo, hi] for CrI."""
    return f"{mean:.{ndec}f} [{lo:.{ndec}f}, {hi:.{ndec}f}]"

intervention_labels = [
    'Lockdown 1 Start', 'Lockdown 1 Lift',
    'Lockdown 2 Start', 'Lockdown 2 Lift',
    'Lockdown 3 Start', 'Lockdown 3 Lift'
]

# sCFR: posterior mean and 95% CrI for beta_abs (step) and beta_slope_abs (slope)
K = K_interventions
beta_rows = []
if K > 0 and "beta_abs" in posterior_samples and "beta_slope_abs" in posterior_samples:
    beta_abs = np.asarray(posterior_samples["beta_abs"])   # (n_samples, K)
    beta_slope_abs = np.asarray(posterior_samples["beta_slope_abs"])  # (n_samples, K)
    if beta_abs.ndim == 1:
        beta_abs = np.expand_dims(beta_abs, 0)
    if beta_slope_abs.ndim == 1:
        beta_slope_abs = np.expand_dims(beta_slope_abs, 0)
    n_k = min(K, beta_abs.shape[1], beta_slope_abs.shape[1])
    for k in range(n_k):
        row = {
            "intervention": intervention_labels[k] if k < len(intervention_labels) else f"Intervention {k+1}",
            "scfr_step_mean": np.mean(beta_abs[:, k]),
            "scfr_step_lower": np.percentile(beta_abs[:, k], 2.5),
            "scfr_step_upper": np.percentile(beta_abs[:, k], 97.5),
            "scfr_slope_mean": np.mean(beta_slope_abs[:, k]),
            "scfr_slope_lower": np.percentile(beta_slope_abs[:, k], 2.5),
            "scfr_slope_upper": np.percentile(beta_slope_abs[:, k], 97.5),
        }
        if PLOT_FULL and benchmark_outputs:
            est_step = benchmark_outputs.get("fsCFR_beta_abs_est")
            est_slope = benchmark_outputs.get("fsCFR_beta_slope_abs_est")
            row["fscfr_step"] = float(est_step[k]) if est_step is not None and k < len(est_step) else np.nan
            row["fscfr_slope"] = float(est_slope[k]) if est_slope is not None and k < len(est_slope) else np.nan
        else:
            row["fscfr_step"] = np.nan
            row["fscfr_slope"] = np.nan
        beta_rows.append(row)
else:
    for k in range(K):
        beta_rows.append({
            "intervention": intervention_labels[k] if k < len(intervention_labels) else f"Intervention {k+1}",
            "scfr_step_mean": np.nan, "scfr_step_lower": np.nan, "scfr_step_upper": np.nan,
            "scfr_slope_mean": np.nan, "scfr_slope_lower": np.nan, "scfr_slope_upper": np.nan,
            "fscfr_step": np.nan, "fscfr_slope": np.nan,
        })

beta_df = pd.DataFrame(beta_rows)
beta_csv_path = os.path.join(OUTPUT_DIR, "beta_estimates.csv")
beta_df.to_csv(beta_csv_path, index=False)
print("  Saved: beta_estimates.csv")

# LaTeX three-line table for Overleaf
def _tex_escape(s):
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")

def _tex_num(x, ndec=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    return f"{float(x):.{ndec}f}"

def _extract_phase_and_intervention(interv_str):
    """Extract phase number and intervention type from string like 'Lockdown 1 Start'."""
    parts = interv_str.split()
    if len(parts) >= 3 and parts[0] == "Lockdown":
        phase = parts[1]
        interv_type = " ".join(parts[2:])
        sign = "(-1)" if "Start" in interv_type else "(+1)"
        interv_short = "Start" + sign if "Start" in interv_type else "Lift" + sign
        return phase, interv_short
    return "?", interv_str

# Preamble comment for Overleaf: add \usepackage{booktabs} to your document preamble if not already present.
tex_lines = [
    "% Requires \\usepackage{booktabs} in your LaTeX preamble.",
    "\\begin{table}[htbp]",
    "\\centering",
    "\\caption{Estimated intervention effect magnitudes (absolute value) for the UK COVID-19 analysis. "
    "We provide both posterior mean and 95\\% credible interval for sCFR, and only point estimate for fsCFR. "
    "We denote the absolute value of immediate level change as $\\beta_{\\mathrm{abs}}^{(L)}$ while the gradual slope change as $\\beta_{\\mathrm{abs}}^{(S)}$.}",
    "\\label{tab:uk_beta_estimates}",
    "\\resizebox{\\textwidth}{!}{%",
    "\\begin{tabular}{clcccc}",
    "\\toprule",
    "\\shortstack{Lockdown\\\\Phase} & Intervention & \\multicolumn{2}{c}{$\\beta_{\\mathrm{abs}}^{(L)}$} & \\multicolumn{2}{c}{$\\beta_{\\mathrm{abs}}^{(S)}$} \\\\",
    "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
    "& & sCFR (95\\% CrI) & fsCFR & sCFR (95\\% CrI) & fsCFR \\\\",
    "\\midrule",
]
for _, r in beta_df.iterrows():
    def _fmt_cri(mean, lo, hi):
        if pd.isna(mean) or pd.isna(lo) or pd.isna(hi):
            return "---"
        return _fmt_interval(float(mean), float(lo), float(hi))
    scfr_step_str = _fmt_cri(r["scfr_step_mean"], r["scfr_step_lower"], r["scfr_step_upper"])
    scfr_slope_str = _fmt_cri(r["scfr_slope_mean"], r["scfr_slope_lower"], r["scfr_slope_upper"])
    fscfr_s = _tex_num(r.get("fscfr_step"))
    fscfr_sl = _tex_num(r.get("fscfr_slope"))
    interv = str(r["intervention"])
    phase, interv_short = _extract_phase_and_intervention(interv)
    # Only show phase number on first row of each phase group
    if interv.endswith("Start"):
        phase_display = phase
    else:
        phase_display = ""
    tex_lines.append(f"{phase_display} & {interv_short} & {scfr_step_str} & {fscfr_s} & {scfr_slope_str} & {fscfr_sl} \\\\")
tex_lines.extend([
    "\\bottomrule",
    "\\end{tabular}%",
    "}",
    "\\end{table}",
])
tex_content = "\n".join(tex_lines)
tex_path = os.path.join(OUTPUT_DIR, "beta_estimates.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(tex_content)
print("  Saved: beta_estimates.tex (ready for Overleaf)")

print(f"\n{'='*80}")
print("Analysis complete! All outputs saved to:", OUTPUT_DIR)
print(f"{'='*80}")
