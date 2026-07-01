"""
Generate the intervention-magnitude LaTeX table (Table beta_abs_mae_Kpos_nocov).

Reads the aggregated beta-magnitude MAE summary CSV produced by the simulation
analysis and writes a booktabs LaTeX table comparing sCFR and fsCFR on the step
and slope MAE for scenarios with interventions (K > 0), two rows per scenario
(sCFR, fsCFR). Adjust CSV_PATH / OUT_TEX below to point at the desired result set.

Usage:
    python table_gen.py
"""

import pandas as pd
import numpy as np

# -----------------------------
# User inputs
# -----------------------------
CSV_PATH = "./simulation_outputs/server_plots/beta_abs_metrics_summary.csv"   # or full path
OUT_TEX  = "./simulation_outputs/server_plots/beta_abs_mae_Kpos_nocov.tex"

CAPTION = (
    "Summary of estimation performance for intervention magnitude parameters "
    "($\\beta_{\\mathrm{abs}}$) across scenarios with interventions ($K>0$). "
    "Entries report mean absolute error (MAE) for sCFR and the frequentist benchmark (fsCFR)."
)
LABEL = "tab:beta_abs_mae_Kpos_nocov"

# -----------------------------
# Helpers
# -----------------------------
def fmt(x):
    """Format numeric values to 3 decimals; NaN -> '--'."""
    if pd.isna(x):
        return "--"
    return f"{float(x):.3f}"

# -----------------------------
# Load + filter
# -----------------------------
df = pd.read_csv(CSV_PATH)

# Keep only scenarios with interventions
df = df.loc[df["K"] > 0].copy()

# Required columns (MAE only; no coverage)
cols_needed = [
    "scenario_id", "K",
    "beta_abs_step_1_mae_sCFR",  "beta_abs_slope_1_mae_sCFR",
    "beta_abs_step_1_mae_fsCFR", "beta_abs_slope_1_mae_fsCFR",
    "beta_abs_step_2_mae_sCFR",  "beta_abs_slope_2_mae_sCFR",
    "beta_abs_step_2_mae_fsCFR", "beta_abs_slope_2_mae_fsCFR",
]
missing = [c for c in cols_needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

# -----------------------------
# Build LaTeX table body (two rows per scenario: sCFR, fsCFR)
# -----------------------------
lines = []
for _, row in df.iterrows():
    scen = f'{row["scenario_id"]} ($K={int(row["K"])}$)'

    # sCFR row
    s_step1 = fmt(row["beta_abs_step_1_mae_sCFR"])
    s_slope1 = fmt(row["beta_abs_slope_1_mae_sCFR"])
    s_step2 = fmt(row["beta_abs_step_2_mae_sCFR"])
    s_slope2 = fmt(row["beta_abs_slope_2_mae_sCFR"])

    # fsCFR row
    f_step1 = fmt(row["beta_abs_step_1_mae_fsCFR"])
    f_slope1 = fmt(row["beta_abs_slope_1_mae_fsCFR"])
    f_step2 = fmt(row["beta_abs_step_2_mae_fsCFR"])
    f_slope2 = fmt(row["beta_abs_slope_2_mae_fsCFR"])

    lines.append(
        rf"\multirow{{2}}{{*}}{{{scen}}} & sCFR  & {s_step1} & {s_slope1} & {s_step2} & {s_slope2} \\"
    )
    lines.append(
        rf" & fsCFR & {f_step1} & {f_slope1} & {f_step2} & {f_slope2} \\"
    )
    lines.append(r"\addlinespace")

# drop the last \addlinespace
if lines and lines[-1] == r"\addlinespace":
    lines = lines[:-1]

body = "\n".join(lines)

# -----------------------------
# Assemble full LaTeX table
# -----------------------------
tex = rf"""\begin{{table}}[!t]
\centering
\caption{{{CAPTION}}}
\label{{{LABEL}}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llcccc}}
\toprule
\multirow{{2}}{{*}}{{Scenario}} & \multirow{{2}}{{*}}{{Method}}
& \multicolumn{{2}}{{c}}{{Intervention 1}} & \multicolumn{{2}}{{c}}{{Intervention 2}} \\
\cmidrule(lr){{3-4}}\cmidrule(lr){{5-6}}
& & Step MAE & Slope MAE & Step MAE & Slope MAE \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
"""

# -----------------------------
# Write to file
# -----------------------------
with open(OUT_TEX, "w", encoding="utf-8") as f:
    f.write(tex)

print(f"Wrote LaTeX table to: {OUT_TEX}")
