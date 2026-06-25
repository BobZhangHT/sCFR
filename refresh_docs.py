r"""
refresh_docs.py

After a full-scale run (on the server, then downloaded locally), this script
syncs every result that docs/main.tex consumes:

  * copies all result figure PDFs into docs/figs_tables/
  * regenerates / copies all data-driven LaTeX tables into docs/figs_tables/

docs/main.tex uses \graphicspath{{figs_tables/}} for figures and
\input{figs_tables/...} for tables, so a plain `pdflatex` picks up the latest
numbers with no hand editing after this script runs. Run from the repo root:

    C:/Users/bobzh/anaconda3/python.exe refresh_docs.py

Figures consumed: aggregated_factual_summary, aggregated_counterfactual_summary,
  combined_metrics_summary, effectiveness_summary, runtime_scaling,
  uk_cases_and_deaths, uk_cfr_comparison.

Tables consumed: beta_abs_mae_Kpos_nocov (regenerated from CSV),
  uk_beta_estimates, knot_sensitivity, prior_sensitivity, misspecification.

Missing inputs (experiment not yet run) are reported and skipped; the
corresponding docs file is left untouched.
"""

import os
import shutil

import numpy as np
import pandas as pd

import config

DOCS = "docs"
# All paper-consumed figures and tables live in one folder that main.tex reads from
# (\graphicspath{{figs_tables/}} for figures, \input{figs_tables/...} for tables).
FIGS_TABLES_DIR = os.path.join(DOCS, "figs_tables")
TABLES_DIR = FIGS_TABLES_DIR

# ---------------------------------------------------------------------------
# Figures: dest file name -> ordered list of candidate source paths
# ---------------------------------------------------------------------------
FIGURES = {
    "aggregated_factual_summary.pdf":
        ["simulation_outputs/plots/aggregated_factual_summary.pdf"],
    "aggregated_counterfactual_summary.pdf":
        ["simulation_outputs/plots/aggregated_counterfactual_summary.pdf"],
    "combined_metrics_summary.pdf":
        ["simulation_outputs/plots/combined_metrics_summary.pdf"],
    "effectiveness_summary.pdf":
        ["simulation_outputs/plots/effectiveness_summary.pdf"],
    "runtime_scaling.pdf":
        ["simulation_outputs/plots/runtime_scaling.pdf"],
    "uk_cases_and_deaths.pdf":
        ["real_data_outputs/uk_cases_and_deaths.pdf"],
    "uk_cfr_comparison.pdf":
        ["real_data_outputs/uk_cfr_comparison.pdf"],
}

# Table files copied verbatim from the generating script's output directory.
TABLE_COPIES = {
    "uk_beta_estimates.tex":
        ["real_data_outputs/beta_estimates.tex"],
    "knot_sensitivity.tex":
        ["simulation_outputs/tables/knot_sensitivity.tex"],
    "prior_sensitivity.tex":
        ["simulation_outputs/tables/prior_sensitivity.tex"],
    "misspecification.tex":
        ["simulation_outputs/tables/misspecification.tex"],
}

SIM_METRICS_CSV = "simulation_outputs/results_csv/all_scenarios_metrics_aggregated.csv"
SIM_BETA_TABLE = os.path.join(TABLES_DIR, "beta_abs_mae_Kpos_nocov.tex")


def _first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def copy_figures():
    os.makedirs(FIGS_TABLES_DIR, exist_ok=True)
    done, missing = [], []
    for dest, srcs in FIGURES.items():
        src = _first_existing(srcs)
        if src is None:
            missing.append(dest)
            continue
        shutil.copy(src, os.path.join(FIGS_TABLES_DIR, dest))
        done.append(dest)
    print(f"[figures] copied {len(done)}: {', '.join(done) if done else '(none)'}")
    if missing:
        print(f"[figures] MISSING (not regenerated, left as-is): {', '.join(missing)}")


def copy_tables():
    os.makedirs(TABLES_DIR, exist_ok=True)
    done, missing = [], []
    for dest, srcs in TABLE_COPIES.items():
        src = _first_existing(srcs)
        if src is None:
            missing.append(dest)
            continue
        shutil.copy(src, os.path.join(TABLES_DIR, dest))
        done.append(dest)
    print(f"[tables ] copied {len(done)}: {', '.join(done) if done else '(none)'}")
    if missing:
        print(f"[tables ] MISSING (left as-is): {', '.join(missing)}")


def _fmt(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "--"
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return "--"


def regenerate_sim_beta_table():
    """Rebuild tables/beta_abs_mae_Kpos_nocov.tex from the aggregated metrics CSV."""
    if not os.path.isfile(SIM_METRICS_CSV):
        print(f"[tables ] MISSING {SIM_METRICS_CSV}; sim beta-MAE table left as-is.")
        return
    df = pd.read_csv(SIM_METRICS_CSV).set_index("scenario_id")

    def cell(scen, kind, k, method):
        col = f"beta_abs_{kind}_{k}_mae_{method}_mean"
        if scen in df.index and col in df.columns:
            return _fmt(df.at[scen, col])
        return "--"

    k_map = {s["id"]: int(s.get("num_interventions_K_true", 0)) for s in config.SCENARIOS}
    rows = []
    for s in config.SCENARIOS:
        scen, K = s["id"], k_map[s["id"]]
        if K <= 0:
            continue
        s_l1, s_s1 = cell(scen, "step", 1, "sCFR"), cell(scen, "slope", 1, "sCFR")
        f_l1, f_s1 = cell(scen, "step", 1, "fsCFR"), cell(scen, "slope", 1, "fsCFR")
        if K >= 2:
            s_l2, s_s2 = cell(scen, "step", 2, "sCFR"), cell(scen, "slope", 2, "sCFR")
            f_l2, f_s2 = cell(scen, "step", 2, "fsCFR"), cell(scen, "slope", 2, "fsCFR")
        else:
            s_l2 = s_s2 = f_l2 = f_s2 = "--"
        rows.append(rf"\multirow{{2}}{{*}}{{{scen} ($K={K}$)}} & sCFR  & {s_l1} & {s_s1} & {s_l2} & {s_s2} \\")
        rows.append(rf" & fsCFR & {f_l1} & {f_s1} & {f_l2} & {f_s2} \\")
        rows.append(r"\addlinespace")
    if rows and rows[-1] == r"\addlinespace":
        rows = rows[:-1]

    tex = (
        r"\begin{table}[!t]" "\n"
        r"\centering" "\n"
        r"\caption{Summary of estimation performance for intervention magnitude parameters for the level indicators and slope hinge basis ($\beta_{\mathrm{abs}}$) across scenarios with interventions ($K>0$). Entries report Mean Absolute Error (MAE) for sCFR and the frequentist benchmark (fsCFR).}" "\n"
        r"\label{tab:beta_abs_mae_Kpos_nocov}" "\n"
        r"\resizebox{\textwidth}{!}{%" "\n"
        r"\begin{tabular}{llcccc}" "\n"
        r"\toprule" "\n"
        r"\multirow{2}{*}{Scenario} & \multirow{2}{*}{Method}" "\n"
        r"& \multicolumn{2}{c}{Intervention 1} & \multicolumn{2}{c}{Intervention 2} \\" "\n"
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}" "\n"
        r"& & Level MAE & Slope MAE & Level MAE & Slope MAE \\" "\n"
        r"\midrule" "\n"
        + "\n".join(rows) + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}%" "\n"
        r"}" "\n"
        r"\end{table}" "\n"
    )
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(SIM_BETA_TABLE, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print(f"[tables ] regenerated {SIM_BETA_TABLE} from {SIM_METRICS_CSV}")


def main():
    print("=" * 60)
    print("Refreshing docs/ figures and tables from latest results")
    print("=" * 60)
    copy_figures()
    regenerate_sim_beta_table()
    copy_tables()
    print("-" * 60)
    print("Done. Now rebuild the paper:")
    print("    cd docs && pdflatex main; bibtex main; pdflatex main; pdflatex main")


if __name__ == "__main__":
    main()
