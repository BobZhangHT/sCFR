# A Bayesian Semiparametric Framework for Factual and Counterfactual Time-Varying Case Fatality Rate Estimation

This repository contains the complete source code and simulation framework for the manuscript, "A Bayesian Semiparametric Framework for Factual and Counterfactual Time-Varying Case Fatality Rate Estimation." The paper introduces a robust Bayesian semiparametric model to estimate the time-varying Case Fatality Rate (CFR) while accounting for reporting delays and the confounding effects of non-pharmaceutical interventions (NPIs).

## Key Features

* **Bayesian Semiparametric Model**: A flexible model that separates smooth baseline trends from sharp intervention shocks.

* **Causal Inference**: Enables the estimation of counterfactual outcomes (what would have happened without NPIs).

* **Theoretical Guarantees**: The estimator is supported by a posterior contraction rate theorem, ensuring statistical consistency.

* **Simulation & Analysis Pipeline**: Includes a complete, parallelized pipeline for running Monte Carlo simulations and analyzing results.

* **Real Data Application**: Code to replicate the analysis of COVID-19 data in the UK.

## Getting Started

### Prerequisites

* Python 3.8+
* JAX
* NumPyro
* NumPy
* SciPy
* pandas
* matplotlib
* joblib
* tqdm

### Running the Simulation Study

The entire simulation study can be executed using the command-line script `simulation.py`. This script provides both simulation and analysis modes with automatic checkpoint support.

#### Quick Start

```bash
# Run demo mode (5 runs per scenario) with automatic analysis
python simulation.py --simulate --demo

# Run full mode (all configured runs) with automatic analysis
python simulation.py --simulate --full

# Run analysis only on existing results
python simulation.py --analyze

# Reset all outputs and start fresh
python simulation.py --simulate --full --reset
```

#### Command-Line Options

- `--simulate`: Run simulation mode (generates data and fits models)
- `--analyze`: Run analysis mode only (generates plots and tables from existing results)
- `--demo`: Demo mode with 5 runs per scenario (for quick testing)
- `--full`: Full mode with all configured runs (default)
- `--reset`: Clear all output directories before starting (requires confirmation)
- `--jobs N`: Number of parallel jobs (default: `NUM_CORES_TO_USE` from config)

#### Automatic Analysis

When using `--simulate --full` or `--simulate --demo`, the script will **automatically** run analysis after simulation completes, generating:

**Plots** (in `simulation_outputs/plots/`):
- `aggregated_factual_summary.png` - Factual CFR estimates across all scenarios
- `aggregated_counterfactual_summary.png` - Counterfactual CFR estimates
- `metric_summary_boxplots_mae.png` - MAE boxplots by model
- `combined_metrics_summary.png` - Combined metrics summary

**Tables** (in `simulation_outputs/results_csv/`):
- `all_scenarios_metrics_aggregated.csv` - Aggregated metrics (mean and std)

**LaTeX Tables** (in `simulation_outputs/tables/`):
- Formatted tables for manuscript inclusion

#### Checkpoint Support

The script supports automatic checkpointing:
- If a run is interrupted, you can resume it later
- The script will skip completed runs (based on existing metrics JSON files)
- Continue from where it left off

### Analyzing the Results

After the simulation is complete, you can use the `--analyze` flag to generate plots and tables:

```bash
python simulation.py --analyze
```

### UK Real Data Application

You can run the UK analysis from the command line:

```bash
python real_data_analysis.py
```

Outputs are written to `./real_data_outputs/`. Set `PLOT_FULL = True` inside `real_data_analysis.py` if you want to include benchmark curves (cCFR, aCFR, fsCFR).

## Project Structure

```
BICE-CFR/
- config.py                           # Configuration parameters
- data_generation.py                  # Data generation functions
- methods.py                          # Model fitting and benchmark methods (NO CI for fsCFR, aCFR, cCFR)
- evaluation.py                       # Unified evaluation and visualization module
- simulation.py                       # Main simulation and analysis script
- real_data_analysis.py               # UK real data analysis (CLI)
- UK_Analysis.ipynb                   # UK real data analysis (notebook)
- WHO-COVID-19-global-daily-data.csv  # WHO daily data (place in repo root)
- README.md                           # This file
```

## Key Modules

### evaluation.py

Unified evaluation and visualization module that combines metric calculation with plotting functionality. Provides:

- **Data Classes**: `PosteriorEstimates`, `ModelEvaluationResult`, `ScenarioEvaluationResult`, `AggregatedEvaluationResult`
- **Main Class**: `CFREvaluatorVisualizer` with comprehensive methods for evaluation and visualization
- **Convenience Functions**: For backward compatibility with original modules

### methods.py

Provides unified methods for benchmark calculations, model fitting, and statistical analysis:

- **Benchmark CFR Calculations**: `cCFR_model()`, `aCFR_model()` (NO confidence intervals)
- **fsCFR Model**: `fsCFR_model()` (NO confidence intervals)
- **sCFR Model**: `sCFR_model()`, `run_numpyro_sampler()`, `fit_proposed_model()` (with Bayesian CI)
- **Note**: Only sCFR retains confidence interval estimation; fsCFR, aCFR, and cCFR provide point estimates only

### simulation.py

Main script for running simulations and analysis:

- **Simulation Mode**: Runs Monte Carlo simulations with checkpoint support
- **Analysis Mode**: Generates plots and tables from existing results
- **Automatic Analysis**: Automatically runs analysis after simulation completes
- **Parallel Processing**: Uses joblib for parallel execution

## Output Structure

After running simulations and analysis, `simulation_outputs/` directory will contain:

```
simulation_outputs/
- plots/                                # Generated visualizations
  - aggregated_factual_summary.png
  - aggregated_counterfactual_summary.png
  - metric_summary_boxplots_mae.png
  - combined_metrics_summary.png
- tables/                               # LaTeX tables for manuscript
- results_csv/                          # CSV files with metrics
  - all_scenarios_metrics_aggregated.csv
- posterior_samples_raw/                # Saved posterior samples
- benchmarks_results/                   # Benchmark method results
- posterior_summaries/                  # Posterior summary statistics
- run_metrics_json/                     # Individual run metrics
- logs/                                 # Error logs and analysis summary
```

## Contact

For any questions, comments, or suggestions, please feel free to contact the first author or corresponding author:

* Hengtao Zhang: zhanght@gdou.edu.cn

* Yuanke Qu: quxiaoke@gdou.edu.cn
