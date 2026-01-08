# Bayesian Semiparametric Framework for Time-Varying CFR Estimation

A Bayesian semiparametric model for estimating time-varying Case Fatality Rate (CFR) with intervention effects and counterfactual inference.

## Features

- **Bayesian sCFR Model**: Separates smooth baseline trends from intervention effects
- **Counterfactual Inference**: Estimates what would have happened without interventions
- **Benchmark Methods**: Includes cCFR, aCFR, and frequentist fsCFR for comparison
- **Simulation Framework**: Monte Carlo study with parallel execution and checkpointing

## Installation

```bash
pip install -r requirements.txt
```

Requirements: JAX, NumPyro, NumPy, SciPy, pandas, matplotlib, joblib, tqdm, numba

## Quick Start

### Simulation Study

```bash
# Demo mode (5 runs per scenario)
python simulation.py --simulate --demo

# Full simulation
python simulation.py --simulate --full

# Analysis only
python simulation.py --analyze

# Reset and restart
python simulation.py --simulate --full --reset
```

### UK Real Data Analysis

```bash
# sCFR only
python real_data_analysis.py

# Include all benchmarks
python real_data_analysis.py --full
```

## Project Structure

```
├── config.py              # Configuration parameters
├── data_generation.py     # Synthetic data generation
├── methods.py             # CFR estimation methods (sCFR, cCFR, aCFR)
├── fsCFR_python.py        # Frequentist sCFR implementation
├── evaluation.py          # Metrics and visualization
├── simulation.py          # Monte Carlo simulation runner
├── real_data_analysis.py  # UK COVID-19 analysis
└── requirements.txt       # Dependencies
```

## Model Structure

The sCFR model decomposes the logit-CFR as:

```
logit(r_t) = B(t)α + δ_t + β·Z(t) + γ·H(t)
```

- `B(t)α`: B-spline baseline with RW2 penalty
- `δ_t`: i.i.d. random effects ~ N(0, σ_δ)
- `β·Z(t)`: Step intervention effects
- `γ·H(t)`: Hinge (slope) intervention effects

Priors:
- `σ_δ ~ HalfCauchy(0.1)`
- `|β|, |γ| ~ LogNormal(log(0.5), 0.5)`
- Likelihood: Poisson with delay convolution

## Output

Simulation outputs are saved to `./simulation_outputs/`:
- `plots/`: Visualization figures
- `results_csv/`: Aggregated metrics
- `posterior_samples_raw/`: MCMC samples
- `benchmarks_results/`: Benchmark estimates

Real data outputs are saved to `./real_data_outputs/`.


## Contact

- Hengtao Zhang: zhanght@gdou.edu.cn
- Yuanke Qu: quxiaoke@gdou.edu.cn
