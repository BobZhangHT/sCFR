# sCFR: Bayesian Semiparametric Framework for Time-Varying CFR Estimation

Code accompanying the paper *A Bayesian Semiparametric Framework for Factual and
Counterfactual Time-Varying Case Fatality Rate Estimation* (Zhang, Lee, Qu).

The **sCFR** model estimates a time-varying case fatality rate (CFR) from daily
case and death counts under reporting delay. It separates a smooth baseline trend
(the counterfactual CFR in the absence of interventions) from structured
non-pharmaceutical intervention (NPI) effects and high-frequency day-level noise,
and quantifies the mortality-risk reduction attributable to each intervention with
full Bayesian uncertainty. A frequentist counterpart (**fsCFR**) sharing the same
model and penalties, and the classical crude (**cCFR**) and Nishiura-adjusted
(**aCFR**) estimators, are provided as benchmarks.

## Model

Deaths are Poisson with a delay-convolution mean, and the logit-CFR is decomposed
additively:

```
d_t ~ Poisson(mu_t),      mu = Q r
logit(r_t) = B(t)'alpha + u_t + z(t)'beta,      t = 1, ..., T
```

| Component | Role |
|---|---|
| `B(t)'alpha` | Cubic B-spline baseline (the counterfactual trend), RW2 / P-spline smoothness penalty |
| `u_t` | Centered i.i.d. day-level random effect (overdispersion / high-frequency noise), `sum_t u_t = 0` |
| `z(t)'beta` | Interrupted time-series intervention term: signed level step + normalized slope hinge per NPI |
| `Q` | Lower-triangular operator convolving past cases with the onset-to-death delay |

The **counterfactual CFR** sets the intervention term to zero while keeping the
baseline and the day-level effect, `r_CF = sigmoid(B alpha + u)`, and the
**intervention effectiveness** is the proportional CFR reduction
`Eff_t = 1 - r_F,t / r_CF,t`.

**Priors / penalties** (`methods.py`, matching the paper):

- `alpha ~ N(0, 5^2 I)` with RW2 roughness penalty `(tau_alpha/2)||D alpha||^2`, `tau_alpha ~ Gamma(0.01, 0.01)`
- `u_t ~ N(0, sigma_u^2)` centered, `sigma_u ~ HalfCauchy(0.1)`
- `|beta| ~ LogNormal(log 0.5, 0.5)`, with signs fixed a priori from domain knowledge
- Onset-to-death delay: Gamma with mean 15.43 days, shape 2.03

Inference is Bayesian via NumPyro NUTS (1000 warm-up + 1000 sampling iterations by
default). fsCFR maximizes the same penalized likelihood (L-BFGS-B over the
coefficients, plus an EM/REML update for the variance component).

## Installation

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

Dependencies: JAX (CPU build is sufficient), NumPyro, NumPy, SciPy, pandas,
matplotlib, joblib, tqdm, numba. The Bayesian sCFR fit needs JAX + NumPyro; the
frequentist fsCFR needs only SciPy + numba. Analysis and plotting of cached results
run without JAX.

## Data

`WHO-COVID-19-global-daily-data.csv` (WHO COVID-19 global daily data) is included so
the UK real-data analysis is reproducible out of the box. UK national lockdown dates
are taken from the Institute for Government timeline; see the paper for provenance.

## Usage

### Simulation study

`simulation.py` is the one-stop driver. Run it from the repository root so that
`import config` resolves.

```bash
# Quick demo: 5 reps/scenario for the main grid, 10 for each auxiliary experiment
python simulation.py --simulate --demo

# Full study: 500 reps over all 12 scenarios + auxiliary experiments (server scale)
python simulation.py --simulate --full

# Rebuild figures/tables from existing outputs, without refitting
python simulation.py --analyze

# Clear all outputs and restart
python simulation.py --simulate --full --reset
```

`--simulate` runs, in order: the main 12-scenario Monte Carlo grid, the four
auxiliary experiments, `run_analysis()` to build figures and the intervention-MAE
table, and `refresh_docs.py` to sync `docs/`. Completed runs and cached auxiliary
replicates are skipped, so re-invoking `--simulate --full` resumes where it left off.
Use `--main-only` to skip the auxiliary experiments and `--no-refresh` to skip the
doc sync.

The auxiliary experiments can also be run in isolation:

```bash
python simulation.py --runtime   # timing / scalability (T up to 1200, K up to 8)
python simulation.py --knot      # knot-count sensitivity on scenario S09
python simulation.py --prior     # prior sensitivity on scenario S09
python simulation.py --misspec   # misspecified data-generating processes
```

Add `--demo` for a fast reduced-replicate pass, `--jobs N` to set the parallel job
count. For deterministic parallel timing, pin the numerical libraries to a single
thread (the script sets these defaults itself, but exporting them is safest):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python simulation.py --simulate --full
```

### UK real-data analysis

```bash
python real_data_analysis.py          # sCFR only
python real_data_analysis.py --full   # include cCFR, aCFR, and fsCFR benchmarks
```

`UK_Analysis.ipynb` reproduces the same analysis interactively.

### Syncing paper figures and tables

```bash
python refresh_docs.py   # copy result figures/tables into docs/ and regenerate the beta-MAE table
```

## Project structure

```
├── config.py               # Central configuration: scenarios, priors, MCMC, paths
├── data_generation.py      # Synthetic epidemic data, delay/Q matrix, spline/intervention bases
├── methods.py              # sCFR (NumPyro) model + cCFR / aCFR benchmarks
├── fsCFR_python.py         # Frequentist counterpart: penalized likelihood (L-BFGS-B + EM/REML)
├── evaluation.py           # Metrics (logit-MAE, coverage, effectiveness) and figures
├── simulation.py           # Monte Carlo runner + runtime/knot/prior/misspec experiments
├── real_data_analysis.py   # UK COVID-19 factual/counterfactual CFR analysis
├── table_gen.py            # Build the intervention-magnitude LaTeX table from result CSVs
├── refresh_docs.py         # Sync result figures/tables into docs/
├── UK_Analysis.ipynb       # Interactive UK analysis notebook
├── WHO-COVID-19-global-daily-data.csv   # WHO source data for the UK analysis
└── requirements.txt        # Python dependencies
```

## Outputs

- Simulation results land in `./simulation_outputs/` (`plots/`, `results_csv/`,
  `posterior_summaries/`, `benchmarks_results/`, `tables/`).
- Real-data results land in `./real_data_outputs/`.

Both directories are git-ignored. By default only posterior summaries (JSON) are
saved (`config.SAVE_RAW_POSTERIOR_SAMPLES = False`), which keeps disk use modest and
is sufficient for all plotting and analysis.

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@article{zhang_scfr,
  title   = {A Bayesian Semiparametric Framework for Factual and Counterfactual
             Time-Varying Case Fatality Rate Estimation},
  author  = {Zhang, Hengtao and Lee, Chun Yin and Qu, Yuanke},
  note    = {Under review at Infectious Disease Modelling},
  year    = {2025}
}
```

## License

Released under the MIT License; see [LICENSE](LICENSE).

## Contact

- Hengtao Zhang: zhanght@gdou.edu.cn
- Yuanke Qu (corresponding author): quxiaoke@gdou.edu.cn
