"""
Methods module for CFR estimation.

This module provides:
- Benchmark CFR methods: cCFR (crude), aCFR (adjusted)
- Frequentist sCFR (fsCFR) via penalized MLE
- Bayesian sCFR model using NumPyro

Model structure:
- Baseline: B-spline with RW2 penalty
- Random effect: i.i.d. Gaussian with HalfCauchy prior on scale
- Intervention: Step + Hinge functions with LogNormal priors
- Likelihood: Poisson
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy.special import logit, expit as sigmoid
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import config
import warnings


# =============================================================================
# Utility Functions
# =============================================================================

def construct_Q_matrix(c_t: np.ndarray, f_s: np.ndarray, T_sim: int) -> np.ndarray:
    """Construct convolution matrix Q from daily cases and delay distribution."""
    N = T_sim
    fs_padded = np.zeros(N)
    len_f_s = len(f_s)
    if len_f_s > 0:
        fs_padded[:min(len_f_s, N)] = f_s[:min(len_f_s, N)]
    
    f_matrix_conv = np.zeros((N, N))
    for i in range(N):
        relevant_fs = fs_padded[:i+1][::-1]
        f_matrix_conv[i, :i+1] = relevant_fs
        
    return f_matrix_conv @ np.diag(c_t)


# =============================================================================
# Benchmark CFR Methods
# =============================================================================

def cCFR_model(
    d_t: np.ndarray,
    c_t: np.ndarray,
    window: Optional[int] = None,
    cumulative: bool = False
) -> np.ndarray:
    """Calculate crude Case Fatality Rate (cCFR)."""
    if cumulative:
        d_t_agg, c_t_agg = np.cumsum(d_t), np.cumsum(c_t)
    elif window:
        d_t_agg = pd.Series(d_t).rolling(window=window, min_periods=1).sum().values
        c_t_agg = pd.Series(c_t).rolling(window=window, min_periods=1).sum().values
    else:
        d_t_agg, c_t_agg = d_t, c_t

    crude_cfr = np.zeros_like(d_t_agg, dtype=float)
    non_zero_cases = c_t_agg > 0
    crude_cfr[non_zero_cases] = d_t_agg[non_zero_cases] / c_t_agg[non_zero_cases]
    crude_cfr = np.clip(crude_cfr, 0, 1)
    return crude_cfr


def aCFR_model(
    d_t: np.ndarray,
    c_t: np.ndarray,
    f_s: np.ndarray
) -> np.ndarray:
    """Calculate Nishiura-style adjusted cumulative CFR (aCFR)."""
    T = len(c_t)
    d_t_cumulative = np.cumsum(d_t)
    
    fs_padded = np.zeros(T)
    fs_padded[:min(len(f_s), T)] = f_s[:min(len(f_s), T)]
    
    convolved_cases = np.convolve(c_t, fs_padded, mode='full')[:T]
    c_t_d_cumulative = np.cumsum(convolved_cases)

    nishiura_cfr = np.zeros_like(d_t_cumulative, dtype=float)
    non_zero_adj_cases = c_t_d_cumulative > 1e-9
    nishiura_cfr[non_zero_adj_cases] = d_t_cumulative[non_zero_adj_cases] / c_t_d_cumulative[non_zero_adj_cases]
    nishiura_cfr = np.clip(nishiura_cfr, 0, 1)
    return nishiura_cfr


# =============================================================================
# Frequentist sCFR (fsCFR) Model
# =============================================================================

from fsCFR_python import fsCFR_model_wrapper as fsCFR_model


def run_all_benchmarks(sim_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run all benchmark CFR estimation methods."""
    benchmark_r_t_estimates = {
        "cCFR_model": cCFR_model(sim_data["d_t"], sim_data["c_t"], cumulative=True),
        "aCFR_model": aCFR_model(sim_data["d_t"], sim_data["c_t"], sim_data["f_s_true"])
    }

    fscfr_results = fsCFR_model(
        d_t=sim_data["d_t"],
        c_t=sim_data["c_t"],
        f_s=sim_data["f_s_true"],
        Bm=sim_data["Bm_true"],
        intervention_times_abs=sim_data["true_intervention_times_0_abs"],
        intervention_signs=sim_data["beta_signs_true"],
        verbose=False
    )

    all_benchmark_results = {**benchmark_r_t_estimates, **fscfr_results}
    return all_benchmark_results


# =============================================================================
# Bayesian sCFR Model (NumPyro)
# =============================================================================

def sCFR_model(data: Dict[str, Any]):
    """
    Bayesian semiparametric CFR model.
    
    Structure:
    - Baseline: B @ alpha with RW2 penalty
    - Random effect: delta ~ N(0, sigma_delta), centered
    - Intervention: Step + Hinge functions
    - Likelihood: Poisson
    """
    dt = data['dt']
    fc_mat = data['fc_mat']
    Bm = data['Bm']
    Z = data['Z']
    beta_signs = data.get('beta_signs', None)
    T = dt.shape[0]
    J = Bm.shape[1]

    # Baseline effect with RW2 penalty
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 5.0).expand([J]).to_event(1))
    tau_alpha = numpyro.sample("tau_alpha", dist.Gamma(0.01, 0.01))

    if J >= 3:
        d2_alpha = alpha[2:] - 2.0 * alpha[1:-1] + alpha[:-2]
        numpyro.factor("rw2_alpha_penalty", -0.5 * tau_alpha * jnp.sum(d2_alpha ** 2))

    main_logit = jnp.dot(Bm, alpha)

    # Random effect with HalfCauchy prior for shrinkage
    sigma_delta = numpyro.sample("sigma_delta", dist.HalfCauchy(0.1))
    
    delta_raw = numpyro.sample("delta_raw", dist.Normal(0.0, 1.0).expand([T]).to_event(1))
    delta_uncentered = sigma_delta * delta_raw
    delta = delta_uncentered - jnp.mean(delta_uncentered)
    numpyro.deterministic("delta", delta)

    # Intervention effect
    intervention_logit = 0.0
    if Z.shape[1] > 0:
        if beta_signs is not None:
            # LogNormal priors centered around typical effect size (~0.5)
            beta_abs = numpyro.sample("beta_abs", 
                dist.LogNormal(jnp.log(0.5), 0.5).expand([Z.shape[1]]).to_event(1))
            beta_slope_abs = numpyro.sample("beta_slope_abs", 
                dist.LogNormal(jnp.log(0.5), 0.5).expand([Z.shape[1]]).to_event(1))
            beta_step = beta_abs * beta_signs
            beta_slope = beta_slope_abs * beta_signs
        else:
            beta_step = numpyro.sample(
                "beta", dist.Normal(0, 1.0).expand([Z.shape[1]]).to_event(1)
            )
            beta_slope = numpyro.sample(
                "beta_slope", dist.Normal(0, 1.0).expand([Z.shape[1]]).to_event(1)
            )

        denom = jnp.maximum(T - 1, 1)
        Z_hinge = (jnp.cumsum(Z, axis=0) - Z) / denom
        intervention_logit = jnp.dot(Z, beta_step) + jnp.dot(Z_hinge, beta_slope)

    # Likelihood
    eta = main_logit + delta + intervention_logit
    r_t_val = jax.nn.sigmoid(eta)
    
    mu = jnp.dot(fc_mat, r_t_val)
    mu = jnp.maximum(mu, 1e-9)

    numpyro.sample("obs_deaths", dist.Poisson(mu), obs=dt)

    # Deterministic outputs
    numpyro.deterministic("r_t", r_t_val)
    numpyro.deterministic("r_cf", jax.nn.sigmoid(main_logit + delta))
    numpyro.deterministic("baseline_logit", main_logit)


def run_numpyro_sampler(
    model_data: Dict[str, Any],
    rng_key: jax.random.PRNGKey,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 1,
    init_params: Optional[Dict[str, Any]] = None
) -> MCMC:
    """Run NUTS sampler for the sCFR model."""
    kernel = NUTS(
        sCFR_model,
        target_accept_prob=0.9,
        init_strategy=numpyro.infer.init_to_value(values=init_params) if init_params else numpyro.infer.init_to_median
    )

    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
        chain_method='parallel' if num_chains > 1 else 'sequential'
    )

    mcmc.run(rng_key, model_data)
    return mcmc


# =============================================================================
# Model Fitting Functions
# =============================================================================

def get_ols_initial_values(
    d_t: np.ndarray,
    Q_mat: np.ndarray,
    Z: np.ndarray,
    Bm: np.ndarray,
    beta_signs: Optional[np.ndarray] = None,
    c_t: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Initialize parameters using OLS for warm-starting the sampler."""
    T = len(d_t)
    J = Bm.shape[1]
    K = Z.shape[1]
    
    if c_t is not None:
        cCFR = cCFR_model(d_t, c_t, cumulative=True)
        cCFR = np.clip(cCFR, 1e-4, 1.0 - 1e-4)
        y = logit(cCFR)
    else:
        total_deaths = np.sum(d_t)
        total_effective_cases = np.sum(Q_mat) + 1e-9
        global_r = np.clip(total_deaths / total_effective_cases, 1e-4, 0.99)
        global_logit = logit(global_r)
        y = np.full(T, float(global_logit))

    B = np.asarray(Bm)
    b = B.T @ y
    BtB = B.T @ B + 1e-6 * np.eye(J)
    try:
        alpha_init = np.linalg.solve(BtB, b)
    except np.linalg.LinAlgError:
        alpha_init = np.linalg.lstsq(B, y, rcond=None)[0]

    init_values = {
        'alpha': alpha_init,
        'tau_alpha': 1.0,
        'sigma_delta': 0.1,
        'delta_raw': np.zeros(T)
    }

    if K > 0:
        if beta_signs is not None:
            init_values['beta_abs'] = np.full(K, 0.5, dtype=float)
            init_values['beta_slope_abs'] = np.full(K, 0.5, dtype=float)
        else:
            init_values['beta'] = np.zeros(K)
            init_values['beta_slope'] = np.zeros(K)
            
    return init_values


def _validate_init_params(init_params: Optional[Dict[str, Any]]) -> bool:
    """Validate initialization parameters."""
    if not init_params:
        return False
    for key in ("alpha", "delta_raw"):
        if key in init_params and not np.all(np.isfinite(init_params[key])):
            return False
    for key in ("tau_alpha", "sigma_delta"):
        if key in init_params:
            val = init_params[key]
            if not np.isfinite(val) or val <= 0:
                return False
    for key in ("beta_abs", "beta_slope_abs"):
        if key in init_params:
            vals = np.asarray(init_params[key])
            if vals.size > 0 and (np.any(~np.isfinite(vals)) or np.any(vals <= 0)):
                return False
    return True


def fit_proposed_model(
    sim_data: Dict[str, Any],
    jax_prng_key: jax.random.PRNGKey
) -> Tuple[Dict[str, np.ndarray], Any]:
    """Fit the Bayesian sCFR model and return posterior samples."""
    Z_input = sim_data["Z_input_true"]
    if sim_data["num_interventions_true_K"] == 0:
        Z_input = np.empty((sim_data["N_obs"], 0))
    
    beta_signs_jax = None
    if sim_data["num_interventions_true_K"] > 0 and "beta_signs_true" in sim_data:
        beta_signs_jax = jnp.array(sim_data["beta_signs_true"])

    data_for_sampler = {
        'dt': jnp.array(sim_data["d_t"]),
        'fc_mat': jnp.array(sim_data["Q_true"]),
        'Bm': jnp.array(sim_data["Bm_true"]),
        'Z': jnp.array(Z_input),
        'beta_signs': beta_signs_jax
    }

    try:
        init_vals = get_ols_initial_values(
            sim_data["d_t"],
            sim_data["Q_true"],
            Z_input,
            sim_data["Bm_true"],
            beta_signs=sim_data.get("beta_signs_true"),
            c_t=sim_data.get("c_t")
        )
    except Exception as e:
        print(f"Warning: OLS Initialization failed ({e}), falling back to default.")
        init_vals = None
    if not _validate_init_params(init_vals):
        if init_vals is not None:
            print("Warning: Invalid init values detected; falling back to default initialization.")
        init_vals = None

    mcmc = run_numpyro_sampler(
        model_data=data_for_sampler,
        rng_key=jax_prng_key,
        num_warmup=config.NUM_WARMUP,
        num_samples=config.NUM_SAMPLES,
        num_chains=config.NUM_CHAINS,
        init_params=init_vals
    )

    posterior_samples = mcmc.get_samples()
    posterior_samples_np = {k: np.array(v) for k, v in posterior_samples.items()}
    
    return posterior_samples_np, mcmc
