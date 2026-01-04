"""
Methods module for sCFR project.

This module provides unified methods for benchmark calculations, model fitting,
and statistical analysis in Case Fatality Rate (CFR) estimation framework.

MAJOR UPDATES:
1. Merged 'sampler.py' functionality into 'sCFR_model' and 'run_numpyro_sampler'.
2. Refactored Intervention Effect to use Step + Hinge functions.
3. Added fsCFR_model as a frequentist counterpart with REML-tuned penalties.
4. No confidence intervals for benchmark methods (only sCFR retains CI).
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

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def construct_Q_matrix(
    c_t: np.ndarray,
    f_s: np.ndarray,
    T_sim: int
) -> np.ndarray:
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
# BENCHMARK CFR CALCULATIONS
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
# FREQUENTIST sCFR (fsCFR) MODEL - PENALIZED MLE WITH REML
# =============================================================================

def fsCFR_model(
    d_t: np.ndarray,
    c_t: np.ndarray,
    f_s: np.ndarray,
    Bm: np.ndarray,
    intervention_times_abs: List[int],
    intervention_signs: List[int]
) -> Dict[str, np.ndarray]:
    """
    Frequentist semiparametric counterpart to sCFR with REML-tuned penalties.

    Model:
      eta = Bm @ alpha + delta + Z @ beta_step + Z_hinge @ beta_slope
      r_t = sigmoid(eta), d_t ~ Poisson(Q r_t)

    Penalties:
      p_alpha * ||D2 alpha||^2 + p_delta * ||delta||^2 + p_beta * (||beta_step||^2 + ||beta_slope||^2)
    """
    T, J = Bm.shape
    K = len(intervention_times_abs)
    intervention_signs = np.asarray(intervention_signs, dtype=float)

    Q_matrix = construct_Q_matrix(c_t, f_s, T)

    # Step basis Z and hinge basis
    Z_step = np.zeros((T, K))
    t_array = np.arange(T, dtype=float) / max(T - 1, 1)
    if K > 0:
        for k in range(K):
            t_k = intervention_times_abs[k] / max(T - 1, 1)
            Z_step[:, k] = (t_array >= t_k).astype(float)
    denom = max(T - 1, 1)
    Z_hinge = (np.cumsum(Z_step, axis=0) - Z_step) / denom

    # Second difference matrix for alpha
    def second_diff_matrix(k: int) -> np.ndarray:
        if k < 3:
            return np.zeros((0, k))
        d2 = np.zeros((k - 2, k))
        for i in range(k - 2):
            d2[i, i:i+3] = [1, -2, 1]
        return d2

    D2 = second_diff_matrix(J)
    num_params = J + T + (2 * K)

    def objective_func(
        params: np.ndarray,
        p_alpha: float,
        p_delta: float,
        p_beta: float
    ) -> float:
        alpha = params[:J]
        delta = params[J:J + T]
        beta_step_abs = params[J + T:J + T + K] if K > 0 else np.array([])
        beta_slope_abs = params[J + T + K:] if K > 0 else np.array([])

        eta = Bm @ alpha + delta
        if K > 0:
            beta_step = beta_step_abs * intervention_signs
            beta_slope = beta_slope_abs * intervention_signs
            eta = eta + Z_step @ beta_step + Z_hinge @ beta_slope
        if not np.all(np.isfinite(eta)):
            return 1e12

        r_t = sigmoid(eta)
        if not np.all(np.isfinite(r_t)):
            return 1e12
        mu = Q_matrix @ r_t
        if not np.all(np.isfinite(mu)):
            return 1e12
        mu = np.maximum(mu, 1e-9)
        nll = -np.sum(d_t * np.log(mu) - mu)
        if not np.isfinite(nll):
            return 1e12

        penalty = 0.0
        if D2.shape[0] > 0:
            diff2 = D2 @ alpha
            penalty += p_alpha * np.sum(diff2 ** 2)
        penalty += p_delta * np.sum(delta ** 2)
        if K > 0:
            penalty += p_beta * (np.sum(beta_step_abs ** 2) + np.sum(beta_slope_abs ** 2))
        total = nll + penalty
        if not np.isfinite(total):
            return 1e12
        return float(total)

    def build_penalty_matrix(p_alpha: float, p_delta: float, p_beta: float) -> np.ndarray:
        S = np.zeros((num_params, num_params))
        if D2.shape[0] > 0:
            S_alpha = p_alpha * (D2.T @ D2)
            S[:J, :J] = S_alpha
        S[J:J + T, J:J + T] = p_delta * np.eye(T)
        if K > 0:
            S[J + T:, J + T:] = p_beta * np.eye(2 * K)
        return S

    def jacobian_eta(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        alpha = params[:J]
        delta = params[J:J + T]
        beta_step_abs = params[J + T:J + T + K] if K > 0 else np.array([])
        beta_slope_abs = params[J + T + K:] if K > 0 else np.array([])

        eta = Bm @ alpha + delta
        d_eta = np.zeros((T, num_params))
        d_eta[:, :J] = Bm
        d_eta[:, J:J + T] = np.eye(T)
        if K > 0:
            beta_step = beta_step_abs * intervention_signs
            beta_slope = beta_slope_abs * intervention_signs
            eta = eta + Z_step @ beta_step + Z_hinge @ beta_slope
            d_eta[:, J + T:J + T + K] = Z_step * intervention_signs
            d_eta[:, J + T + K:] = Z_hinge * intervention_signs
        return eta, d_eta

    def jacobian_mu(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        eta, d_eta = jacobian_eta(params)
        r_t = sigmoid(eta)
        dr_deta = r_t * (1 - r_t)
        dr_dtheta = dr_deta[:, None] * d_eta
        J_mu = Q_matrix @ dr_dtheta
        return r_t, J_mu

    def reml_criterion(log_penalties: np.ndarray) -> float:
        p_alpha = float(np.exp(log_penalties[0]))
        p_delta = float(np.exp(log_penalties[1]))
        p_beta = float(np.exp(log_penalties[2])) if K > 0 else 0.0

        res = minimize(
            objective_func, x0=p0,
            args=(p_alpha, p_delta, p_beta),
            method='L-BFGS-B', bounds=bounds
        )
        if not res.success:
            return 1e12

        params_hat = res.x
        eta_hat, _ = jacobian_eta(params_hat)
        r_t_hat = sigmoid(eta_hat)
        mu_hat = np.maximum(Q_matrix @ r_t_hat, 1e-9)
        log_like = np.sum(d_t * np.log(mu_hat) - mu_hat)

        penalty_term = 0.0
        if D2.shape[0] > 0:
            diff2 = D2 @ params_hat[:J]
            penalty_term += p_alpha * np.sum(diff2 ** 2)
        delta_hat = params_hat[J:J + T]
        penalty_term += p_delta * np.sum(delta_hat ** 2)
        if K > 0:
            beta_step_abs = params_hat[J + T:J + T + K]
            beta_slope_abs = params_hat[J + T + K:]
            penalty_term += p_beta * (np.sum(beta_step_abs ** 2) + np.sum(beta_slope_abs ** 2))

        _, J_mu = jacobian_mu(params_hat)
        weight = 1.0 / np.maximum(mu_hat, 1e-9)
        H = J_mu.T @ (J_mu * weight[:, None])

        S = build_penalty_matrix(p_alpha, p_delta, p_beta)
        S_plus_H = H + S

        def safe_logdet_pd(mat: np.ndarray) -> Optional[float]:
            mat_sym = 0.5 * (mat + mat.T)
            for jitter in (1e-10, 1e-8, 1e-6, 1e-4):
                try:
                    L = np.linalg.cholesky(mat_sym + jitter * np.eye(mat_sym.shape[0]))
                    return float(2.0 * np.sum(np.log(np.diag(L))))
                except np.linalg.LinAlgError:
                    continue
            return None

        logdet_hs = safe_logdet_pd(S_plus_H)
        if logdet_hs is None or not np.isfinite(logdet_hs):
            return 1e12

        eigvals_S = None
        S_sym = 0.5 * (S + S.T)
        for jitter in (0.0, 1e-10, 1e-8, 1e-6):
            try:
                eigvals_S = np.linalg.eigvalsh(S_sym + jitter * np.eye(num_params))
                break
            except np.linalg.LinAlgError:
                continue
        if eigvals_S is None or not np.all(np.isfinite(eigvals_S)):
            return 1e12

        positive_eigs = eigvals_S[eigvals_S > 1e-12]
        logdet_s = np.sum(np.log(positive_eigs)) if positive_eigs.size > 0 else 0.0
        null_dim = np.sum(eigvals_S <= 1e-12)

        lr = (
            log_like
            - 0.5 * penalty_term
            + 0.5 * logdet_s
            - 0.5 * logdet_hs
            - 0.5 * null_dim * np.log(2 * np.pi)
        )
        if not np.isfinite(lr):
            return 1e12
        return -lr

    p0 = np.zeros(num_params)
    if K > 0:
        p0[J + T:J + T + K] = 0.05
        p0[J + T + K:] = 0.05

    bounds = [(None, None)] * (J + T)
    if K > 0:
        bounds += [(0, None)] * (2 * K)

    log_p0 = np.log([1.0, 1.0, 1.0]) if K > 0 else np.log([1.0, 1.0])
    opt_penalties = minimize(
        reml_criterion, x0=log_p0,
        method='L-BFGS-B', bounds=[(-10, 10)] * len(log_p0)
    )
    if not opt_penalties.success:
        best_p_alpha = float(np.exp(log_p0[0]))
        best_p_delta = float(np.exp(log_p0[1]))
        best_p_beta = float(np.exp(log_p0[2])) if K > 0 else 0.0
    else:
        best_p_alpha = float(np.exp(opt_penalties.x[0]))
        best_p_delta = float(np.exp(opt_penalties.x[1]))
        best_p_beta = float(np.exp(opt_penalties.x[2])) if K > 0 else 0.0

    final_res = minimize(
        objective_func, x0=p0,
        args=(best_p_alpha, best_p_delta, best_p_beta),
        method='L-BFGS-B', bounds=bounds
    )
    popt = final_res.x

    alpha_hat = popt[:J]
    delta_hat = popt[J:J + T]
    eta_hat, _ = jacobian_eta(popt)
    r_t_factual = sigmoid(eta_hat)
    eta_cf = Bm @ alpha_hat + delta_hat
    r_t_counterfactual = sigmoid(eta_cf)

    res = {
        "fsCFR_factual_mean": r_t_factual,
        "fsCFR_counterfactual_mean": r_t_counterfactual,
        "fsCFR_baseline_logit": Bm @ alpha_hat,
        "fsCFR_delta": delta_hat,
    }
    if K > 0:
        res["fsCFR_beta_abs_est"] = popt[J + T:J + T + K]
        res["fsCFR_beta_slope_abs_est"] = popt[J + T + K:]
    else:
        res["fsCFR_beta_abs_est"] = np.array([])
        res["fsCFR_beta_slope_abs_est"] = np.array([])
    return res


def run_all_benchmarks(sim_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run all benchmark CFR estimation methods on simulated data (no CIs)."""
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
        intervention_signs=sim_data["beta_signs_true"]
    )

    all_benchmark_results = {**benchmark_r_t_estimates, **fscfr_results}
    return all_benchmark_results


# =============================================================================
# NUMPYRO MODEL DEFINITION (Merged from sampler.py)
# =============================================================================

def sCFR_model(data: Dict[str, Any]):
    """
    Optimized sCFR Model with spline main effect, i.i.d. random effect,
    and hinge + step-function interventions.
    Structure:
      - Baseline: RW2 with L2 penalty (Gaussian innovations on 2nd differences).
      - Intervention: Linear combination of indicator functions.
      - Likelihood: Poisson.
    """
    # --- 1. Unpack Data ---
    dt = data['dt']            # Observed deaths
    fc_mat = data['fc_mat']    # Convolution matrix Q
    Bm = data['Bm']            # B-spline basis (T x J)
    Z = data['Z']              # Intervention Matrix (T x K)
    beta_signs = data.get('beta_signs', None) # Optional direction constraints
    T = dt.shape[0]
    J = Bm.shape[1]

    # --- 2. Main Effect: B @ alpha with RW2-L2 prior on alpha ---
    # Overall shrinkage on alpha
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 5.0).expand([J]).to_event(1))
    tau_alpha = numpyro.sample("tau_alpha", dist.Gamma(0.01, 0.01))

    if J >= 3:
        d2_alpha = alpha[2:] - 2.0 * alpha[1:-1] + alpha[:-2]
        numpyro.factor("rw2_alpha_penalty", -0.5 * tau_alpha * jnp.sum(d2_alpha ** 2))

    main_logit = jnp.dot(Bm, alpha)

    # --- 3. Random Effect: i.i.d. with centering constraint ---
    sigma_delta = numpyro.sample("sigma_delta", dist.HalfNormal(1.0))
    delta_raw = numpyro.sample("delta_raw", dist.Normal(0.0, sigma_delta).expand([T]).to_event(1))
    delta = delta_raw - jnp.mean(delta_raw)
    numpyro.deterministic("delta", delta)

    # --- 4. Intervention Effect: Hinge + Step Functions ---
    intervention_logit = 0.0
    if Z.shape[1] > 0:
        if beta_signs is not None:
            beta_abs = numpyro.sample("beta_abs", dist.HalfNormal(1.0).expand([Z.shape[1]]).to_event(1))
            beta_slope_abs = numpyro.sample(
                "beta_slope_abs", dist.HalfNormal(1.0).expand([Z.shape[1]]).to_event(1)
            )
            beta_step = beta_abs * beta_signs
            beta_slope = beta_slope_abs * beta_signs
        else:
            beta_step = numpyro.sample(
                "beta", dist.Normal(0, 1.0).expand([Z.shape[1]]).to_event(1)
            )
            beta_slope = numpyro.sample(
                "beta_slope", dist.Normal(0, 1.0).expand([Z.shape[1]]).to_event(1)
            )

        # Hinge basis: (t - t_k)_+ derived from step Z; scaled to [0, 1]
        denom = jnp.maximum(T - 1, 1)
        Z_hinge = (jnp.cumsum(Z, axis=0) - Z) / denom
        intervention_logit = jnp.dot(Z, beta_step) + jnp.dot(Z_hinge, beta_slope)

    # --- 5. Likelihood (Poisson) ---
    eta = main_logit + delta + intervention_logit
    r_t_val = jax.nn.sigmoid(eta)
    
    # Convolution: Expected deaths = Q * r_t
    mu = jnp.dot(fc_mat, r_t_val)
    mu = jnp.maximum(mu, 1e-9) # Numerical stability

    # Poisson Likelihood as requested
    numpyro.sample("obs_deaths", dist.Poisson(mu), obs=dt)

    # --- 5. Deterministic Outputs ---
    # Essential for analysis and plotting
    numpyro.deterministic("r_t", r_t_val)
    # Counterfactual: Baseline only (excludes intervention_logit)
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
    """
    Runs the NUTS sampler for the sCFR model.
    """
    # Kernel definition
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
        progress_bar=True, # Enable progress bar
        chain_method='parallel' if num_chains > 1 else 'sequential'
    )

    # Run MCMC
    mcmc.run(rng_key, model_data)
    return mcmc

# =============================================================================
# MODEL FITTING FUNCTIONS
# =============================================================================

def get_ols_initial_values(
    d_t: np.ndarray,
    Q_mat: np.ndarray,
    Z: np.ndarray,
    Bm: np.ndarray,
    beta_signs: Optional[np.ndarray] = None,
    c_t: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Initializes parameters to warm-start the sampler.
    Baseline alpha is obtained by OLS: argmin ||B alpha - logit(cCFR)||^2.
    Other parameters use simple small-value initializations aligned with sCFR_model.
    """
    if c_t is not None:
        cCFR = cCFR_model(d_t, c_t, cumulative=True)
        cCFR = np.clip(cCFR, 1e-4, 1.0 - 1e-4)
        y = logit(cCFR)
    else:
        total_deaths = np.sum(d_t)
        total_effective_cases = np.sum(Q_mat) + 1e-9
        global_r = np.clip(total_deaths / total_effective_cases, 1e-4, 0.99)
        global_logit = logit(global_r)
        y = np.full(len(d_t), float(global_logit))

    B = np.asarray(Bm)
    b = B.T @ y
    BtB = B.T @ B + 1e-6 * np.eye(B.shape[1])
    try:
        alpha_init = np.linalg.solve(BtB, b)
    except np.linalg.LinAlgError:
        alpha_init = np.linalg.lstsq(B, y, rcond=None)[0]

    init_values = {
        'alpha': alpha_init,
        'tau_alpha': 1.0,
        'sigma_delta': 0.1,
        'delta_raw': np.zeros(len(d_t))
    }

    # 2. If interventions exist, initialize coefficients
    if Z.shape[1] > 0:
        if beta_signs is not None:
            init_values['beta_abs'] = np.full(Z.shape[1], 0.05, dtype=float)
            init_values['beta_slope_abs'] = np.full(Z.shape[1], 0.05, dtype=float)
        else:
            init_values['beta'] = np.zeros(Z.shape[1])
            init_values['beta_slope'] = np.zeros(Z.shape[1])
            
    return init_values


def _validate_init_params(init_params: Optional[Dict[str, Any]]) -> bool:
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
    """
    Main entry point: Prepares data, initializes params, and fits the model.
    Returns posterior samples and the MCMC object.
    """
    # 1. Prepare Data Dictionary for JAX
    Z_input = sim_data["Z_input_true"]
    if sim_data["num_interventions_true_K"] == 0:
        Z_input = np.empty((sim_data["N_obs"], 0))
    
    # Handle beta signs if present
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

    # 2. Get Initial Values (Warm Start)
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

    # 3. Run Sampler
    mcmc = run_numpyro_sampler(
        model_data=data_for_sampler,
        rng_key=jax_prng_key,
        num_warmup=config.NUM_WARMUP,
        num_samples=config.NUM_SAMPLES,
        num_chains=config.NUM_CHAINS,
        init_params=init_vals
    )

    # 4. Extract Samples
    posterior_samples = mcmc.get_samples()
    
    # Convert to numpy for downstream compatibility
    posterior_samples_np = {k: np.array(v) for k, v in posterior_samples.items()}
    
    return posterior_samples_np, mcmc
