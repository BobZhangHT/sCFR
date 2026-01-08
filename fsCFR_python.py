"""
Frequentist sCFR (fsCFR) implementation using EM algorithm.

Model structure (matching Bayesian sCFR):
- sigma_delta ~ HalfCauchy(0.1)
- tau_alpha ~ Gamma(0.01, 0.01)
- alpha ~ Normal(0, 5)
- delta ~ Normal(0, sigma_delta), centered
- beta_abs, beta_slope_abs ~ LogNormal(log(0.5), 0.5)
- Likelihood: Poisson

Uses EM algorithm for sigma_delta estimation.
"""

import numpy as np
from scipy import optimize
from scipy.special import expit as sigmoid
from numba import jit
import time
from typing import Dict, Any, Optional, List
from tqdm import tqdm


# =============================================================================
# Numba-accelerated Core Functions
# =============================================================================

@jit(nopython=True, cache=True)
def _compute_eta_and_r(alpha, delta, int_eff, B):
    """Compute eta = B @ alpha + delta + int_eff, then r = sigmoid(eta)."""
    T = B.shape[0]
    eta = np.zeros(T)
    for i in range(T):
        eta[i] = delta[i] + int_eff[i]
        for j in range(B.shape[1]):
            eta[i] += B[i, j] * alpha[j]
    
    r = np.zeros(T)
    for i in range(T):
        if eta[i] > 10.0:
            r[i] = 1.0 - 1e-7
        elif eta[i] < -10.0:
            r[i] = 1e-7
        else:
            r[i] = 1.0 / (1.0 + np.exp(-eta[i]))
    return eta, r


@jit(nopython=True, cache=True)
def _compute_mu(Q, r):
    """Compute mu = Q @ r (convolution)."""
    T = Q.shape[0]
    mu = np.zeros(T)
    for i in range(T):
        for j in range(i + 1):
            mu[i] += Q[i, j] * r[j]
        if mu[i] < 1e-8:
            mu[i] = 1e-8
    return mu


@jit(nopython=True, cache=True)
def _poisson_log_lik(d_t, mu):
    """Poisson log-likelihood: sum(d*log(mu) - mu)."""
    ll = 0.0
    for i in range(len(d_t)):
        if mu[i] > 0:
            ll += d_t[i] * np.log(mu[i]) - mu[i]
    return ll


@jit(nopython=True, cache=True)
def _compute_intervention_effect(t_k, beta_step, beta_slope, T):
    """Compute intervention effect: step + hinge functions."""
    int_eff = np.zeros(T)
    K = len(t_k)
    if K == 0:
        return int_eff
    
    denom = max(T - 1, 1)
    for k in range(K):
        t_start = t_k[k]
        cumsum_val = 0.0
        for t in range(T):
            step = 1.0 if t >= t_start else 0.0
            cumsum_val += step
            hinge = (cumsum_val - step) / denom
            int_eff[t] += beta_step[k] * step + beta_slope[k] * hinge
    return int_eff


@jit(nopython=True, cache=True)
def _center_delta(delta_raw):
    """Center delta: delta = delta_raw - mean(delta_raw)."""
    return delta_raw - np.mean(delta_raw)


@jit(nopython=True, cache=True)
def _compute_nll_fixed_sigma(alpha, delta_raw, gamma, sigma_delta, tau_alpha,
                              d_t, Q, B, t_k, beta_sign, K, J, T):
    """Compute negative log-likelihood with fixed sigma_delta."""
    delta = _center_delta(delta_raw)
    
    if K > 0:
        beta_abs = np.zeros(2*K)
        for k in range(2*K):
            g = min(max(gamma[k], -5.0), 5.0)
            beta_abs[k] = np.exp(g)
        beta_step = np.zeros(K)
        beta_slope = np.zeros(K)
        for k in range(K):
            beta_step[k] = beta_sign[k] * beta_abs[k]
            beta_slope[k] = beta_sign[K + k] * beta_abs[K + k]
    else:
        beta_step = np.zeros(0)
        beta_slope = np.zeros(0)
        beta_abs = np.zeros(0)
    
    int_eff = _compute_intervention_effect(t_k, beta_step, beta_slope, T)
    eta, r = _compute_eta_and_r(alpha, delta, int_eff, B)
    mu = _compute_mu(Q, r)
    
    nll = -_poisson_log_lik(d_t, mu)
    
    # Priors
    # sigma_delta ~ HalfCauchy(0.1)
    nll += np.log(1.0 + (sigma_delta / 0.1) ** 2)
    
    # tau_alpha ~ Gamma(0.01, 0.01)
    nll += 0.99 * np.log(tau_alpha + 1e-10) + 0.01 * tau_alpha
    
    # alpha ~ Normal(0, 5)
    nll += 0.5 * np.sum(alpha * alpha) / 25.0
    
    # RW2 penalty
    if J >= 3:
        rw2_sum = 0.0
        for j in range(2, J):
            d2 = alpha[j] - 2.0 * alpha[j-1] + alpha[j-2]
            rw2_sum += d2 * d2
        nll += 0.5 * tau_alpha * rw2_sum
    
    # delta prior
    sigma2 = sigma_delta * sigma_delta + 1e-10
    nll += 0.5 * np.sum(delta_raw * delta_raw) / sigma2
    
    # beta prior: LogNormal(log(0.5), 0.5)
    if K > 0:
        log_half = -0.6931471805599453
        for k in range(2*K):
            log_beta = np.log(beta_abs[k] + 1e-10)
            nll += log_beta + 2.0 * (log_beta - log_half) ** 2
    
    return nll


def _fit_given_sigma(sigma_delta, d_t, Q, B, t_k, beta_sign, K, J, T, 
                     alpha_init, max_iter=100):
    """Fit model with fixed sigma_delta."""
    n_params = J + T + 2*K + 1
    params_init = np.zeros(n_params)
    params_init[:J] = alpha_init
    if K > 0:
        params_init[J+T:J+T+2*K] = np.log(0.5)
    
    def objective(params):
        alpha = params[:J]
        delta_raw = params[J:J+T]
        gamma = params[J+T:J+T+2*K] if K > 0 else np.array([])
        tau_alpha = np.exp(params[J+T+2*K])
        return _compute_nll_fixed_sigma(alpha, delta_raw, gamma, sigma_delta, tau_alpha,
                                        d_t, Q, B, t_k, beta_sign, K, J, T)
    
    result = optimize.minimize(objective, params_init, method='L-BFGS-B',
                               options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-5})
    return result.fun, result.x


def fsCFR_model(d_t: np.ndarray, c_t: np.ndarray, f_s: np.ndarray, 
                B: np.ndarray, t_k: Optional[np.ndarray] = None,
                beta_sign: Optional[np.ndarray] = None,
                max_iter: int = 200, em_iter: int = 10,
                verbose: bool = True) -> Dict[str, Any]:
    """
    Fit fsCFR model using EM algorithm for sigma_delta estimation.
    
    Parameters
    ----------
    d_t : array (T,) - observed deaths
    c_t : array (T,) - observed cases  
    f_s : array (L,) - delay distribution
    B : array (T, J) - B-spline basis matrix
    t_k : array (K,) - intervention time points (0-indexed)
    beta_sign : array (2K,) - sign constraints [step_signs, slope_signs]
    max_iter : int - max iterations per optimization
    em_iter : int - number of EM iterations
    verbose : bool - show progress bar
    
    Returns
    -------
    dict with fitted parameters and predictions
    """
    start_time = time.time()
    
    T = len(d_t)
    J = B.shape[1]
    L = len(f_s)
    
    d_t = np.ascontiguousarray(d_t, dtype=np.float64)
    c_t = np.ascontiguousarray(c_t, dtype=np.float64)
    f_s = np.ascontiguousarray(f_s, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    
    if t_k is None or len(t_k) == 0:
        t_k = np.array([], dtype=np.int64)
        beta_sign = np.array([], dtype=np.float64)
        K = 0
    else:
        t_k = np.ascontiguousarray(t_k, dtype=np.int64)
        K = len(t_k)
        if beta_sign is None:
            beta_sign = np.concatenate([np.full(K, -1.0), np.full(K, -1.0)])
        beta_sign = np.ascontiguousarray(beta_sign, dtype=np.float64)
    
    # Construct Q matrix
    Q = np.zeros((T, T), dtype=np.float64)
    for i in range(T):
        for j in range(i + 1):
            if i - j < L:
                Q[i, j] = f_s[i - j] * c_t[j]
    
    # Initialize alpha
    global_cfr = np.clip(np.sum(d_t) / max(np.sum(c_t), 1), 0.001, 0.3)
    crude_cfr = np.clip(np.cumsum(d_t) / np.maximum(np.cumsum(c_t), 1), 0.01, 0.9)
    y_logit = np.log(crude_cfr / (1 - crude_cfr))
    
    try:
        BtB = B.T @ B + 0.1 * np.eye(J)
        alpha_init = np.linalg.solve(BtB, B.T @ y_logit)
    except:
        alpha_init = np.full(J, np.log(global_cfr / (1 - global_cfr)) / J)
    
    # EM Algorithm
    sigma_est = 0.1
    
    pbar = tqdm(range(em_iter), desc="fsCFR EM", disable=not verbose)
    
    for em_i in pbar:
        # E-step
        nll, params = _fit_given_sigma(sigma_est, d_t, Q, B, t_k, beta_sign, 
                                       K, J, T, alpha_init, max_iter)
        
        delta_raw = params[J:J+T]
        alpha = params[:J]
        gamma = params[J+T:J+T+2*K] if K > 0 else np.array([])
        
        delta = _center_delta(delta_raw)
        if K > 0:
            beta_abs = np.exp(np.clip(gamma, -5, 5))
            beta_step = beta_sign[:K] * beta_abs[:K]
            beta_slope = beta_sign[K:] * beta_abs[K:]
        else:
            beta_step, beta_slope = np.array([]), np.array([])
        
        int_eff = _compute_intervention_effect(t_k, beta_step, beta_slope, T)
        eta, r = _compute_eta_and_r(alpha, delta, int_eff, B)
        mu = _compute_mu(Q, r)
        
        # Approximate Hessian diagonal for delta
        sigma2_old = sigma_est**2 + 1e-10
        hess_lik = np.zeros(T)
        dr_ddelta = r * (1 - r)
        for i in range(T):
            for t in range(i, T):
                if t - i < Q.shape[1]:
                    hess_lik[i] += mu[t] * dr_ddelta[i]**2
        
        post_var = 1.0 / (hess_lik + 1.0 / sigma2_old + 1e-10)
        
        # M-step
        expected_delta2 = delta_raw**2 + post_var
        sigma_new = np.sqrt(np.mean(expected_delta2))
        sigma_new = np.clip(sigma_new, 0.001, 1.0)
        
        pbar.set_postfix({'σ': f'{sigma_est:.3f}→{sigma_new:.3f}', 'NLL': f'{nll:.0f}'})
        
        if abs(sigma_new - sigma_est) < 0.005:
            break
        sigma_est = sigma_new
    
    pbar.close()
    
    # Final fit
    final_nll, final_params = _fit_given_sigma(sigma_est, d_t, Q, B, t_k, beta_sign,
                                                K, J, T, alpha_init, max_iter)
    
    # Extract results
    alpha = final_params[:J]
    delta_raw = final_params[J:J+T]
    delta = _center_delta(delta_raw)
    
    if K > 0:
        gamma = final_params[J+T:J+T+2*K]
        beta_abs = np.exp(np.clip(gamma, -5, 5))
        beta_step = beta_sign[:K] * beta_abs[:K]
        beta_slope = beta_sign[K:] * beta_abs[K:]
        beta = np.concatenate([beta_step, beta_slope])
    else:
        beta = np.array([])
        beta_step = np.array([])
        beta_slope = np.array([])
    
    tau_alpha = np.exp(final_params[J+T+2*K])
    
    # Compute predictions
    int_eff = _compute_intervention_effect(t_k, beta_step, beta_slope, T)
    eta, r = _compute_eta_and_r(alpha, delta, int_eff, B)
    mu = _compute_mu(Q, r)
    
    # Counterfactual
    eta_cf, r_cf = _compute_eta_and_r(alpha, delta, np.zeros(T), B)
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"fsCFR done: T={T}, K={K}, σ_δ={sigma_est:.4f}, time={total_time:.1f}s")
    
    return {
        'converged': True,
        'value': final_nll,
        'alpha': alpha,
        'delta': delta,
        'beta': beta,
        'sigma_delta': sigma_est,
        'tau_alpha': tau_alpha,
        'eta': eta,
        'r': r,
        'r_cf': r_cf,
        'mu': mu,
        'time': total_time,
    }


# =============================================================================
# Wrapper for methods.py Compatibility
# =============================================================================

def fsCFR_model_wrapper(
    d_t: np.ndarray,
    c_t: np.ndarray,
    f_s: np.ndarray,
    Bm: np.ndarray,
    intervention_times_abs: List[int],
    intervention_signs: List[int],
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """Wrapper function compatible with methods.py interface."""
    T = len(d_t)
    K = len(intervention_times_abs)
    
    if K > 0:
        t_k = np.array(intervention_times_abs, dtype=np.int64)
        if len(intervention_signs) == K:
            beta_sign = np.concatenate([
                np.array(intervention_signs, dtype=np.float64),
                np.array(intervention_signs, dtype=np.float64)
            ])
        else:
            beta_sign = np.array(intervention_signs, dtype=np.float64)
    else:
        t_k = None
        beta_sign = None
    
    result = fsCFR_model(d_t, c_t, f_s, Bm, t_k, beta_sign, verbose=verbose)
    
    return {
        "fsCFR_factual_mean": result['r'],
        "fsCFR_counterfactual_mean": result['r_cf'],
        "fsCFR_baseline_logit": Bm @ result['alpha'],
        "fsCFR_delta": result['delta'],
        "fsCFR_beta_abs_est": np.abs(result['beta'][:K]) if K > 0 else np.array([]),
        "fsCFR_beta_slope_abs_est": np.abs(result['beta'][K:]) if K > 0 else np.array([]),
        "fsCFR_sigma_delta": result['sigma_delta'],
    }
