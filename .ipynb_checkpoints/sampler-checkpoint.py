# File: sampler.py
# Description: This version uses a standard, unconstrained B-spline basis for the
#              baseline CFR component. Regularization is handled by priors/penalties.

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
import numpy as np

def create_sum_to_zero_contrast_matrix(J):
    """Helper to create the contrast matrix C inside the model."""
    C = jnp.vstack([jnp.eye(J - 1), -jnp.ones((1, J - 1))])
    return C

def model(N, K, Bm, Z, beta_signs, dt, fc_mat, use_constraint=False):
    """
    The proposed Bayesian model for fatality rate estimation.
    This version uses a standard B-spline basis for the baseline CFR.
    
    Args:
        N (int): Total number of time points.
        K (int): Number of B-spline basis functions (J).
        Bm (jnp.ndarray): The B-spline basis matrix (shape N, J).
        Z (jnp.ndarray): The intervention indicator matrix.
        beta_signs (jnp.ndarray): Signs of the intervention effects.
        dt (jnp.ndarray): Observed daily number of deaths.
        fc_mat (jnp.ndarray): The convolution matrix (Q).
    """

    K_alpha_dim = K 
    K_beta_dim = Z.shape[1] if Z.ndim == 2 and Z.shape[1] > 0 else 0
    
    # --- Priors and Penalty for B-spline coefficients (alpha) ---
    if use_constraint:
        # Sample J-1 unconstrained coefficients
        alpha_tilde = numpyro.sample('alpha_tilde', dist.Normal(0, 5).expand([K_alpha_dim - 1]))
        
        # Create the full alpha vector deterministically
        C_contrast = create_sum_to_zero_contrast_matrix(K_alpha_dim)
        alpha = numpyro.deterministic("alpha", C_contrast @ alpha_tilde)
        
        # Apply penalty to the J-1 coefficients
        if K_alpha_dim - 1 >= 3:
            diff_alpha_tilde = alpha_tilde[2:] - 2 * alpha_tilde[1:-1] + alpha_tilde[:-2]
            #lambda_alpha = numpyro.sample("lambda_alpha", dist.Exponential(0.1))
            lambda_alpha = numpyro.sample("lambda_alpha", dist.Gamma(0.01,0.01))
            numpyro.factor("alpha_penalty", -lambda_alpha * jnp.sum(jnp.abs(diff_alpha_tilde)))
    else:
        # Original unconstrained model
        alpha = numpyro.sample('alpha', dist.Normal(0, 5).expand([K_alpha_dim]))
        lambda_alpha = numpyro.sample("lambda_alpha", dist.Exponential(0.1)) 
        if K_alpha_dim >= 3:
            diff_alpha = alpha[2:] - 2 * alpha[1:-1] + alpha[:-2]
            numpyro.factor("alpha_penalty", -lambda_alpha * jnp.sum(jnp.abs(diff_alpha)))

    if K_beta_dim > 0:
        # beta_abs = numpyro.sample("beta_abs", dist.Exponential(1.0).expand([K_beta_dim]))
        beta_abs = numpyro.sample("beta_abs", dist.HalfNormal(1.0).expand([K_beta_dim]))
        beta = numpyro.deterministic("beta", beta_abs * beta_signs) 
        lambda_param = numpyro.sample("lambda", dist.Exponential(1.0).expand([K_beta_dim]))
        lag_matrix = numpyro.deterministic("lag_matrix", (1 - jnp.exp(-lambda_param * Z))) 
        I_eff = jnp.dot(lag_matrix, beta)
    else:
        I_eff = jnp.zeros(N)
        numpyro.deterministic("beta", jnp.array([]))
        numpyro.deterministic("beta_abs", jnp.array([]))

    # # --- Priors for intervention effects ---
    # if K_beta_dim > 0:
    #     # gamma = numpyro.sample("gamma", dist.Normal(0, 1).expand([K_beta_dim]))
    #     # beta_abs = numpyro.deterministic("beta_abs", jnp.exp(gamma))
    #     # beta = numpyro.deterministic("beta", beta_abs * beta_signs) 
        
    #     # # Global shrinkage parameter (controls overall sparsity)
    #     # tau_beta = numpyro.sample("tau_beta", dist.HalfCauchy(0.1))

    #     # # Local shrinkage parameters (one for each beta coefficient)
    #     # lambda_local = numpyro.sample("lambda_local", dist.HalfCauchy(1).expand([K_beta_dim]))
        
    #     # # Regularization for stability
    #     # c_sq = numpyro.sample("c_sq", dist.InverseGamma(0.5, 0.5))
    #     # lambda_tilde = jnp.sqrt( (c_sq * lambda_local**2) / (c_sq + tau_beta**2 * lambda_local**2) )

    #     # # Unscaled coefficients
    #     # beta_unscaled = numpyro.sample("beta_unscaled", dist.Normal(0, 1).expand([K_beta_dim]))
        
    #     # # Final beta_abs with global and local shrinkage
    #     # beta_abs = numpyro.deterministic("beta_abs", jnp.abs(tau_beta * lambda_tilde * beta_unscaled))
    #     # beta = numpyro.deterministic("beta", beta_abs * beta_signs)
    #     # beta_abs = numpyro.sample("beta_abs", dist.Exponential(1.0).expand([K_beta_dim]))
    #     # tau_beta_abs = numpyro.sample("tau_beta_abs", dist.Half)
    #     # beta_abs = numpyro.sample("beta_abs", dist.LogNormal(0.0, tau_beta_abs).expand([K_beta_dim]))

    #     mu_beta = numpyro.sample("mu_beta", dist.Normal(0, 1.0))
    #     sigma_beta = numpyro.sample("sigma_beta", dist.HalfCauchy(1.0))
    #     beta_abs = numpyro.sample(
    #         "beta_abs", 
    #         dist.LogNormal(loc=mu_beta, scale=sigma_beta).expand([K_beta_dim])
    #     )
    #     beta = numpyro.deterministic("beta", beta_abs * beta_signs) 
        
    #     lambda_param = numpyro.sample("lambda", dist.Exponential(1.0).expand([K_beta_dim]))
    #     lag_matrix = numpyro.deterministic("lag_matrix", (1 - jnp.exp(-lambda_param * Z))) 
    #     print(beta.shape)
    #     print(beta_abs.shape)
    #     print(K_beta_dim)
        
    #     I_eff = jnp.dot(lag_matrix, beta)
    # else: # No interventions
    #     I_eff = jnp.zeros(N)
    #     numpyro.deterministic("beta", jnp.array([]))
    #     numpyro.deterministic("beta_abs", jnp.array([]))
    
    # --- Random Effect and Likelihood ---
    # sigma_eps = numpyro.sample('phi', dist.HalfNormal(0.15))
    # eps = numpyro.sample('eps', dist.Normal(0, sigma_eps).expand([N]))
    
    M = jnp.dot(Bm, alpha)
    I = I_eff
    
    # logit_p_cf = M + eps
    # logit_p = M + I + eps    

    logit_p_cf = M 
    logit_p = M + I    
    
    p_cf = jax.scipy.special.expit(logit_p_cf) 
    p = jax.scipy.special.expit(logit_p)      
    
    numpyro.deterministic('p_cf', p_cf)
    numpyro.deterministic('p', p)
    
    mu = jnp.dot(fc_mat, p)
    mu = jnp.maximum(mu, 1e-9) 
    numpyro.sample('dt', dist.Poisson(mu), obs=dt)
    
def sample(model, data, data_init=None, method='NUTS', num_warmup=1000, num_samples=1000, num_chains=1, rng_key=None):
    """Runs the MCMC sampler for the given model and data."""
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    if method == 'NUTS':
        nuts_kernel = NUTS(model, init_strategy=init_to_value(values=data_init), target_accept_prob=0.9)
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup,
                    num_samples=num_samples, num_chains=num_chains, progress_bar=False) 
        
        with numpyro.validation_enabled():
            mcmc.run(rng_key, **data)
            
        samples = mcmc.get_samples()
        return samples, mcmc
    else:
        raise ValueError("Only 'NUTS' method is supported in this version.")
