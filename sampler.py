import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_value, init_to_median
from numpyro.optim import Adam
from tqdm import tqdm
import numpy as np # For fallback rng_key generation

def model(N, K, Bm, Z, beta_signs, dt, fc_mat):
    """
    The proposed Bayesian model for fatality rate estimation.
    This model includes B-splines for the baseline CFR, intervention effects,
    and L1 penalties for regularization.
    
    Args:
        N (int): Total number of time points (sample size).
        K (int): Number of B-spline basis functions (knots).
        Bm (jnp.ndarray): The B-spline basis matrix of shape (N, K).
        Z (jnp.ndarray): The intervention indicator matrix of shape (N, num_interventions),
                         containing (t - t_k)_+ terms.
        beta_signs (jnp.ndarray): An array of -1 or 1 indicating the known direction of each intervention.
        dt (jnp.ndarray): Observed daily number of deaths.
        fc_mat (jnp.ndarray): The convolution matrix (Q) that combines case counts and the delay distribution.
    """

    K_alpha_dim = K 
    K_beta_dim = Z.shape[1] if Z.ndim == 2 and Z.shape[1] > 0 else 0

    # --- Priors and Penalty for B-spline coefficients (alpha) ---
    alpha = numpyro.sample('alpha', dist.Normal(0, 5).expand([K_alpha_dim]))
    lambda_alpha = numpyro.sample("lambda_alpha", dist.Exponential(0.1)) # Prior encouraging smoothness
    if K_alpha_dim >= 3:
        diff_alpha = alpha[2:] - 2 * alpha[1:-1] + alpha[:-2]
        numpyro.factor("alpha_penalty", -lambda_alpha * jnp.sum(jnp.abs(diff_alpha)))

    # --- REPARAMETERIZATION: Priors for gamma, which defines beta_abs ---
    if K_beta_dim > 0:
        # Prior on the unconstrained gamma. Normal(0, 1) is a good default.
        # It induces a Log-Normal prior on beta_abs with a median of exp(0)=1.
        gamma = numpyro.sample("gamma", dist.Normal(0, 1).expand([K_beta_dim]))
        
        # beta_abs is now a deterministic function of gamma
        beta_abs = numpyro.deterministic("beta_abs", jnp.exp(gamma))
        
        beta = numpyro.deterministic("beta", beta_abs * beta_signs) 
        
        # LASSO penalty still applies to beta_abs (or beta)
        # lambda_beta = numpyro.sample("lambda_beta", dist.Exponential(1.0))
        # penalty_val_beta = jnp.sum(jnp.abs(beta)) 
        # numpyro.factor("beta_penalty", -lambda_beta * penalty_val_beta)

        lambda_param = numpyro.sample("lambda", dist.Exponential(1.0).expand([K_beta_dim]))
        lag_matrix = numpyro.deterministic("lag_matrix", (1 - jnp.exp(-lambda_param * Z))) 
        I_eff = jnp.dot(lag_matrix, beta)
    else: # No interventions
        I_eff = jnp.zeros(N)
        numpyro.deterministic("beta", jnp.array([]))
        numpyro.deterministic("beta_abs", jnp.array([]))
        numpyro.deterministic("gamma", jnp.array([])) # Deterministic for consistency
    
    # Random effect term
    sigma_eps = numpyro.sample('phi', dist.HalfNormal(0.15))
    eps = numpyro.sample('eps', dist.Normal(0, sigma_eps).expand([N]))
    
    # Model components and Likelihood (unchanged)
    M = jnp.dot(Bm, alpha)
    I = I_eff

    numpyro.deterministic("M", M) # Baseline logit component
    numpyro.deterministic("I", I) # Intervention logit component
    
    logit_p_cf = M + eps #(if eps were included)
    logit_p = M + I + eps #(if eps were included)
    
    p_cf = jax.scipy.special.expit(logit_p_cf) 
    p = jax.scipy.special.expit(logit_p)      
    
    numpyro.deterministic('p_cf', p_cf)
    numpyro.deterministic('p', p)
    
    mu = jnp.dot(fc_mat, p)
    mu = jnp.maximum(mu, 1e-9) 
    numpyro.sample('dt', dist.Poisson(mu), obs=dt)
    
def sample(model, data, data_init=None,
           method='NUTS', num_warmup=1000, num_samples=1000, num_chains=1, rng_key=None):
    """
    Runs the MCMC sampler for the given model and data.

    Args:
        model (callable): The NumPyro model function.
        data (dict): Dictionary of observed data and model inputs.
        data_init (dict, optional): Dictionary of initial values for parameters.
        method (str, optional): Sampling method, 'NUTS' or 'SVI'. Defaults to 'NUTS'.
        num_warmup (int, optional): Number of warmup steps.
        num_samples (int, optional): Number of posterior samples to generate.
        num_chains (int, optional): Number of MCMC chains to run.
        rng_key (jax.random.PRNGKey, optional): JAX random key for reproducibility. 
                                                **CRITICAL** for parallel simulations.

    Returns:
        tuple: (posterior_samples, mcmc_object)
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(np.random.randint(0, 1_000_000_000))
    
    if method == 'NUTS':
        if data_init is None:
            nuts_kernel = NUTS(model, init_strategy=init_to_median(), target_accept_prob=0.9)
        else:
            nuts_kernel = NUTS(model, init_strategy=init_to_value(values=data_init), target_accept_prob=0.9)
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup,
                    num_samples=num_samples, num_chains=num_chains, progress_bar=False) 
        with numpyro.validation_enabled():
            mcmc.run(rng_key, **data)
        samples = mcmc.get_samples()
        return samples, mcmc
    elif method == 'SVI':
        guide = AutoDiagonalNormal(model)
        optimizer = Adam(1e-2)
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        num_steps = 10000 # SVI specific
        rng_key_svi_run, rng_key_svi_sample = jax.random.split(rng_key)
        svi_result = svi.run(rng_key_svi_run, num_steps, progress_bar=False, **data)

        posterior_samples = {}
        current_guide_sample_key = rng_key_svi_sample
        for _ in tqdm(range(num_samples), desc="SVI Sampling", leave=False):
            current_guide_sample_key, subkey = jax.random.split(current_guide_sample_key)
            single_sample = guide.sample_posterior(subkey, svi_result.params)
            for key_param in single_sample:
                if key_param not in posterior_samples: posterior_samples[key_param] = [single_sample[key_param]]
                else: posterior_samples[key_param].append(single_sample[key_param])
        for key_param in posterior_samples:
            posterior_samples[key_param] = jnp.stack(posterior_samples[key_param])
        return posterior_samples, svi
    else:
        raise ValueError("Invalid sampling method. Choose either 'NUTS' or 'SVI'.")
