import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
import numpy as np

def create_sum_to_zero_contrast_matrix(J):
    """
    Creates a contrast matrix C for sum-to-zero constraints.

    Args:
        J (int): The dimension of the original parameter vector.

    Returns:
        jnp.ndarray: The contrast matrix of shape (J, J-1).
    """
    C = jnp.vstack([jnp.eye(J - 1), -jnp.ones((1, J - 1))])
    return C

def model(N, K, Bm, Z, beta_signs, dt, fc_mat, use_constraint=False):
    """
    The Bayesian model for fatality rate estimation using a B-spline basis.

    Args:
        N (int): Total number of time points.
        K (int): Number of B-spline basis functions (J).
        Bm (jnp.ndarray): The B-spline basis matrix (shape N, J).
        Z (jnp.ndarray): The intervention indicator matrix.
        beta_signs (jnp.ndarray): Signs of the intervention effects.
        dt (jnp.ndarray): Observed daily number of deaths.
        fc_mat (jnp.ndarray): The convolution matrix (Q).
        use_constraint (bool): Flag to indicate whether to use sum-to-zero constraints on alpha.
    """
    K_alpha_dim = K 
    K_beta_dim = Z.shape[1] if Z.ndim == 2 and Z.shape[1] > 0 else 0
    
    if use_constraint:
        alpha_tilde = numpyro.sample('alpha_tilde', dist.Normal(0, 5).expand([K_alpha_dim - 1]))
        C_contrast = create_sum_to_zero_contrast_matrix(K_alpha_dim)
        alpha = numpyro.deterministic("alpha", C_contrast @ alpha_tilde)
        
        if K_alpha_dim - 1 >= 3:
            diff_alpha_tilde = alpha_tilde[2:] - 2 * alpha_tilde[1:-1] + alpha_tilde[:-2]
            lambda_alpha = numpyro.sample("lambda_alpha", dist.Gamma(0.01,0.01))
            numpyro.factor("alpha_penalty", -lambda_alpha * jnp.sum(jnp.abs(diff_alpha_tilde)))
    else:
        alpha = numpyro.sample('alpha', dist.Normal(0, 5).expand([K_alpha_dim]))
        lambda_alpha = numpyro.sample("lambda_alpha", dist.Exponential(0.1)) 
        if K_alpha_dim >= 3:
            diff_alpha = alpha[2:] - 2 * alpha[1:-1] + alpha[:-2]
            numpyro.factor("alpha_penalty", -lambda_alpha * jnp.sum(jnp.abs(diff_alpha)))

    if K_beta_dim > 0:
        beta_abs = numpyro.sample("beta_abs", dist.HalfNormal(1.0).expand([K_beta_dim]))
        beta = numpyro.deterministic("beta", beta_abs * beta_signs) 
        lambda_param = numpyro.sample("lambda", dist.Exponential(1.0).expand([K_beta_dim]))
        lag_matrix = numpyro.deterministic("lag_matrix", (1 - jnp.exp(-lambda_param * Z))) 
        I_eff = jnp.dot(lag_matrix, beta)
    else:
        I_eff = jnp.zeros(N)
        numpyro.deterministic("beta", jnp.array([]))
        numpyro.deterministic("beta_abs", jnp.array([]))
    
    M = jnp.dot(Bm, alpha)
    I = I_eff
    
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
    """
    Runs the MCMC sampler for the given model and data.

    Args:
        model (function): The NumPyro model to be sampled.
        data (dict): A dictionary of data to be passed to the model.
        data_init (dict, optional): A dictionary of initial values for the parameters. Defaults to None.
        method (str, optional): The MCMC method to use. Defaults to 'NUTS'.
        num_warmup (int, optional): The number of warmup steps. Defaults to 1000.
        num_samples (int, optional): The number of samples to generate. Defaults to 1000.
        num_chains (int, optional): The number of MCMC chains to run. Defaults to 1.
        rng_key (jax.random.PRNGKey, optional): The random key for reproducibility. Defaults to None.

    Returns:
        tuple: A tuple containing the posterior samples and the MCMC object.
    """
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
