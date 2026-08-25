import os
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["OPENBLAS_NUM_THREADS"] = "12"

import numpy as np
import sys
import torch
import gpytorch
import emcee
import corner

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "..", "src"))
import paths
import loading
from loading import mean_std_without_zeros

sys.path.insert(0, os.path.join(_script_dir, "train"))
from GP_models import SMF_Model, fgas_Model

###########################################
# Define quantities which we will fit

mcmc_type = 'fgas' # "smf", "fgas", "joint" or any of those with -BF in the end
reset = True
###########################################

###########################################
# Choose the range of the parameters

par_range = 'regular'  #'regular' #'extended', #BF-included
###########################################

###########################################
# Choose the X-ray dataset we wish to fit

dataset = 'kugel' # 'popesso' or 'kugel'
###########################################

print("OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"))
print("MKL_NUM_THREADS =", os.environ.get("MKL_NUM_THREADS"))
np.show_config()

if dataset == 'popesso':
   # Gas Fractions from Popesso et al (2024)
   m500 = np.asarray([12.3,12.6,12.9,13.1,13.3,13.6,14.0,14.5])
   fgas = np.asarray([0.011,0.028,0.019,0.035,0.033,0.039,0.052,0.104])
   err_fgas = np.asarray([0.006,0.013,0.011,0.014,0.013,0.013,0.014,0.016])

elif dataset=='kugel':
   m500 = np.asarray([13.89, 14.06, 14.23, 14.40, 14.57])
   fgas = np.asarray([0.083, 0.094, 0.105, 0.115, 0.130])
   err_fgas = np.asarray([0.002, 0.003, 0.005, 0.008, 0.002])

gsmf = np.loadtxt(os.path.join(paths.REPO_ROOT, 'data', 'GAMA_SDSS_stitched_GSMF_h0p6774.csv'),
                  delimiter=',', comments='#', skiprows=33,
                  usecols=(0, 1, 2))

# Simulation error on fgas
fgas_draws = np.load(os.path.join(paths.RESULTS_DIR, "fgas", "fgas_draws", "fgas_draws100_nbins10.npy"), allow_pickle=True).item()
sim_fgas, err_sim_fgas = mean_std_without_zeros(fgas_draws['ens_fgas'])
sim_m500c, _ = mean_std_without_zeros(fgas_draws['m500'])

# Simulation error on the SMF
smf_draws = np.load(os.path.join(paths.RESULTS_DIR, "smf", "smf_draws", "smf_draws100_nbins10.npy"), allow_pickle=True).item()

sim_smf, err_sim_smf = mean_std_without_zeros(smf_draws['ens_smf'])
err_sim_log_smf = err_sim_smf / sim_smf / np.log(10)
sim_mstar, _ = mean_std_without_zeros(smf_draws['mstar'])

def model_gsmf(gsmf, b_star, b_cv):

    gsmf_shifted = np.copy(gsmf)
    gsmf_shifted[:,1] = gsmf_shifted[:,1] + np.log10(b_cv)
    gsmf_shifted[:,0] = gsmf_shifted[:,0] + b_star

    return gsmf_shifted

##########################################################
################## Load Parameters #######################
##########################################################

# Subgrid parameters of the 31 runs: raw (log10 on rho_rec/ef_kin before
# statistics) and standardized, in the legacy column order.
design = loading.get_lh_design()
(wind_en, wind_vel, rho_rec, sf_ts, ef_kin, ef_high, f_re) = \
    tuple(design['design'][:, j] for j in range(7))

# ---------- 3. Build likelihoods ----------

if mcmc_type in ['smf', 'fgas', 'joint']:
    model_smf = torch.load(os.path.join(paths.GP_MODELS_DIR, "full_model_smf.pth"))
    model_fgas = torch.load(os.path.join(paths.GP_MODELS_DIR, "full_model_fgas.pth"))

    likelihood_smf = torch.load(os.path.join(paths.GP_MODELS_DIR, "full_likelihood_smf.pth"))
    likelihood_fgas = torch.load(os.path.join(paths.GP_MODELS_DIR, "full_likelihood_fgas.pth"))

param_means = design['mean']
param_stds  = design['sigma']

# Lower bounds defined by the simulation campaign itself
if par_range == 'regular':
    lower_bound = np.asarray([0.9,  3.7,  np.log10(0.005), 0.001135, np.log10(0.001), 0.05, 10, -np.inf, -np.inf])
    upper_bound = np.asarray([14.4, 14.8, np.log10(0.5),   0.00454,  np.log10(2),     0.2,  40, np.inf, np.inf])

if par_range == 'extended':
    lower_bound = np.asarray([0.9,  3.7,  np.log10(0.005/100), 0.001135, np.log10(0.001), 0.05, 10, -np.inf, -np.inf])
    upper_bound = np.asarray([14.4, 14.8, np.log10(0.5),   0.00454,  np.log10(2*1000),  0.2,  40, np.inf, np.inf])

if par_range == 'super_extended':
    lower_bound = np.asarray([0.0,  3.7,  np.log10(0.005/10), 0.001135, np.log10(0.001),   0.05,   10, -np.inf, -np.inf])
    upper_bound = np.asarray([14.4, 14.8, np.log10(0.5*100),   0.00454, np.log10(2*100),  0.2,  80, np.inf, np.inf])


def prepare_theta(theta):

    theta_std = np.copy(theta)
    
    theta_std[0]  = (theta_std[0] - param_means[0]) / param_stds[0]
    theta_std[1]  = (theta_std[1] - param_means[1]) / param_stds[1]
    theta_std[2]  = (theta_std[2] - param_means[2]) / param_stds[2]
    theta_std[3]  = (theta_std[3] - param_means[3]) / param_stds[3]
    theta_std[4]  = (theta_std[4] - param_means[4]) / param_stds[4]
    theta_std[5]  = (theta_std[5] - param_means[5]) / param_stds[5]
    theta_std[6]  = (theta_std[6] - param_means[6]) / param_stds[6]

    return theta_std

# --- THE JOINT LIKELIHOOD ---

def log_p(theta):
    # 1. Check for non-physical parameters (Prior check)
    # Define bounds for the 7 physical parameters (theta_1 to theta_7)
    lower_bounds = lower_bound
    upper_bounds = upper_bound
    if not (np.all(theta > lower_bounds) and np.all(theta < upper_bounds)):
        return -np.inf # Implements a flat prior
    else:
        # Prior on b_star
        prior_bstar = -0.5 * ( (theta[7]/0.14)**2 + np.log(2 * np.pi * 0.14**2) )

        # Prior on b_cv
        prior_bcv = -0.5 * ( ( (theta[8]-1)/0.06)**2 + np.log(2 * np.pi * 0.06**2) )

        return prior_bstar + prior_bcv

def log_likelihood_joint(theta, data_dict, models_dict, likelihood_dict, w_smf, w_fgas):
    """
    Joint likelihood for SMF and Gas Fractions, using full predictive covariance.
    data_dict: contains 'mstar', 'smf_obs', 'smf_err', 'mhalo', 'fgas_obs', 'fgas_err', 'mhalo_err'
    models_dict: contains 'gp_smf', 'gp_fgas'
    likelihood_dict: part responsible for adding random noise components, besides unexplained variance
    w_smf: weight that tells us wether to account for SMF during fits
    w_fgas: same but for gas-fractions
    """

    theta_std = prepare_theta(theta)

    if w_smf > 0:
        # --- Part A: SMF Likelihood ---
        gsmf_data = model_gsmf(gsmf, theta[7], theta[8])

        mstar = gsmf_data[:, 0]
        obs_log_phi = gsmf_data[:, 1]
        obs_uncertainty = gsmf_data[:, 2]

        smf_input = np.hstack([mstar[:, None], np.repeat(theta_std[None, :7], len(mstar), axis=0)])
        err_smf_interp = np.interp(mstar, np.log10(sim_mstar), err_sim_log_smf)

        with torch.no_grad():
            pred_smf = likelihood_dict['gp_smf'](
                models_dict['gp_smf'](torch.from_numpy(smf_input).float()),
                noise = torch.from_numpy(err_smf_interp**2).float()
            )
            mu_smf = pred_smf.mean.numpy()
            cov_smf = pred_smf.covariance_matrix.numpy()

        # Total covariance = GP covariance + diagonal observational noise
        Sigma_smf = cov_smf + np.diag(obs_uncertainty**2)

        lnL_smf = _gaussian_loglik(obs_log_phi - mu_smf, Sigma_smf)
        if not np.isfinite(lnL_smf):
            return -np.inf
    else:
        lnL_smf = 0

    if w_fgas > 0:
        # --- Part B: Gas Fraction Likelihood ---
        mhalo = data_dict['mhalo']
        fgas_input = np.hstack([mhalo[:, None], np.repeat(theta_std[None, :7], len(mhalo), axis=0)])
        err_fgas_interp = np.interp(mhalo, np.log10(sim_m500c), err_sim_fgas)

        with torch.no_grad():
            pred_fgas = likelihood_dict['gp_fgas'](
                models_dict['gp_fgas'](torch.from_numpy(fgas_input).float()),
                noise = torch.from_numpy(err_fgas_interp**2).float()
            )
            mu_fgas = pred_fgas.mean.numpy()
            cov_fgas = pred_fgas.covariance_matrix.numpy()

        # Total covariance = GP covariance + diagonal observational noise on f_gas
        # (M_halo uncertainty propagation omitted, as in your original)
        Sigma_fgas = cov_fgas + np.diag(data_dict['fgas_err']**2)

        lnL_fgas = _gaussian_loglik(data_dict['fgas_obs'] - mu_fgas, Sigma_fgas)
        if not np.isfinite(lnL_fgas):
            return -np.inf
    else:
        lnL_fgas = 0

    # --- Part C: Combined ---
    total_lnL = w_smf * lnL_smf + w_fgas * lnL_fgas + log_p(theta)

    return total_lnL if np.isfinite(total_lnL) else -np.inf


def _gaussian_loglik(residual, Sigma, jitter=1e-8):
    """
    Stable multivariate Gaussian log-likelihood via Cholesky:
        -0.5 * [ r^T Sigma^-1 r + log|Sigma| + N log(2 pi) ]
    """
    n = residual.shape[0]
    # Symmetrize defensively (covariance_matrix from GPyTorch can have tiny asymmetry)
    Sigma = 0.5 * (Sigma + Sigma.T)
    try:
        L = np.linalg.cholesky(Sigma + jitter * np.eye(n))
    except np.linalg.LinAlgError:
        # Retry with larger jitter once
        try:
            L = np.linalg.cholesky(Sigma + 1e-4 * np.eye(n))
        except np.linalg.LinAlgError:
            return -np.inf

    alpha = np.linalg.solve(L, residual)            # L alpha = r
    quad = alpha @ alpha                             # r^T Sigma^-1 r
    logdet = 2.0 * np.sum(np.log(np.diag(L)))        # log|Sigma|
    return -0.5 * (quad + logdet + n * np.log(2 * np.pi))


##########################################################
# --- 1. SET UP DATA AND MODELS ---

# Observations for SMF
data_dict = {
    'mstar':     np.array(gsmf[:,0]), # Mass bins
    'smf_obs':   np.array(gsmf[:,1]), # log10(Phi)
    'smf_err':   np.array(gsmf[:,2]), # Uncertainty in log10(Phi)
    
    # Observations for Gas Fractions
    'mhalo':     m500, # Halo mass bins (input)
    'fgas_obs':  fgas, # Measured gas fraction
    'fgas_err':  err_fgas, # Uncertainty in gas fraction (y-axis)
    'mhalo_err': np.array(np.zeros_like(m500))  # Uncertainty in halo mass (x-axis)
}

# Wrap models
models_dict = {
    'gp_smf':  model_smf,
    'gp_fgas': model_fgas
}

likelihood_dict = {
    'gp_smf': likelihood_smf,
    'gp_fgas': likelihood_fgas
}

# Ensure models are in evaluation mode
for m in models_dict.values():
    m.eval()

# Ensure likelihoods are in evaluation mode
for l in likelihood_dict.values():
    l.eval()


############################################################

def run_MCMC(w_smf, w_fgas):

    ndim = 9
    nwalkers = 2 * ndim

    # Initialize walkers within the physical bounds (lower_bound, upper_bound)
    # We start them in a small ball around a reasonable central guess
    p0_start = np.asarray([param_means[0], param_means[1], param_means[2], param_means[3],\
                        param_means[4], param_means[5], param_means[6], 0, 1])
    p0 = p0_start + 1e-4 * np.random.randn(nwalkers, ndim)

    # Ensure initial positions are actually within bounds
    p0 = np.clip(p0, lower_bound + 1e-5, upper_bound - 1e-5)

    # Define where to save the results of this chain
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    os.makedirs(paths.MCMC_CHAINS_DIR, exist_ok=True)
    if mcmc_type in ['smf', 'smf-BF']:
        filename = os.path.join(paths.MCMC_CHAINS_DIR, mcmc_type+"_"+par_range+"_chain.h5")
    elif mcmc_type in ['fgas', 'fgas-BF', 'joint', 'joint-BF']:
        filename = os.path.join(paths.MCMC_CHAINS_DIR, mcmc_type+"_"+par_range+"_"+dataset+"_chain.h5")
    backend = emcee.backends.HDFBackend(filename)
    if reset:
        backend.reset(nwalkers, ndim) # If you want to restart from your current progress, comment this line

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(
        nwalkers, 
        ndim, 
        log_likelihood_joint, # Our joint function
        args=(data_dict, models_dict, likelihood_dict, w_smf, w_fgas),
	backend=backend
    )

    # --- RUNNING MCMC WITH CONVERGENCE CHECK ---
    print("Starting MCMC run with convergence monitoring...")

    # Initialize variables
    max_n = 200000        # Maximum number of steps to allow
    autocorr_tol = 0.01  # Tolerance for fractional change in tau (e.g., 1%)
    min_steps = 50000     # Minimum number of steps before checking tau
    index = 0
    old_tau = np.inf

    # sampler.run_mcmc(p0, max_n, progress=True, thin_by=1)

    # The main loop that runs the sampler until convergence criteria are met
    for sample in sampler.sample(p0, iterations=max_n, progress=False):
        
        # Only check for convergence every 'check_interval' steps
        check_interval = 100
        if sampler.iteration % check_interval:
            continue

        index += 1
        
        # 1. Estimate the autocorrelation time (tau)
        # The default 'c' argument is 50, meaning it requires at least 50 independent samples
        # to estimate tau robustly.
        try:
            # Use discard=0 since we are checking the full chain so far
            tau = sampler.get_autocorr_time(tol=0, discard=0) 
        except emcee.autocorr.AutocorrError:
            # Not enough samples yet to estimate tau. Keep sampling.
            print(f"Step {sampler.iteration}: Not enough samples to estimate tau. Continuing...")
            continue

        # Take the maximum tau across all dimensions
        current_tau = np.max(tau)
        
        # 2. Check for burn-in and chain length convergence
        # We require the chain length to be at least 50 times the estimated tau
        converged = sampler.iteration > (50 * current_tau)

        # 3. Check for stabilization of tau
        # We require the estimated tau to have stabilized (change by less than the tolerance)
        stabilized = np.all(np.abs(old_tau - current_tau) / current_tau < autocorr_tol)
        
        print(f"Step {sampler.iteration}: Max tau = {current_tau:.2f}. Old tau = {old_tau:.2f}. Stabilized: {stabilized}.")

        # Update tau for the next check
        old_tau = current_tau
        
        # Break if both conditions are met AND we are past the minimum steps
        if converged and stabilized and sampler.iteration >= min_steps:
            print("MCMC converged!")
            break

    return sampler

if mcmc_type in ['smf', 'smf-BF']:
    # --- Just SMF
    sampler_smf = run_MCMC(1, 0)
    burnin = int(2 * np.max(sampler_smf.get_autocorr_time(tol=0)))
    smf_samples = sampler_smf.get_chain(discard=burnin, flat=True)
    log_smf = sampler_smf.get_log_prob(discard=burnin, flat=True)

elif mcmc_type in ['fgas', 'fgas-BF']:
    # --- Just fgas
    sampler_fgas = run_MCMC(0, 1)
    burnin = int(2 * np.max(sampler_fgas.get_autocorr_time(tol=0)))
    fgas_samples = sampler_fgas.get_chain(discard=burnin, flat=True)
    load_fgas = sampler_fgas.get_log_prob(discard=burnin, flat=True)

elif mcmc_type in ['joint', 'joint-BF']:
    # --- SMF & fgas
    sampler_joint = run_MCMC(1, 1)
    burnin = int(2 * np.max(sampler_joint.get_autocorr_time(tol=0)))
    joint_samples = sampler_joint.get_chain(discard=burnin, flat=True)
    load_joint = sampler_joint.get_log_prob(discard=burnin, flat=True)
