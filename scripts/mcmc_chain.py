import os
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["OPENBLAS_NUM_THREADS"] = "12"

import numpy as np
import sys
import torch
import gpytorch
import emcee
import corner
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
   m500 = np.asarray([13.89, 14.06, 14.23, 14.40, 14.57, 14.74, 14.91])
   fgas = np.asarray([0.083, 0.094, 0.105, 0.115, 0.130, 0.130, 0.139])
   err_fgas = np.asarray([0.002, 0.003, 0.005, 0.008, 0.002, 0.002, 0.003])

# Stellar Mass Function from  GAMA (corrected to h=0.7)
# https://arxiv.org/pdf/2203.08539
# For SDSS correction, add 0.0807 dex to number density

GAMA = np.array([
    [6.875, -0.691, 0.176],
    [7.125, -1.084, 0.125],
    [7.375, -1.011, 0.071],
    [7.625, -1.349, 0.092],
    [7.875, -1.287, 0.079],
    [8.125, -1.544, 0.071],
    [8.375, -1.669, 0.045],
    [8.625, -1.688, 0.032],
    [8.875, -1.795, 0.024],
    [9.125, -1.886, 0.020],
    [9.375, -2.055, 0.014],
    [9.625, -2.142, 0.010],
    [9.875, -2.219, 0.009],
    [10.125, -2.274, 0.009],
    [10.375, -2.292, 0.009],
    [10.625, -2.361, 0.010],
    [10.875, -2.561, 0.013],
    [11.125, -2.922, 0.019],
    [11.375, -3.414, 0.032],
    [11.625, -4.704, 0.138]
])

GAMA_corr = np.zeros((GAMA.shape[0], 4))
GAMA_corr[:,0] = GAMA[:,0] - 2*np.log10(0.7)
GAMA_corr[:,1] = GAMA[:,1] + 3*np.log10(0.7)
GAMA_corr[:,2] = GAMA[:,2]

def model_GAMA(GAMA_corr, b_star, b_cv):

    GAMA_corr = np.copy(GAMA_corr)
    GAMA_corr[:,1] = GAMA_corr[:,1] + np.log10(b_cv)
    GAMA_corr[:,0] = GAMA_corr[:,0] + b_star

    return GAMA_corr

##########################################################
################## Load Parameters #######################
##########################################################

wind_en_or      = []
wind_vel_or     = []
rho_rec_or      = []
sf_ts_or        = []
ef_kin_or       = []
ef_high_or      = []
f_re_or         = []

for i in range(31):
    if i<30:
        filename = "/cosmos_storage/simulations/TNG_Family/MN5_resims/param_LH/param_MTNG-hydro_{:d}.txt".format(i)
    else:
        filename = "/cosmos_storage/simulations/TNG_Family/MN5_resims/param_LH/param_MTNG-hydro.txt"

    with open(filename, 'r') as f:
        for line in f.readlines():
            if len(line.split())!=0:
                if line.split()[0] == 'WindEnergyIn1e51erg':
                    wind_en_or.append(float(line.split()[1]))
                if line.split()[0] == 'VariableWindVelFactor':
                    wind_vel_or.append(float(line.split()[1]))
                if line.split()[0] == 'WindFreeTravelDensFac':
                    rho_rec_or.append(float(line.split()[1]))
                if line.split()[0] == 'MaxSfrTimescale':
                    sf_ts_or.append(float(line.split()[1]))
                if line.split()[0] == 'RadioFeedbackFactor':
                    ef_kin_or.append(float(line.split()[1]))
                if line.split()[0] == 'BlackHoleFeedbackFactor':
                    ef_high_or.append(float(line.split()[1]))
                if line.split()[0] == 'RadioFeedbackReiorientationFactor':
                    f_re_or.append(float(line.split()[1]))
        
rho_rec_or = np.log10(rho_rec_or)
ef_kin_or  = np.log10(ef_kin_or)

wind_en   = (np.asarray(wind_en_or) - np.mean(wind_en_or)) / np.std(wind_en_or)
wind_vel  = (np.asarray(wind_vel_or) - np.mean(wind_vel_or)) / np.std(wind_vel_or)
rho_rec   = (np.asarray(rho_rec_or) - np.mean(rho_rec_or)) / np.std(rho_rec_or)
sf_ts     = (np.asarray(sf_ts_or) - np.mean(sf_ts_or)) / np.std(sf_ts_or)
ef_kin    = (np.asarray(ef_kin_or) - np.mean(ef_kin_or)) / np.std(ef_kin_or)
ef_high   = (np.asarray(ef_high_or) - np.mean(ef_high_or)) / np.std(ef_high_or)
f_re      = (np.asarray(f_re_or) - np.mean(f_re_or)) / np.std(f_re_or)

###########################################################################
######################## Load SMF & fgas GP Model ################################

if mcmc_type in ['smf', 'fgas', 'joint']:
    model_smf = torch.load("/cosmos_storage/home/fgmaion/MTNG-resims/gp_train_results/full_model_smf.pth")
    likelihood_smf = torch.load("/cosmos_storage/home/fgmaion/MTNG-resims/gp_train_results/full_likelihood_smf.pth")

    model_fgas = torch.load("/cosmos_storage/home/fgmaion/MTNG-resims/gp_train_results/full_model_fgas.pth")
    likelihood_fgas = torch.load("/cosmos_storage/home/fgmaion/MTNG-resims/gp_train_results/full_likelihood_fgas.pth")

elif mcmc_type in ['smf-BF', 'fgas-BF', 'joint-BF']:
    model_smf = torch.load("/cosmos_storage/home/fgmaion/MTNG-resims/gp_train_results/full_model_smf_bf.pth")
    likelihood_smf = torch.load("/cosmos_storage/home/fgmaion/MTNG-resims/gp_train_results/full_likelihood_smf_bf.pth")

    model_fgas = torch.load("/cosmos_storage/home/fgmaion/MTNG-resims/gp_train_results/full_model_fgas_bf.pth")
    likelihood_fgas = torch.load("/cosmos_storage/home/fgmaion/MTNG-resims/gp_train_results/full_likelihood_fgas_bf.pth")

param_means = np.array([np.mean(wind_en_or), np.mean(wind_vel_or), np.mean(rho_rec_or), 
                        np.mean(sf_ts_or), np.mean(ef_kin_or), np.mean(ef_high_or), np.mean(f_re_or)])
param_stds  = np.array([np.std(wind_en_or), np.std(wind_vel_or), np.std(rho_rec_or), 
                        np.std(sf_ts_or), np.std(ef_kin_or), np.std(ef_high_or), np.std(f_re_or)])

# Lower bounds defined by the simulation campaign itself
if par_range == 'regular':
    lower_bound = np.asarray([0.9,  3.7,  np.log10(0.005), 0.001135, np.log10(0.001), 0.05, 10, -np.inf, -np.inf])
    upper_bound = np.asarray([14.4, 14.8, np.log10(0.5),   0.00454,  np.log10(2),     0.2,  40, np.inf, np.inf])

if par_range == 'extended':
    lower_bound = np.asarray([0.9,  3.7,  np.log10(0.005/100), 0.001135, np.log10(0.001), 0.05, 10, -np.inf, -np.inf])
    upper_bound = np.asarray([14.4, 14.8, np.log10(0.5),   0.00454,  np.log10(2*1000),  0.2,  40, np.inf, np.inf])

if par_range == 'super_extended':
    lower_bound = np.asarray([0.0,  3.7,  np.log10(0.005/10), 0.001135, np.log10(0.001),   0.05,   10, -np.inf, -np.inf])
    upper_bound = np.asarray([14.4, 20,   np.log10(0.5*100),   0.00454, np.log10(2*100),  0.2,  80, np.inf, np.inf])


def prepare_theta(theta):

    theta_std = np.copy(theta)
    
    theta_std[0]  = (theta_std[0] - np.mean(wind_en_or)) / np.std(wind_en_or)
    theta_std[1]  = (theta_std[1] - np.mean(wind_vel_or)) / np.std(wind_vel_or)
    theta_std[2]  = (theta_std[2] - np.mean(rho_rec_or)) / np.std(rho_rec_or)
    theta_std[3]  = (theta_std[3] - np.mean(sf_ts_or)) / np.std(sf_ts_or)
    theta_std[4]  = (theta_std[4] - np.mean(ef_kin_or)) / np.std(ef_kin_or)
    theta_std[5]  = (theta_std[5] - np.mean(ef_high_or)) / np.std(ef_high_or)
    theta_std[6]  = (theta_std[6] - np.mean(f_re_or)) / np.std(f_re_or)

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

def log_likelihood_joint(theta, data_dict, models_dict, w_smf, w_fgas):
    """
    Joint likelihood for SMF and Gas Fractions.
    data_dict: contains 'mstar', 'smf_obs', 'smf_err', 'mhalo', 'fgas_obs', 'fgas_err', 'mhalo_err'
    models_dict: contains 'gp_smf', 'gp_fgas'
    """

    theta_std = prepare_theta(theta)
    
    # --- Part A: SMF Likelihood ---
    # Get the observational model
    GAMA_data = model_GAMA(GAMA_corr, theta[7], theta[8])

    obs_log_phi = GAMA_data[9:,1]
    obs_uncertainty = GAMA_data[9:,2]
    mstar = GAMA_data[9:,0]
    
    smf_input = np.hstack([mstar[:, None], np.repeat(theta_std[None, :7], len(mstar), axis=0)])
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_smf = models_dict['gp_smf'](torch.from_numpy(smf_input).float())
        mu_smf = pred_smf.mean.numpy()
        var_smf = pred_smf.variance.numpy() + data_dict['smf_err']**2
    
    lnL_smf = -0.5 * np.sum(((data_dict['smf_obs'] - mu_smf)**2 / var_smf) + np.log(2 * np.pi * var_smf))

    # --- Part B: Gas Fraction Likelihood ---
    mhalo = data_dict['mhalo']
    # Halo mass might also need normalization if your GP was trained on standardized M_h
    fgas_input = np.hstack([mhalo[:, None], np.repeat(theta_std[None, :7], len(mhalo), axis=0)])
    
    # To handle M_h uncertainty properly, we enable gradients for the slope calculation
    # If the GP is smooth, you can approximate this; here we use the predictive mean.
    x_gas_torch = torch.from_numpy(fgas_input).float().requires_grad_(True)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_fgas = models_dict['gp_fgas'](x_gas_torch)
        mu_fgas = pred_fgas.mean
        var_fgas_emu = pred_fgas.variance.detach().numpy()
        
        # # Calculate slope df/dmhalo to propagate M_halo uncertainty
        # # This is the 'errors-in-variables' correction
        # mu_fgas.sum().backward()
        # slope = x_gas_torch.grad[:, 0].numpy() # Derivative w.r.t the first column (M_h)
        
    mu_fgas = mu_fgas.detach().numpy()
    
    # Total variance = GP_var + Obs_fgas_var + (slope * Obs_Mhalo_var)^2
    var_fgas_total = var_fgas_emu + data_dict['fgas_err']**2 #+ (slope * data_dict['mhalo_err'])**2
    
    lnL_fgas = -0.5 * np.sum(((data_dict['fgas_obs'] - mu_fgas)**2 / var_fgas_total) + np.log(2 * np.pi * var_fgas_total))

    # --- Part C: Combined ---
    total_lnL = w_smf * lnL_smf + w_fgas * lnL_fgas + log_p(theta)
    
    return total_lnL if np.isfinite(total_lnL) else -np.inf


##########################################################
# --- 1. SET UP DATA AND MODELS ---

# Observations for SMF
data_dict = {
    'mstar':     np.array(GAMA_corr[9:,0]), # Mass bins
    'smf_obs':   np.array(GAMA_corr[9:,1]), # log10(Phi)
    'smf_err':   np.array(GAMA[9:,2]), # Uncertainty in log10(Phi)
    
    # Observations for Gas Fractions
    'mhalo':     m500, # Halo mass bins (input)
    'fgas_obs':  fgas, # Measured gas fraction
    'fgas_err':  err_fgas, # Uncertainty in gas fraction (y-axis)
    'mhalo_err': np.array(np.zeros_like(m500))  # Uncertainty in halo mass (x-axis)
}

# Wrap models
models_dict = {
    'gp_smf':  model_smf, # Your first trained GP
    'gp_fgas': model_fgas # Your second trained GP
}

# Ensure models are in evaluation mode
for m in models_dict.values():
    m.eval()

############################################################

def run_MCMC(w_smf, w_fgas):

    ndim = 9
    nwalkers = 2 * ndim

    # Initialize walkers within the physical bounds (lower_bound, upper_bound)
    # We start them in a small ball around a reasonable central guess
    p0_start = np.asarray([np.mean(wind_en_or), np.mean(wind_vel_or), np.mean(rho_rec_or), np.mean(sf_ts_or),\
                        np.mean(ef_kin_or), np.mean(ef_high_or), np.mean(f_re_or), 0, 1])
    p0 = p0_start + 1e-4 * np.random.randn(nwalkers, ndim)

    # Ensure initial positions are actually within bounds
    p0 = np.clip(p0, lower_bound + 1e-5, upper_bound - 1e-5)

    # Define where to save the results of this chain
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = "/cosmos_storage/home/fgmaion/MTNG-resims/mcmc_chains/"+mcmc_type+"_"+par_range+"_"+dataset+"_chain.h5"
    backend = emcee.backends.HDFBackend(filename)
    if reset:
    	backend.reset(nwalkers, ndim) # If you want to restart from your current progress, comment this line

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(
        nwalkers, 
        ndim, 
        log_likelihood_joint, # Our joint function
        args=(data_dict, models_dict, w_smf, w_fgas),
	backend=backend
    )

    # --- RUNNING MCMC WITH CONVERGENCE CHECK ---
    print("Starting MCMC run with convergence monitoring...")

    # Initialize variables
    max_n = 100000        # Maximum number of steps to allow
    autocorr_tol = 0.01  # Tolerance for fractional change in tau (e.g., 1%)
    min_steps = 20000     # Minimum number of steps before checking tau
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
