import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import h5py
import numpy as np
import random
import bacco
import sys
import copy

torch.set_num_threads(16)

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

sys.path.insert(0, "/cosmos_storage/home/fgmaion/MTNG-resims/src")
import utils

sys.path.insert(0, "/cosmos_storage/home/fgmaion/MTNG-resims/scripts/train")
import GP_models

#####################
# Training Function #
#####################

def train(model, likelihood, n_restarts=10, steps_per_restart=1000):

    n_restarts = 10
    steps_per_restart = 1000

    best_state = None
    best_loss = float('inf')

    for r in range(n_restarts):
        print(f"\n=== Restart {r+1}/{n_restarts} ===")
        model.initialize()

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(steps_per_restart):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -torch.sum(mll(output, train_y))
            loss.backward()
            optimizer.step()

        # compute final loss for this restart
        with torch.no_grad():
            final_output = model(train_x)
            final_loss = float(-mll(final_output, train_y))

        print(f"Final loss restart {r+1}: {final_loss:.4f}")

        # store best
        if final_loss < best_loss:
            best_loss = final_loss
            best_state = {
                'model': copy.deepcopy(model.state_dict()),
                'likelihood': copy.deepcopy(likelihood.state_dict()),
            }

    return best_state

#################################################
# Define some information on the Simulation Set #
#################################################

name_list = ['LH_{:d}'.format(i) for i in range(30)] + ['fiducial']
Nsims = len(name_list)

wind_en, wind_vel, rho_rec, sf_ts, ef_kin, ef_high, f_re = utils.get_parameters()

fid_pars = np.array([wind_en[30], wind_vel[30], rho_rec[30], sf_ts[30], ef_kin[30], ef_high[30], f_re[30]])
Nbins_smf = 10

## Load the SMF data
zoom_smf = {}
for i in range(Nsims):
    zoom_smf[name_list[i]] = np.load("/cosmos_storage/home/fgmaion/MTNG-resims/results/smf/new_smf_{}_Nbins{:d}.npy".format(name_list[i], Nbins_smf), allow_pickle=True)[0]

# Filter NaNs
for i in range(Nsims):
    mask = ~np.isnan(zoom_smf[name_list[i]]['smf'][0]) & ~np.isnan(zoom_smf[name_list[i]]['mstar'][0])
    
    zoom_smf[name_list[i]]['smf'][0] = zoom_smf[name_list[i]]['smf'][0][mask]
    zoom_smf[name_list[i]]['mstar'][0] = zoom_smf[name_list[i]]['mstar'][0][mask]

# Estimate the SMF simulation error
smf_draws = np.load("/cosmos_storage/home/fgmaion/MTNG-resims/results/smf/smf_draws/smf_draws100_nbins10.npy", allow_pickle=True).item()

err_smf = np.std(smf_draws['ens_smf'], axis=0)
err_log_smf = torch.asarray(err_smf / np.mean(smf_draws['ens_smf'], axis=0) / np.log(10), dtype=torch.float)

########################################
# Now let's start the cross-validation #
########################################

k = 10

quotient = Nsims // k
remain = Nsims % k

best_state = {}
models = {}
likes = {}

for i in range(k):
    test_sel = list(range(i*quotient, (i+1)*quotient)) if i < k-1 else list(range(i*quotient, Nsims))
    train_sel = [j for j in range(Nsims) if j not in test_sel]

    # Build the training arrays
    pars_global = utils.pars(train_sel[0], np.log10(zoom_smf[name_list[train_sel[0]]]['mstar'][0]) )
    smf_global = np.log10(zoom_smf[name_list[train_sel[0]]]['smf'][0])

    for j in range(len(train_sel)):
        mstar = np.log10(zoom_smf[name_list[train_sel[j]]]['mstar'][0])

        arr = utils.pars(train_sel[j], mstar)

        pars_global = np.vstack((pars_global, arr))

        smf_global = np.hstack((smf_global, np.log10(zoom_smf[name_list[train_sel[j]]]['smf'][0])))

    train_x = torch.asarray(pars_global, dtype=torch.float)
    train_y = torch.asarray(smf_global, dtype=torch.float)

    likes[i] = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=err_log_smf**2, learn_additional_noise=True)
    models[i] = GP_models.SMF_Model(train_x, train_y, likes[i])

    best_state[i] = train(models[i], likes[i], n_restarts=10, steps_per_restart=1000)

    models[i].load_state_dict(best_state[i]['model'])
    likes[i].load_state_dict(best_state[i]['likelihood'])

    save_path = "/cosmos_storage/home/fgmaion/MTNG-resims/scripts/cross-validation/best_models_k{:d}_nbins{:d}/".format(k, Nbins_smf)

    torch.save(models[i], save_path+"full_model_smf_{:d}.pth".format(i))
    torch.save(likes[i], save_path+"full_likelihood_smf_{:d}.pth".format(i))