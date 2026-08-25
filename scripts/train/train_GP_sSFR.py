import numpy as np
import torch
import gpytorch
import os
import copy
from GP_models import SMF_Model

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import paths
import loading

# Subgrid parameters of the 31 runs, standardized (legacy column order)
wind_en, wind_vel, rho_rec, sf_ts, ef_kin, ef_high, f_re = \
    tuple(loading.get_lh_design()['design'][:, j] for j in range(7))

pars = loading.pars

name_list = ['LH_{:d}'.format(i) for i in range(30)] + ['fiducial']# + ['bf_sim'] 

# Load sSFR-Mstar relation
Nbins_sSFR = 10

zoom_sSFR = {}
for i in range(len(name_list)):
    zoom_sSFR[name_list[i]] = np.load(os.path.join(paths.RESULTS_DIR, "sSFR", "sSFR_{}_Nbins{:d}.npy".format(name_list[i], Nbins_sSFR)), allow_pickle=True)[0]

# Filter NaNs
for i in range(len(name_list)):
    mask = ~np.isnan(zoom_sSFR[name_list[i]]['sSFR_mean']) & ~np.isnan(zoom_sSFR[name_list[i]]['mstar_mean'])
    
    zoom_sSFR[name_list[i]]['sSFR_mean'] = zoom_sSFR[name_list[i]]['sSFR_mean'][mask]
    zoom_sSFR[name_list[i]]['mstar_mean'] = zoom_sSFR[name_list[i]]['mstar_mean'][mask]

# Filter infs
for i in range(len(name_list)):
    mask = ~np.isinf(np.log10(zoom_sSFR[name_list[i]]['sSFR_mean'])) & ~np.isinf(np.log10(zoom_sSFR[name_list[i]]['mstar_mean']))
    
    zoom_sSFR[name_list[i]]['sSFR_mean'] = zoom_sSFR[name_list[i]]['sSFR_mean'][mask]
    zoom_sSFR[name_list[i]]['mstar_mean'] = zoom_sSFR[name_list[i]]['mstar_mean'][mask]

# Define the training set
train_sel = np.arange(31)

pars_global = pars(train_sel[0], np.log10(zoom_sSFR[name_list[train_sel[0]]]['mstar_mean']) )
sSFR_global = np.log10(zoom_sSFR[name_list[train_sel[0]]]['sSFR_mean'])

for i in range(len(train_sel)):
    mstar = np.log10(zoom_sSFR[name_list[train_sel[i]]]['mstar_mean'])

    arr = pars(train_sel[i], mstar)

    pars_global = np.vstack((pars_global, arr))

    sSFR_global = np.hstack((sSFR_global, np.log10(zoom_sSFR[name_list[train_sel[i]]]['sSFR_mean'])))
    
train_x = torch.asarray(pars_global, dtype=torch.float)
train_y = torch.asarray(sSFR_global, dtype=torch.float)

torch.set_num_threads(8)

### Define the GP model

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-2))
model = SMF_Model(train_x, train_y, likelihood)

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
#training_iter = 2 if smoke_test else 200

n_restarts = 30
steps_per_restart = 30

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

# ---- restore best ----
model.load_state_dict(best_state['model'])
likelihood.load_state_dict(best_state['likelihood'])

save_path = paths.GP_MODELS_DIR
os.makedirs(save_path, exist_ok=True)

torch.save(model, os.path.join(save_path, "full_model_sSFR.pth"))
torch.save(likelihood, os.path.join(save_path, "full_likelihood_sSFR.pth"))
