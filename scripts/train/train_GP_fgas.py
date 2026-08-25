import numpy as np
import torch
import gpytorch
import os
import copy
from GP_models import fgas_Model

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import paths
import loading
from loading import mean_std_without_zeros

# Subgrid parameters of the 31 runs, standardized (legacy column order)
wind_en, wind_vel, rho_rec, sf_ts, ef_kin, ef_high, f_re = \
    tuple(loading.get_lh_design()['design'][:, j] for j in range(7))

pars = loading.pars

name_list = ['LH_{:d}'.format(i) for i in range(30)] + ['fiducial']# + ['bf_sim']

# Load SMF
Nbins_fgas = 8

zoom_fgas = {}
for i in range(len(name_list)):
    zoom_fgas[name_list[i]] = np.load(os.path.join(paths.RESULTS_DIR, "fgas", "fgas_{}_Nbins{:d}.npy".format(name_list[i], Nbins_fgas)), allow_pickle=True)[0]

# Filter NaNs
for i in range(len(name_list)):
    mask = ~np.isnan(zoom_fgas[name_list[i]]['m500c'][0])
    
    zoom_fgas[name_list[i]]['f_gas'][0] = zoom_fgas[name_list[i]]['f_gas'][0][mask]
    zoom_fgas[name_list[i]]['m500c'][0] = zoom_fgas[name_list[i]]['m500c'][0][mask]

# Estimate the fgas simulation error
fgas_draws = np.load(os.path.join(paths.RESULTS_DIR, "fgas", "fgas_draws", "fgas_draws100_nbins10.npy"), allow_pickle=True).item()
sim_fgas, err_sim_fgas = mean_std_without_zeros(fgas_draws['ens_fgas'])
sim_m500c, _ = mean_std_without_zeros(fgas_draws['m500'])

# Define the training set
train_sel = np.arange(31)

pars_global = pars(train_sel[0], np.log10(zoom_fgas[name_list[train_sel[0]]]['m500c'][0]) )
fgas_global = zoom_fgas[name_list[train_sel[0]]]['f_gas'][0]

for i in range(len(train_sel)):
    m500c = np.log10(zoom_fgas[name_list[train_sel[i]]]['m500c'][0])

    arr = pars(train_sel[i], m500c)

    pars_global = np.vstack((pars_global, arr))

    fgas_global = np.hstack((fgas_global, zoom_fgas[name_list[train_sel[i]]]['f_gas'][0]))

train_x = torch.asarray(pars_global, dtype=torch.float)
train_y = torch.asarray(fgas_global, dtype=torch.float)

torch.set_num_threads(8)

### Define the GP model

# initialize likelihood and model
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.asarray(err_sim_fgas**2, dtype=torch.float), learn_additional_noise=True)
model = fgas_Model(train_x, train_y, likelihood)

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
#training_iter = 2 if smoke_test else 200

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

# ---- restore best ----
model.load_state_dict(best_state['model'])
likelihood.load_state_dict(best_state['likelihood'])

save_path = paths.GP_MODELS_DIR
os.makedirs(save_path, exist_ok=True)

torch.save(model, os.path.join(save_path, "full_model_fgas.pth"))
torch.save(likelihood, os.path.join(save_path, "full_likelihood_fgas.pth"))
