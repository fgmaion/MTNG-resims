import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
import sys

sys.path.insert(0, "/cosmos_storage/home/fgmaion/MTNG-resims/src")
import utils

sys.path.insert(0, "/cosmos_storage/home/fgmaion/MTNG-resims/scripts/train")
import GP_models

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

def mean_std_without_zeros(array):
    ndraws, nbins = array.shape
    mean_values = []
    std_values = []

    for i in range(nbins):
        non_zero_values = array[:, i][array[:, i] != 0]
        mean_values.append(np.mean(non_zero_values))
        std_values.append(np.std(non_zero_values))

    return np.array(mean_values), np.array(std_values)

k = 10

name_list = ['LH_{:d}'.format(i) for i in range(30)] + ['fiducial']
Nsims = len(name_list)

quotient = Nsims // k
remain = Nsims % k

Nbins_fgas = 10

## Load the fgas data
zoom_fgas = {}
for i in range(Nsims):
    zoom_fgas[name_list[i]] = np.load("/cosmos_storage/home/fgmaion/MTNG-resims/results/fgas/fgas_{}_Nbins{:d}.npy".format(name_list[i], Nbins_fgas), allow_pickle=True)[0]

# Estimate the fgas simulation error
fgas_draws = np.load("/cosmos_storage/home/fgmaion/MTNG-resims/results/fgas/fgas_draws/fgas_draws100_nbins10.npy", allow_pickle=True).item()
mean_fgas, err_fgas = mean_std_without_zeros(fgas_draws['ens_fgas'])
mean_m500c, _ = mean_std_without_zeros(fgas_draws['m500'])

err_fgas = torch.asarray(err_fgas, dtype=torch.float)

f, ax = plt.subplots(figsize=(5.5, 5), dpi=100)
plt.subplots_adjust(wspace=0.15, hspace=0.3)

ax.set_ylabel('$\Delta f_\mathrm{gas}$', fontsize=16)
ax.set_xlabel('$\log_{10}(M_{500,c}/M_{\odot})$', fontsize=16)

for spine in ax.spines.values():
    spine.set_linewidth(2.5)

# Major and minor ticks
ax.tick_params(axis='both', which='major', width=2.5, length=8, labelsize=12, top=True, right=True, direction='in')
ax.tick_params(axis='both', which='minor', width=1.5, length=4, top=True, right=True, direction='in')
ax.minorticks_on()

ax.set_ylim(-0.05, 0.05)

for i in range(k):
    model = torch.load("/cosmos_storage/home/fgmaion/MTNG-resims/scripts/cross-validation/best_models_k{:d}_nbins{:d}/full_model_fgas_{:d}.pth".format(k, Nbins_fgas, i))
    likelihood = torch.load("/cosmos_storage/home/fgmaion/MTNG-resims/scripts/cross-validation/best_models_k{:d}_nbins{:d}/full_likelihood_fgas_{:d}.pth".format(k, Nbins_fgas, i))

    model.eval()
    likelihood.eval()

    test_sel = list(range(i*quotient, (i+1)*quotient)) if i < k-1 else list(range(i*quotient, Nsims))
    train_sel = [j for j in range(Nsims) if j not in test_sel]

    for j in range(len(test_sel)):
        sim_m500c = np.log10(zoom_fgas[name_list[test_sel[j]]]['m500c'][0])
        sim_fgas = zoom_fgas[name_list[test_sel[j]]]['f_gas'][0]

        test_x = torch.asarray(utils.pars(test_sel[j], sim_m500c), dtype=torch.float)
        err_interp = np.interp(sim_m500c, np.log10(mean_m500c), err_fgas)
        observed_pred = likelihood(model(torch.asarray(test_x, dtype=torch.float)), noise=torch.asarray(err_interp**2, dtype=torch.float))        

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        diff_low = sim_fgas - lower.detach().numpy()
        diff_high = sim_fgas - upper.detach().numpy()

        ax.fill_between(sim_m500c, diff_low, diff_high, color="C0", alpha=0.05)
        ax.plot(sim_m500c, sim_fgas - observed_pred.mean.detach().numpy(),  color='gray', lw=1, alpha=0.9)

plt.savefig("/cosmos_storage/home/fgmaion/MTNG-resims/scripts/cross-validation/cv_k3_fgas_nbins10.pdf", bbox_inches='tight')