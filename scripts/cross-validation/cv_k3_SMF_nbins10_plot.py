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
Nbins_smf = 15

name_list = ['LH_{:d}'.format(i) for i in range(30)] + ['fiducial']
Nsims = len(name_list)

quotient = Nsims // k
remain = Nsims % k

## Load the SMF data
zoom_smf = {}
for i in range(Nsims):
    zoom_smf[name_list[i]] = np.load("/cosmos_storage/home/fgmaion/MTNG-resims/results/smf/smf_{}_Nbins{:d}.npy".format(name_list[i], Nbins_smf), allow_pickle=True)[0]
    mask = ~np.isnan(zoom_smf[name_list[i]]['smf'][0]) & ~np.isnan(zoom_smf[name_list[i]]['mstar'][0])

    zoom_smf[name_list[i]]['smf'][0] = zoom_smf[name_list[i]]['smf'][0][mask]
    zoom_smf[name_list[i]]['mstar'][0] = zoom_smf[name_list[i]]['mstar'][0][mask]

# Estimate the SMF simulation error
smf_draws = np.load("/cosmos_storage/home/fgmaion/MTNG-resims/results/smf/smf_draws/smf_draws100_nbins14.npy", allow_pickle=True).item()

mean_smf_nz, err_smf_nz = mean_std_without_zeros(smf_draws['ens_smf'])
err_log_smf = err_smf_nz / mean_smf_nz / np.log(10)
mstar_mean, _ = mean_std_without_zeros(smf_draws['mstar'])

f, ax = plt.subplots(figsize=(5.5, 5), dpi=100)
plt.subplots_adjust(wspace=0.15, hspace=0.3)

ax.set_ylabel('$\Delta \log_{10}\Phi$', fontsize=16)
ax.set_xlabel('$\log_{10}(M_*/M_{\odot})$', fontsize=16)

for spine in ax.spines.values():
    spine.set_linewidth(2.5)

# Major and minor ticks
ax.tick_params(axis='both', which='major', width=2.5, length=8, labelsize=12, top=True, right=True, direction='in')
ax.tick_params(axis='both', which='minor', width=1.5, length=4, top=True, right=True, direction='in')
ax.minorticks_on()

diff_all = np.zeros((Nsims, Nbins_smf-1))

count = 0
for i in range(k):
    model = torch.load("/cosmos_storage/home/fgmaion/MTNG-resims/scripts/cross-validation/best_models_k{:d}_nbins{:d}/full_model_smf_{:d}.pth".format(k, Nbins_smf, i))
    likelihood = torch.load("/cosmos_storage/home/fgmaion/MTNG-resims/scripts/cross-validation/best_models_k{:d}_nbins{:d}/full_likelihood_smf_{:d}.pth".format(k, Nbins_smf, i))

    model.eval()
    likelihood.eval()

    test_sel = list(range(i*quotient, (i+1)*quotient)) if i < k-1 else list(range(i*quotient, Nsims))
    train_sel = [j for j in range(Nsims) if j not in test_sel]

    for j in range(len(test_sel)):
        mstar_j = np.log10(zoom_smf[name_list[test_sel[j]]]['mstar'][0])
        smf_j = np.log10(zoom_smf[name_list[test_sel[j]]]['smf'][0])

        test_x = torch.asarray(utils.pars(test_sel[j], mstar_j), dtype=torch.float)

        err_log_smf_j = np.interp(mstar_j, np.log10(mstar_mean), err_log_smf)
        observed_pred = likelihood(model(torch.asarray(test_x, dtype=torch.float)), noise=torch.asarray(err_log_smf_j**2, dtype=torch.float))        

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()

        diff = smf_j - observed_pred.mean.detach().numpy()
        diff_low = smf_j - lower.detach().numpy()
        diff_high = smf_j - upper.detach().numpy()

        ax.plot(mstar_j, diff, color='gray', lw=1, alpha=0.9)
        ax.fill_between(mstar_j, diff_low, diff_high, color='C0', alpha=0.05, edgecolor=None)

        count += 1

ax.set_ylim(-1.2,1.2)

m_min = np.min([np.min(np.log10(zoom_smf[name_list[i]]['mstar'][0])) for i in range(Nsims)])
m_max = np.max([np.max(np.log10(zoom_smf[name_list[i]]['mstar'][0])) for i in range(Nsims)])
mstar_array = np.linspace(m_min, m_max, len(err_log_smf))

plt.savefig("/cosmos_storage/home/fgmaion/MTNG-resims/scripts/cross-validation/cv_k3_SMF_nbins{:d}.pdf".format(Nbins_smf), bbox_inches='tight')