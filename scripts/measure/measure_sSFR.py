import numpy as np
import bacco
import matplotlib.pyplot as plt

import sys
sys.path.append("/cosmos_storage/home/fgmaion/MTNG-resims/src")
import utils

Nbins_sSFR = 10

## Load the Zooms
sigma8 = 0.8159 #CHECK ME
ns     = 0.9667 #CHECK ME
tau    = 0.0965 #CHECK ME

name_list = ['LH_{:d}'.format(i) for i in range(30)] + ['fiducial']
#name_list = ['bf_sim']

snap = 264
zoom = {}

for i in range(len(name_list)):
    base = "/cosmos_storage/simulations/TNG_Family/MN5_resims/"+name_list[i]+"/hydro_output/"
    zoom[name_list[i]] = bacco.Simulation(basedir=base, halo_file="groups_{:03d}/fof_subhalo_tab_{:03d}".format(snap,snap), sim_format='TNG500', fixedPk=True, use_orphans=False,\
                            tau=tau, ns=ns, sigma8=sigma8, dm_file="snapdir_{:03d}/snapshot_{:03d}".format(snap,snap), use_ids=True, numpart=4320)

# Load halo selection
with open("/cosmos_storage/home/fgmaion/MTNG-resims/halo_selection/hydro_halo_sel_1pmbin.txt") as f:
    final_sel = []
    for line in f.readlines():
        final_sel.append(int(line.split()[0]))
final_sel = np.array(final_sel)

# Perform the cross-match with MTNG halos
xmatch = {}

for i in range(len(name_list)):
    xmatch[name_list[i]] = utils.cross_match(zoom[name_list[i]], snap=264, name=name_list[i])

# Load MTNG and get the fraction of halos to do the upweighting
mtng = bacco.utils.load_MTNG(adr="/cosmos_storage/simulations/TNG_Family/MTNG/", snap=264)

m200b = np.log10(1e10 * mtng.fof['halo_m200b'])

mbins = np.concatenate(
    (np.arange(11, 11.5, 0.0025),
    np.arange(11.5, 12.5, 0.005),
    np.arange(12.5, 13.5, 0.025),
    np.arange(13.5, 15.01, 0.125))
)

h_frac = np.zeros(len(final_sel))
for m in range(len(mbins)-1):
    h_frac[m] = np.where(( m200b[final_sel]>=mbins[m]) & ( m200b[final_sel]<mbins[m+1]))[0].shape[0] / \
             np.where(( m200b>=mbins[m]) & ( m200b<mbins[m+1]))[0].shape[0]

zoom_split = {}
zoom_sel = {}
for i in range(len(name_list)):
    zoom_split[name_list[i]] = utils.split_halos(zoom[name_list[i]])

    zoom_sel[name_list[i]] = {}

    zoom_sel[name_list[i]]['sel'] = xmatch[name_list[i]]['ind'][:,np.newaxis,np.newaxis]
    zoom_sel[name_list[i]]['h_frac'] = h_frac[np.newaxis, :]


# Estimate the stellar-mass to halo-mass relation
zoom_sSFR = {}

for i in range(len(name_list)):
    zoom_sSFR[name_list[i]] = zoom_split[name_list[i]].sSFR_mstar(sel_mask=zoom_sel[name_list[i]], nbins=Nbins_sSFR)
    np.save("/cosmos_storage/home/fgmaion/MTNG-resims/results/sSFR/sSFR_{}_Nbins{:d}".format(name_list[i], Nbins_sSFR), [zoom_sSFR[name_list[i]]])
