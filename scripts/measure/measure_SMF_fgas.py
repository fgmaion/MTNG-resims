import numpy as np

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import utils
import paths
import loading

# Some parameters for estimation

Nbins_smf = 15
Nbins_fgas = 10
m30_kpc = True

#name_list = ['LH_{:d}'.format(i) for i in range(30)] + ['fiducial'] + ['bf_sim']
name_list = ['bestfit_run']

snap = 264

## Load the Zooms
zoom = loading.load_all_zooms(name_list, snap=snap)

# Load the Halo Selection
final_sel = loading.load_halo_selection("hydro")

# Perform the cross-match with MTNG halos
xmatch = {}

for name in name_list:
    xmatch[name] = utils.cross_match(zoom[name], snap=264, name=name)

# Load MTNG and get the fraction of halos to do the upweighting
mtng = loading.load_mtng(snap=snap)
h_frac = loading.halo_selection_weights(mtng, sel=final_sel)

zoom_split = {}
zoom_sel = {}
for name in name_list:
    zoom_split[name] = utils.split_halos(zoom[name])

    zoom_sel[name] = {}

    zoom_sel[name]['sel'] = xmatch[name]['ind'][:,np.newaxis,np.newaxis]
    zoom_sel[name]['h_frac'] = h_frac[np.newaxis, :]

# Estimate the SMF

zoom_smf = {}

outdir = os.path.join(paths.RESULTS_DIR, "smf")
os.makedirs(outdir, exist_ok=True)
for name in name_list:
    zoom_smf[name] = zoom_split[name].halo_smf_draws(sel_mask=zoom_sel[name], nbins=Nbins_smf, draws=1, m_30kpc=m30_kpc)
    np.save(os.path.join(outdir, "smf_{}_Nbins{:d}".format(name, Nbins_smf)), [zoom_smf[name]])

zoom_fgas = {}

outdir = os.path.join(paths.RESULTS_DIR, "fgas")
os.makedirs(outdir, exist_ok=True)
for name in name_list:
    zoom_fgas[name] = zoom_split[name].halo_gas_frac_v2(sel_mask=zoom_sel[name], nbins=Nbins_fgas, draws=1)
    np.save(os.path.join(outdir, "fgas_{}_Nbins{:d}".format(name, Nbins_fgas)), [zoom_fgas[name]])
