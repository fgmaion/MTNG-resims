import numpy as np
import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import utils
import paths
import loading

Nbins_sSFR = 10

name_list = ['LH_{:d}'.format(i) for i in range(30)] + ['fiducial']
#name_list = ['bf_sim']

snap = 264

## Load the Zooms
zoom = loading.load_all_zooms(name_list, snap=snap)

# Load halo selection
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


# Estimate the sSFR-mstar relation
zoom_sSFR = {}

outdir = os.path.join(paths.RESULTS_DIR, "sSFR")
os.makedirs(outdir, exist_ok=True)
for name in name_list:
    zoom_sSFR[name] = zoom_split[name].sSFR_mstar(sel_mask=zoom_sel[name], nbins=Nbins_sSFR)
    np.save(os.path.join(outdir, "sSFR_{}_Nbins{:d}".format(name, Nbins_sSFR)), [zoom_sSFR[name]])
