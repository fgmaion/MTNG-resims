import numpy as np

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import utils
import merger_tree_tools as mgt
import paths
import loading

snap_num = [237, 232, 214, 199, 179]

# Some parameters for estimation
Nbins_smf = 10
m30_kpc = True

name_list = ['LH_{:d}'.format(i) for i in range(30)] + ['fiducial']

snap = 264

## Load the Zooms (with merger trees attached)
zoom = loading.load_all_zooms(
    name_list, snap=snap, use_ids=False,
    tree_file="groups_{0:03d}/subhalo_prog_{0:03d}".format(snap))

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

snap_0 = 264

zoom_smf = {}
outdir = os.path.join(paths.RESULTS_DIR, "smf_high_z")
os.makedirs(outdir, exist_ok=True)
for name in name_list:
    base = paths.zoom_output_dir(name) + "/"
    tree = mgt.tree(snap_0=snap_0, tree_format='zoom', name=name, sim=zoom[name])
    tree.read_tree_opt()
    
    for j in range(len(snap_num)):

        zoom_smf[name] = zoom_split[name].halo_smf_draws(
                sel_mask=zoom_sel[name],
                nbins=Nbins_smf,
                draws=1,
                m_30kpc=True,
                depth=264-snap_num[j],
                tree=tree,
                name=name)

        z = utils.redshift_from_snap(snap_num[j], base)

        np.save(os.path.join(outdir, "smf_{}_Nbins{:d}_z{:.2f}".format(name, Nbins_smf, z)), [zoom_smf[name]])
