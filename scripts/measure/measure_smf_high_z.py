import bacco
import numpy as np

import sys
sys.path.append("/cosmos_storage/home/fgmaion/MTNG-resims/src")
import utils
import merger_tree_tools as mgt

snap_num = [237, 232, 214, 199, 179]

# Some parameters for estimation
Nbins_smf = 10
m30_kpc = True

## Load the Zooms
sigma8 = 0.8159 #CHECK ME
ns     = 0.9667 #CHECK ME
tau    = 0.0965 #CHECK ME

name_list = ['LH_{:d}'.format(i) for i in range(30)] + ['fiducial']

snap = 264
zoom = {}

for i in range(len(name_list)):
    base = "/cosmos_storage/simulations/TNG_Family/MN5_resims/"+name_list[i]+"/hydro_output/"
    zoom[name_list[i]] = bacco.Simulation(basedir=base, halo_file="groups_{:03d}/fof_subhalo_tab_{:03d}".format(snap,snap),\
                            tree_file="groups_{0:03d}/subhalo_prog_{0:03d}".format(snap,snap),\
                            sim_format='TNG500', fixedPk=True, use_orphans=False,\
                            tau=tau, ns=ns, sigma8=sigma8, dm_file="snapdir_{:03d}/snapshot_{:03d}".format(snap,snap),\
                            use_ids=False, numpart=4320)

# Load the Halo Selection
with open("/cosmos_storage/simulations/TNG_Family/MN5_resims/resims_info/hydro_halo_sel_1pmbin.txt") as f:
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

snap_0 = 264

zoom_smf = {}
for i in range(len(name_list)):
    base = "/cosmos_storage/simulations/TNG_Family/MN5_resims/"+name_list[i]+"/hydro_output/"
    tree = mgt.tree(snap_0=snap_0, tree_format='zoom', name=name_list[i], sim=zoom[name_list[i]])
    tree.read_tree_opt()
    
    for j in range(len(snap_num)):

        zoom_smf[name_list[i]] = zoom_split[name_list[i]].halo_smf_draws(
                sel_mask=zoom_sel[name_list[i]],
                nbins=Nbins_smf,
                draws=1,
                m_30kpc=True,
                depth=264-snap_num[j],
                tree=tree,
                name=name_list[i])

        z = utils.redshift_from_snap(snap_num[j], base)

        np.save("/cosmos_storage/home/fgmaion/MTNG-resims/results/smf_high_z/smf_{}_Nbins{:d}_z{:.2f}".format(name_list[i], Nbins_smf, z), [zoom_smf[name_list[i]]])