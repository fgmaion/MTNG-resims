import numpy as np
import matplotlib.pyplot as plt
import bacco
from random import sample 
import copy
import h5py
import matplotlib

import sys
sys.path.insert(0, "/cosmos_storage/home/fgmaion/MTNG-resims/src")
import utils

draws = 100

basedir = "/cosmos_storage/simulations/TNG_Family/MTNG/DM-Gadget4/MTNG-L500-1080-A/"

resolution_level = 1

sigma8 = 0.8159 #CHECK ME
ns     = 0.9667 #CHECK ME
tau    = 0.0965 #CHECK ME
numpart = int(1080**3)

# Load Gravity-Only version of MTNG
mtng_dm = bacco.Simulation(basedir=basedir, dm_file="snapdir_264/snapshot_264", halo_file="groups_264/fof_subhalo_tab_264",\
			sim_format='gadget4_hdf5', fixedPk=True, sigma8=sigma8,\
		        tau=tau, ns=ns, numpart=numpart, use_orphans=False, use_ids=False)

# Load Lite snapshot of the MTNG
snap = 264
numpart = int(4320**3/64)

adr = "/cosmos_storage/simulations/TNG_Family/MTNG/"

sim_format = 'TNG500'
mtng = bacco.Simulation(verbose=False,basedir=adr, halo_file='groups_%03d/fof_subhalo_tab_%03d'%(snap,snap),\
                        dm_file='64/lite_snap_%03d_mod/diluted_snapshot_%03d'%(snap,snap),\
                        tau=tau, ns=ns, sigma8=sigma8, numpart=numpart, sim_format=sim_format,\
                        use_orphans=False, total_snapshots=265, use_ids=False)

# Start the Halo-Selection framework
hydro_split = utils.split_halos(mtng)
hydro_split.halo_sel_setup()

# Define the halo-mass ranges
mbins = np.concatenate(
    (np.arange(11, 11.5, 0.0025),
    np.arange(11.5, 12.5, 0.005),
    np.arange(12.5, 13.5, 0.025),
    np.arange(13.5, 15.01, 0.125))
)

mass_edges = np.array(list(zip(mbins[:-1], mbins[1:])))
sel_total = hydro_split.halo_sel(mhalo_edges=mass_edges, Nhalos=None)
ens_sel = hydro_split.halo_sel(mhalo_edges=mass_edges, Nhalos=1, draws=100)

###########################################
# Get the gas-fractions in this framework #
###########################################

nbins=10

ens_fgas = np.zeros((draws, nbins-1))
ens_m500 = np.zeros((draws, nbins-1))
weights = np.ones((draws, nbins-1))

temp = hydro_split.lite_mtng_gas_frac(sel_mask=ens_sel, nbins=nbins, draws=draws, red_fac=64)

for i in range(draws):
    ens_fgas[i] = temp['f_gas'][i]
    ens_m500[i] = temp['m500c'][i]

np.save("/cosmos_storage/home/fgmaion/MTNG-resims/results/fgas/fgas_draws/fgas_draws{:d}_nbins{:d}.npy".format(draws, nbins), {'m500':ens_m500, 'ens_fgas':ens_fgas})
