import numpy as np
import matplotlib.pyplot as plt
import bacco
from random import sample 
import copy
import h5py
import matplotlib

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import utils
import paths
import loading

draws = 100

resolution_level = 1

# Load Gravity-Only version of MTNG (kept from the legacy script; note the
# loaded object was never used downstream there either)
mtng_dm = loading.load_mtng(dm=True)

# Load Lite snapshot of the MTNG
snap = 264
mtng = loading.load_mtng_lite(snap=snap)

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

outdir = os.path.join(paths.RESULTS_DIR, "fgas", "fgas_draws")
os.makedirs(outdir, exist_ok=True)
np.save(os.path.join(outdir, "fgas_draws{:d}_nbins{:d}.npy".format(draws, nbins)), {'m500':ens_m500, 'ens_fgas':ens_fgas})
