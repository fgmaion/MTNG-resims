import numpy as np

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import utils
import loading

## Runs to cross-match (edit here, or set MTNG_RUNS="LH_3,fiducial"; default: full suite)
NAMES = ['LH_{:d}'.format(i) for i in range(30)] + ['fiducial']
if os.environ.get("MTNG_RUNS"):
    NAMES = os.environ["MTNG_RUNS"].split(",")

snap = 264

## Load the Zooms
zoom = loading.load_all_zooms(NAMES, snap=snap)

# Perform the cross-match with MTNG halos (M200b > 1e13 Msun selection)
xmatch = {}

for name in NAMES:
    xmatch[name] = utils.large_halo_cross_match(zoom[name], snap=264, name=name)
