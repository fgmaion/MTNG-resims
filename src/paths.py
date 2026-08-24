"""
Central filesystem locations, cosmology and simulation conventions for the
MTNG-resims project.

This is the ONLY place where absolute paths and global constants should live.
Defaults point at the original DIPC layout; every location can be overridden
with an environment variable, so the code runs elsewhere without edits::

    export MTNG_BASE=/path/to/MTNG
    export MTNG_RESIMS_BASE=/path/to/MN5_resims

Repository-relative outputs (results, GP models, chains) resolve against
REPO_ROOT automatically -- no configuration needed for collaborators who clone
the repo; override ``MTNG_RESIMS_RESULTS`` etc. only if you want the large
output directories elsewhere.

All values are plain strings so they compose with the existing
string-based code.
"""

import os

# --------------------------------------------------------------------------
# Repository location (computed, not configurable)
# --------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
"""Root of this repository (parent of src/)."""

# --------------------------------------------------------------------------
# Simulation inputs (cluster defaults; override via environment)
# --------------------------------------------------------------------------

#: Parent box: MTNG hydrodynamical run (L = 500 Mpc/h, z=0 snap = 264).
MTNG_BASE = os.environ.get(
    "MTNG_BASE", "/cosmos_storage/simulations/TNG_Family/MTNG")

#: Parent box: MTNG gravity-only counterpart (MTNG-L500-1080-A).
MTNG_DM_BASE = os.environ.get(
    "MTNG_DM_BASE",
    os.path.join(MTNG_BASE, "DM-Gadget4", "MTNG-L500-1080-A"))

#: Root of the MN5 zoom resimulations (contains LH_0..LH_29/, fiducial/,
#: bestfit_run/, param_LH/ and resims_info/).
RESIMS_BASE = os.environ.get(
    "MTNG_RESIMS_BASE", "/cosmos_storage/simulations/TNG_Family/MN5_resims")

#: Precomputed MTNG merger-tree helper arrays (prob-bias products).
MTNG_TREE_BASE = os.environ.get(
    "MTNG_TREE_BASE", "/cosmos_storage/home/fgmaion/prob-bias/MTNG/tree_data")

#: MTNG-mimic lightcones run (used by a couple of validation notebooks).
MIMIC_BASE = os.environ.get(
    "MTNG_MIMIC_BASE", "/cosmos_storage/simulations/TNG_Family/MTNG_mimic")

#: Latin-hypercube design file (30 x 7), seed 1997; see latin-hypercube/code.py.
#: Contains '{0}'/'{1}' placeholders for (npoints, seed).
LH_DESIGN_TEMPLATE = os.environ.get(
    "MTNG_LH_DESIGN_FILE", "/cosmos_storage/data_sharing/MN5_resims/cpars_{0}_{1}.h5")

# --------------------------------------------------------------------------
# Parameter files of the zoom suite
# --------------------------------------------------------------------------

PARAM_DIR = os.path.join(RESIMS_BASE, "param_LH")
"""Directory with the Arepo parameter files of the zoom runs."""

PARAM_TEMPLATE = os.path.join(PARAM_DIR, "param_MTNG-hydro.txt")
"""Fiducial Arepo parameter file (also the template for the LH runs)."""

def zoom_param_file(name):
    """Return the Arepo parameter file of zoom ``name``.

    Parameters
    ----------
    name : str
        'LH_0' ... 'LH_29', 'fiducial', 'bf_sim' or 'bestfit_run'.

    Returns
    -------
    str
        Path to the corresponding param_MTNG-hydro*.txt file.
    """
    if name.startswith("LH_"):
        return os.path.join(PARAM_DIR, "param_MTNG-hydro_{}.txt".format(name[3:]))
    if name in ("bf_sim", "bestfit_run", "bestfit"):
        return os.path.join(PARAM_DIR, "param_MTNG-hydro_bf.txt")
    return PARAM_TEMPLATE


#: Layout of zoom output directories relative to RESIMS_BASE. Must contain a
#: '{name}' placeholder. Legacy layout: '{name}/hydro_output'.
#: Flatiron ceph layout: 'simulations/{name}/hydro_output'.
ZOOM_OUTPUT_LAYOUT = os.environ.get(
    "MTNG_ZOOM_LAYOUT", os.path.join("{name}", "hydro_output"))

def zoom_output_dir(name):
    """Return the hydro_output directory of zoom ``name``."""
    return os.path.join(RESIMS_BASE, ZOOM_OUTPUT_LAYOUT.format(name=name))


# --------------------------------------------------------------------------
# Outputs (repo-relative by default; overridable for big shared storage)
# --------------------------------------------------------------------------

RESULTS_DIR = os.environ.get(
    "MTNG_RESIMS_RESULTS", os.path.join(REPO_ROOT, "results"))
"""Measured statistics per run (SMF, fgas, profiles, draws, ...)."""

GP_MODELS_DIR = os.environ.get(
    "MTNG_GP_MODELS", os.path.join(REPO_ROOT, "gp_train_results"))
"""Trained GP emulator objects."""

MCMC_CHAINS_DIR = os.environ.get(
    "MTNG_MCMC_CHAINS", os.path.join(REPO_ROOT, "mcmc_chains"))
"""MCMC chain outputs."""

CROSSMATCH_DIR = os.environ.get(
    "MTNG_CROSSMATCH", os.path.join(REPO_ROOT, "cross-match"))
"""Cached zoom<->MTNG halo cross-matches (cross_match_<name>.npy)."""

HALO_SEL_DIR = os.path.join(REPO_ROOT, "halo_selection")
"""Halo selection lists (tracked in git) -- do not override via env."""

def halo_selection_file(kind="hydro"):
    """Return the *_halo_sel_1pmbin.txt selection file for ``kind``
    ('hydro' or 'dm')."""
    return os.path.join(HALO_SEL_DIR, "{}_halo_sel_1pmbin.txt".format(kind))

# --------------------------------------------------------------------------
# Simulation conventions & cosmology of the zoom suite
# --------------------------------------------------------------------------

#: Fiducial-zoom cosmology. Omega*/HubbleParam come from the Arepo param
#: files; sigma8/ns/tau are the fixedPk values handed to bacco.
COSMO = {
    "Omega0":      0.3089,
    "OmegaBaryon": 0.0486,
    "OmegaLambda": 0.6911,
    "HubbleParam": 0.6774,
    "sigma8":      0.8159,
    "ns":          0.9667,
    "tau":         0.0965,
}

SNAP_Z0 = 264
"""Snapshot corresponding to z=0 in MTNG (and in every zoom)."""

NPART = 4320
"""Particles along one side of the zoom/mimic box (4320^3)."""

BOXSIZE = 500.0
"""Comoving box size in Mpc/h (MTNG and all zooms)."""

X_SHIFT = 125.0
"""x-offset between MTNG and the MTNG-mimic/zoom coordinates (Mpc/h). Correct
with (pos_x - X_SHIFT) % BOXSIZE when comparing the two coordinate systems."""

HIGHZ_SNAPS = (264, 232, 199, 179, 151, 129, 94)
"""Snapshots actually written out by the zoom runs
(see MTNG_OutputList_Selected.txt / src/create_output_list.py)."""
