"""
Centralized data loading for the MTNG-resims project.

This module absorbs boilerplate that was previously copy-pasted across
``scripts/train/*.py``, ``scripts/measure/*.py`` and ``scripts/mcmc_chain.py``:

* reading the 7 varied Arepo subgrid parameters of the LH suite
  (:func:`get_lh_design`) and building GP training design vectors (:func:`pars`);
* the ``mean_std_without_zeros`` helper used for ensemble-draw statistics;
* loading zoom and parent-box simulations with bacco (:func:`load_zoom`,
  :func:`load_all_zooms`, :func:`load_mtng`);
* reading the repository halo-selection files (:func:`load_halo_selection`).

All filesystem locations come from :mod:`paths`; bacco is imported lazily
inside the simulation loaders, so importing this module never requires bacco
(or even numpy-intensive simulation data).
"""

import os

import numpy as np

import paths

# --------------------------------------------------------------------------
# Run naming conventions
# --------------------------------------------------------------------------

LH_NAMES = ['LH_{:d}'.format(i) for i in range(30)]
"""Names of the 30 Latin-hypercube zoom runs."""

ALL_NAMES = LH_NAMES + ['fiducial']
"""Runs entering the legacy 31-run emulator design (LH runs + fiducial)."""

# --------------------------------------------------------------------------
# Subgrid parameters of the LH suite
# --------------------------------------------------------------------------

#: Logical parameter names in the column order used by every legacy script
#: (and therefore by the trained GP models' ARD kernel). Do NOT reorder.
PARAM_NAMES = ('wind_en', 'wind_vel', 'rho_rec', 'sf_ts',
               'ef_kin', 'ef_high', 'f_re')

#: Mapping logical name -> Arepo parameter-file keyword.
_AREPO_KEY = {
    'wind_en':  'WindEnergyIn1e51erg',
    'wind_vel': 'VariableWindVelFactor',
    'rho_rec':  'WindFreeTravelDensFac',
    'sf_ts':    'MaxSfrTimescale',
    'ef_kin':   'RadioFeedbackFactor',
    'ef_high':  'BlackHoleFeedbackFactor',
    'f_re':     'RadioFeedbackReiorientationFactor',
}

#: Parameters log10-transformed before any statistics (as in legacy code).
LOG_PARAMS = ('rho_rec', 'ef_kin')

_DESIGN_CACHE = {}
"""Memoized get_lh_design() results, keyed by tuple(names)."""


def _read_subgrid_params(param_file):
    """Read the 7 varied subgrid parameters from one Arepo parameter file.

    Parameters
    ----------
    param_file : str
        Path to a param_MTNG-hydro*.txt file.

    Returns
    -------
    list of float
        Values in PARAM_NAMES order. First occurrence wins for repeated
        keywords.
    """
    arepo_to_name = dict((v, k) for k, v in _AREPO_KEY.items())
    values = {}
    with open(param_file) as f:
        for line in f:
            cols = line.split()
            if len(cols) >= 2:
                name = arepo_to_name.get(cols[0])
                if name is not None and name not in values:
                    values[name] = float(cols[1])
    missing = [n for n in PARAM_NAMES if n not in values]
    if missing:
        raise ValueError('{0}: missing subgrid parameters {1}'
                         .format(param_file, missing))
    return [values[n] for n in PARAM_NAMES]


def get_lh_design(names=None):
    """Read the 7 subgrid parameters of the zoom suite from the param files.

    Parameters
    ----------
    names : list of str, optional
        Runs to include, in order. Default is ALL_NAMES (LH_0 ... LH_29,
        fiducial), reproducing the legacy 31-run design.

    Returns
    -------
    dict
        ``names`` : list of str
            Run names, in order.
        ``param_names`` : tuple of str
            Column order of ``raw``/``design`` (== PARAM_NAMES).
        ``raw`` : (len(names), 7) array
            Parameter values with log10 applied to LOG_PARAMS.
        ``design`` : (len(names), 7) array
            Column-wise z-scored version of ``raw``.
        ``mean``, ``sigma`` : (7,) arrays
            Standardization reference in (log-transformed) raw space; needed
            e.g. to standardize new parameter vectors in MCMC.

    Notes
    -----
    Statistics use np.std's default ddof=0, matching the legacy scripts.
    Results are cached; parameter files are only parsed on the first call
    with a given ``names`` tuple.
    """
    if names is None:
        names = ALL_NAMES
    key = tuple(names)
    if key not in _DESIGN_CACHE:
        raw = np.array([_read_subgrid_params(paths.zoom_param_file(n))
                        for n in names], dtype=float)
        for j, pname in enumerate(PARAM_NAMES):
            if pname in LOG_PARAMS:
                raw[:, j] = np.log10(raw[:, j])
        mean = raw.mean(axis=0)
        sigma = raw.std(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            design = (raw - mean) / sigma
        _DESIGN_CACHE[key] = {
            'names': list(names),
            'param_names': PARAM_NAMES,
            'raw': raw,
            'design': design,
            'mean': mean,
            'sigma': sigma,
        }
    return _DESIGN_CACHE[key]


def pars(i, coord):
    """Legacy GP design-vector builder: coordinate + 7 subgrid parameters.

    Parameters
    ----------
    i : int
        Row index into the design of get_lh_design() (0..29 = LH runs,
        30 = fiducial with the default design).
    coord : array_like
        Coordinate values (e.g. log10 stellar mass), length N.

    Returns
    -------
    (N, 8) array
        Column 0 is ``coord``; columns 1..7 hold the standardized subgrid
        parameters of run ``i`` broadcast over all rows.
    """
    coord = np.asarray(coord)
    design = get_lh_design()['design']
    arr = np.empty((len(coord), 8))
    arr[:, 0] = coord
    arr[:, 1:] = design[i]
    return arr


def mean_std_without_zeros(array):
    """Per-column mean and standard deviation excluding exact zeros.

    Ensemble draws of the SMF/fgas leave empty mass bins as exactly zero;
    those entries must not bias bin statistics.

    Parameters
    ----------
    array : (ndraws, nbins) array_like

    Returns
    -------
    (mean, std) : tuple of (nbins,) arrays
        Statistics over the non-zero entries of each column (numpy defaults,
        ddof=0). Columns that are all zero yield NaN, as in legacy code.
    """
    array = np.asarray(array)
    ndraws, nbins = array.shape
    mean = np.zeros(nbins)
    std = np.zeros(nbins)
    for j in range(nbins):
        nz = array[:, j][array[:, j] != 0]
        mean[j] = np.mean(nz)
        std[j] = np.std(nz)
    return mean, std


# --------------------------------------------------------------------------
# Simulation loaders (bacco imported lazily)
# --------------------------------------------------------------------------

def _import_bacco():
    """Import bacco or raise an informative ImportError."""
    try:
        import bacco
    except ImportError as e:
        raise ImportError(
            'bacco is required for simulation loading; activate a '
            'bacco-enabled environment first ({0})'.format(e))
    return bacco


def load_zoom(name, snap=paths.SNAP_Z0, **overrides):
    """Load one zoom resimulation via bacco.Simulation.

    Parameters
    ----------
    name : str
        'LH_0' ... 'LH_29', 'fiducial', 'bf_sim'/'bestfit_run', or any
        directory name understood by paths.zoom_output_dir.
    snap : int
        Snapshot to attach (default: paths.SNAP_Z0 = 264).
    **overrides
        Any extra bacco.Simulation keyword argument, or a replacement for a
        default (e.g. use_ids=False, numpart=4320**3, tree_file=...).

    Returns
    -------
    bacco.Simulation
    """
    bacco = _import_bacco()
    cosmo = paths.COSMO
    kw = dict(
        basedir=paths.zoom_output_dir(name),
        halo_file='groups_{0:03d}/fof_subhalo_tab_{0:03d}'.format(snap),
        dm_file='snapdir_{0:03d}/snapshot_{0:03d}'.format(snap),
        sim_format='TNG500',
        fixedPk=True,
        use_orphans=False,
        use_ids=True,
        numpart=paths.NPART,
        tau=cosmo['tau'],
        ns=cosmo['ns'],
        sigma8=cosmo['sigma8'],
    )
    kw.update(overrides)
    return bacco.Simulation(**kw)


def load_all_zooms(names=None, snap=paths.SNAP_Z0, **overrides):
    """Load several zoom resimulations at once.

    Parameters
    ----------
    names : list of str, optional
        Runs to load (default: ALL_NAMES).
    snap, **overrides
        Passed through to load_zoom.

    Returns
    -------
    dict
        {name: bacco.Simulation}.
    """
    if names is None:
        names = ALL_NAMES
    return dict((n, load_zoom(n, snap=snap, **overrides)) for n in names)


def load_mtng(snap=paths.SNAP_Z0, dm=False, **overrides):
    """Load the parent MTNG box.

    Parameters
    ----------
    snap : int
        Snapshot to attach (default: paths.SNAP_Z0 = 264). Note that the DM
        counterpart run(s) may number their z=0 output differently — pass
        the appropriate snap explicitly in that case.
    dm : bool
        False (default): hydrodynamical MTNG via bacco.utils.load_MTNG at
        paths.MTNG_BASE. True: DM-only counterpart as a gadget4_hdf5
        bacco.Simulation at paths.MTNG_DM_BASE (legacy default
        numpart=1080**3; override numpart/snap for other resolutions).
    **overrides
        Extra/replacement keyword arguments for the underlying bacco call.

    Returns
    -------
    bacco.Simulation
    """
    bacco = _import_bacco()
    if not dm:
        kw = dict(adr=os.path.join(paths.MTNG_BASE, ''), snap=snap)
        kw.update(overrides)
        return bacco.utils.load_MTNG(**kw)
    cosmo = paths.COSMO
    kw = dict(
        basedir=os.path.join(paths.MTNG_DM_BASE, ''),
        dm_file='snapdir_{0:03d}/snapshot_{0:03d}'.format(snap),
        halo_file='groups_{0:03d}/fof_subhalo_tab_{0:03d}'.format(snap),
        sim_format='gadget4_hdf5',
        fixedPk=True,
        sigma8=cosmo['sigma8'],
        tau=cosmo['tau'],
        ns=cosmo['ns'],
        numpart=1080**3,
        use_orphans=False,
        use_ids=False,
    )
    kw.update(overrides)
    return bacco.Simulation(**kw)


# --------------------------------------------------------------------------
# Halo selection
# --------------------------------------------------------------------------

def load_halo_selection(kind='hydro'):
    """Read the MTNG halo selection (roughly one halo per mass bin).

    Parameters
    ----------
    kind : {'hydro', 'dm'}
        Which selection file to read (see paths.halo_selection_file).

    Returns
    -------
    (N,) int array
        FoF halo indices (452 for the default 1-per-mass-bin selection).
    """
    sel = []
    with open(paths.halo_selection_file(kind)) as f:
        for line in f:
            cols = line.split()
            if cols:
                sel.append(int(cols[0]))
    return np.array(sel)
