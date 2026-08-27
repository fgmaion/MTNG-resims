# MTNG-resims

Analysis code for the **multi-zoom resimulation campaign of MillenniumTNG (MTNG)
halos** presented in:

> F. Maion, R. E. Angulo, V. Springel, S. Genel & G. L. Bryan,
> *Evaluating the flexibility of the MillenniumTNG galaxy formation model with
> multi-zoom re-simulations*, arXiv:2607.13151 (2026)
> — <https://arxiv.org/abs/2607.13151>

Starting from the MTNG simulation (500 Mpc/h box, z=0 snapshot 264), a sample of
~450 halos was resimulated at higher resolution using a multi-zoom technique that
resimulates several sub-regions of the same volume simultaneously. The suite
comprises **30 resimulations sampling a Latin-hypercube (LH) design over 7
Arepo/TNG subgrid parameters** controlling stellar and AGN feedback, plus a
fiducial run and a best-fit run. This repository contains the full analysis chain:

1. cross-matching of zoom halos to their counterparts in the full MTNG volume;
2. measurement of observables per run: the galaxy stellar-mass function (SMF),
   halo gas fractions, sSFR, the size–mass and stellar-to-halo-mass relations,
   the black-hole mass function, MBH–M*, and group/cluster gas & density
   profiles (also at z>0);
3. Gaussian-process emulation of the SMF and gas fractions over the parameter
   space, with k-fold cross-validation;
4. MCMC calibration of the emulators against observational data
   (GAMA/SDSS GSMF; cluster gas fractions).

## Citation

If you use this code, please cite the paper:

```bibtex
@article{Maion2026_MTNG_resims,
  author  = {Maion, Francisco and Angulo, Raul E. and Springel, Volker
             and Genel, Shy and Bryan, Greg L.},
  title   = {Evaluating the flexibility of the {MillenniumTNG} galaxy
             formation model with multi-zoom re-simulations},
  journal = {arXiv preprint arXiv:2607.13151},
  year    = {2026},
  doi     = {10.48550/arXiv.2607.13151},
}
```

## Repository layout

| Path | Contents |
| ---- | -------- |
| `src/` | Core library: `paths.py` (all locations/constants, env-var overrides), `loading.py` (loaders for zooms/MTNG/LH design), `utils.py` (measurements, cross-matching), `merger_tree_tools.py` (SUBLINK-style tree walking) |
| `scripts/measure/` | Measurement scripts per observable (SMF, fgas, sSFR, SMHM, size–mass, BHMF, MBH–M*, high-z SMF, profiles) |
| `scripts/train/` | GP emulator definitions (`GP_models.py`) and one training script per observable (gpytorch) |
| `scripts/cross-validation/` | k-fold cross-validation of the emulators |
| `scripts/uncertainty/` | Repeated random-subsampling measurements in the full MTNG volumes (sim/emulator uncertainty) |
| `scripts/cross_match.py`, `large_halo_cross_match.py` | Zoom ↔ MTNG halo cross-matching |
| `scripts/mcmc_chain.py` | emcee MCMC fit of the emulators to observational data |
| `latin-hypercube/` | LH design → 30 Arepo parameter files (`code.py`, `param_MTNG-hydro*.txt`) |
| `halo_selection/` | FoF indices of the 452 selected halos (hydro & DM selection lists) |
| `data/` | Observational inputs (literature CSVs, see *Data availability*) |
| `Notebooks/` | Exploratory and paper-analysis notebooks (validation, plots, profiles, ...) |
| `tests/` | `smoke_test.py` (API snapshot) and `test_loading.py` (sandbox + real-data loader tests) |

## Requirements

There is no packaging: every script bootstraps `src/` onto `sys.path` relative to
its own location, so any clone works out of the box. Python dependencies:

- `numpy`, `scipy`, `h5py`, `matplotlib` — everywhere;
- `bacco` (Angulo et al.) — loading MTNG/zoom simulation outputs;
- `torch` + `gpytorch` — GP training, cross-validation and MCMC;
- `emcee` (+ `corner` for chain plots) — MCMC;
- `halotools` — only `scripts/measure/compute_profiles.py`.

The code has been run with Python 3.6–3.11; on the authors' cluster the canonical
module set is gcc/13.3.0, python/3.11.11, fftw/3.3.10, gsl/2.7.1, hdf5/1.12.3.

## Configuration

All absolute paths and simulation conventions live in **`src/paths.py`** and can
be overridden with environment variables — no code edits needed to run on a
different machine:

| Variable | Meaning | Default |
| -------- | ------- | ------- |
| `MTNG_BASE` | MTNG hydro box (z=0 snap = 264) | DIPC layout |
| `MTNG_DM_BASE` | MTNG gravity-only counterpart | `$MTNG_BASE/DM-Gadget4/MTNG-L500-1080-A` |
| `MTNG_RESIMS_BASE` | Root of the zoom suite (`LH_*`, `fiducial`, `bestfit_run/`, `param_LH/`) | DIPC layout |
| `MTNG_ZOOM_LAYOUT` | Output path template relative to `MTNG_RESIMS_BASE`, with `{name}` | `{name}/hydro_output` |
| `MTNG_TREE_BASE` | Precomputed MTNG merger-tree helper arrays | DIPC layout |
| `MTNG_MIMIC_BASE`, `MTNG_MIMIC_DM_BASE` | MTNG-mimic lightcones / DM output (validation notebooks, `get_bpo`) | DIPC layout |
| `MTNG_BIAS_DIR` | Cache for probabilistic-bias fits | DIPC layout |
| `CAMELS_BASE` | CAMELS LH catalogs (`camels_*` helpers) | DIPC layout |
| `MTNG_LH_DESIGN_FILE` | LH design file, with `{0}/{1}` = (npoints, seed) placeholders | DIPC layout |
| `MTNG_RESIMS_RESULTS` | Measured statistics output dir | `<repo>/results` |
| `MTNG_GP_MODELS` | Trained GP model objects | `<repo>/gp_train_results` |
| `MTNG_MCMC_CHAINS` | MCMC chain outputs | `<repo>/mcmc_chains` |
| `MTNG_CROSSMATCH` | Cross-match cache (`cross_match_<name>.npy`) | `<repo>/cross-match` |

In addition, `MTNG_RUNS` (comma-separated run names, e.g. `"LH_3,fiducial"`)
restricts which zooms the cross-match scripts process (default: full suite).

Cosmology and conventions of the suite (Ωm=0.3089, h=0.6774, ..., snapshot 264 ≡
z=0, the 125 Mpc/h x-shift between MTNG and zoom coordinates) are defined once in
`src/paths.py`.

## Usage

The typical pipeline order is:

```bash
# 1. Cross-match zoom halos to the MTNG selection
python scripts/cross_match.py                 # MTNG_RUNS="LH_0,fiducial" to subset

# 2. Measure observables (one script per observable)
python scripts/measure/measure_SMF_fgas.py    # edit name_list near the top to subset

# 3. Train the GP emulators (requires torch + gpytorch)
python scripts/train/train_GP_SMF.py
python scripts/train/train_GP_fgas.py

# 4. Cross-validate
python scripts/cross-validation/<cv_script>.py

# 5. Fit the emulators to observations
#    set mcmc_type ('smf' | 'fgas' | 'joint', optionally '-BF') near the top of the script
python scripts/mcmc_chain.py
```

All of the above are routinely run as SLURM batch jobs on the authors' cluster.

Tests:

```bash
python tests/smoke_test.py    # structural API snapshot (runs anywhere)
python tests/test_loading.py  # sandbox layer runs anywhere; real-data layer
                              # activates automatically when the MTNG_* env vars
                              # point at real simulation data
```

## Notebooks

`Notebooks/` contains the exploratory and paper-analysis notebooks. Notebook
outputs are stripped on commit via `nbstripout` (configured in `.gitattributes`);
after cloning, run once per checkout:

```bash
pip install nbstripout && nbstripout --install
```

## Data availability

Simulation outputs (MTNG, the zoom suite) are not part of this repository.
`data/` contains the observational measurements used for calibration: the
GAMA/SDSS stitched GSMF, Bernardi+2018, Li & White 2009, Moustakas, Wang+2024,
Graham & Sahu 2023, and the GAMA size–mass relation (Pakmor et al.).

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2024–2026 Francisco Maion & contributors.
