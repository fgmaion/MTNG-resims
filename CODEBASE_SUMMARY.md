# Codebase Summary (auto-generated notes)

## Purpose

Analysis code for **zoom-in resimulations of MTNG** (MillenniumTNG) halos, used to
study how **baryonic (subgrid) physics** affects galaxy statistics — with the stated
goal of analysing **intrinsic alignments (IA) and their baryonic dependence** (see
`README.md`). The repo covers the full workflow cone:

> LH parameter sampling → zoom resimulation setup → halo selection & cross-matching
> → observable measurements → Gaussian-Process emulation → MCMC calibration
> against observational data → validation plots.

Git remote: `git@github.com:fgmaion/MTNG-resims.git` (paper-linked per git log).

## Simulations & setup

- Parent sim: **MTNG** (500 Mpc/h box, 8640^3; loaded via `bacco.utils.load_MTNG`).
- Resims: **MN5_resims** zooms on `/cosmos_storage/simulations/TNG_Family/MN5_resims/`.
  - 30 **Latin-hypercube (LH) runs** (`LH_0` … `LH_29`), one **fiducial** run, plus a
    `bf_sim` / `bestfit_run` (best-fit parameters).
  - Zooms share MTNG-mimic ICs: `numpart=4320`, `BoxSize=500`, snapshot 264 ≡ z=0.
  - Fixed cosmology: Ωm=0.3089, ΩΛ=0.6911, h=0.6774; hardcoded extra params
    σ8=0.8159, ns=0.9667, τ=0.0965 (marked `#CHECK ME` in code).
  - **+125 Mpc/h x-shift correction** between MTNG and MTNG-mimic coordinates is
    applied in cross-matching and `q_pos`.
- 7 varied subgrid params (Arepo/TNG): `WindEnergyIn1e51erg`, `VariableWindVelFactor`,
  `MaxSfrTimescale` (two values: min & max), `WindFreeTravelDensFac`,
  `RadioFeedbackFactor`, `BlackHoleFeedbackFactor`,
  `RadioFeedbackReiorientationFactor` (`RadioFeedbackFactor` dense+reorientation in
  log space where used; rho_rec & ef_kin are log10-transformed before training).

## Directory map

### `latin-hypercube/`
- `code.py`: builds the 30 Arepo parameter files from an LH design stored in
  `cpars_30_1997.h5` (30 points × 7 dims, seed 1997).
- `param_MTNG-hydro.txt`: template/fiducial Arepo param file;
  `param_MTNG-hydro_{0..29}.txt`: per-run files (only the 7 subgrid values change).

### `halo_selection/`
- `hydro_halo_sel_1pmbin.txt` / `dm_halo_sel_1pmbin.txt`: FoF indices of **452 halos**
  selected ~1 per mass bin from the hydro and DM MTNG runs (files have no
  trailing newline, so `wc -l` misleadingly reports 451).

### `src/` (core library)
- `utils.py` (~1700 lines): the main toolbox.
  - `class split_halos(sim)`: per-zoom measurements.
    - `halo_sel_setup` / `halo_sel`: sample halos in mass bins (with sampling
      fraction `h_frac` used to upweight the sparse zoom selection to the
      volume-complete MTNG statistics).
    - SMF helpers: `total_smf`, `halo_smf`, `halo_smf_draws`, `total_smf_30kpc`,
      `lite_mtng_smf_draws`; `get_mstar_30kpc[_general][_vec]` — stellar mass within
      30 **physical** kpc incl. Behroozi+19 Eddington-bias scatter
      σ = min(0.07+0.071z, 0.3).
    - Gas fractions: `halo_gas_frac_v2`, `lite_mtng_gas_frac`, `total_gas_frac` (vs M500c).
    - Scaling relations: `rhalf_m2half` (size–mass), `sSFR_mstar`, `smhm_ratio`,
      `bh_mstar`, `bh_mf`, `hmf`, `bias_sm` + `get_bpo` (bias via `bacco`).
    - `q_pos`: decode Lagrangian position from particle ID (MTNG offset handling).
  - Module-level functions:
    - `cross_match(zoom, snap)`: KD-tree (100 nn) position match zoom↔MTNG selection
      combined with mass/velocity similarity **metric** (function `metric`) to find
      counterpart halos; caches to `halo_selection/cross_match_<name>.npy`.
    - `large_halo_cross_match` (only M>1e13), `cross_match_zooms` (zoom↔zoom).
    - `camels_*`: SMF / fgas / sSFR / LH params for **CAMELS** comparison.
    - `get_parameters` / `pars`: read the 7 subgrid params from all 31 param files,
      standardize them (z-score) → design matrix for the emulator
      (WARNING: `get_parameters` references `self.zoom_base` at module scope —
      appears buggy if called standalone).
    - `get_zoom_smf`, `get_mtng_bhmf`, `read_central_xmatch`,
      `redshift_from_snap`, `read_cpu` (CPU time logs).
- `merger_tree_tools.py`: `class tree` — reads/walks SUBLINK-style merger trees
  (`treedata/trees.*.hdf5`) for MTNG or a zoom: inverse tree-link build, root info,
  progenitor walks (`get_all_progs`, main/secondary progenitor props), ex-situ
  stellar mass, density-bias & **IA-bias history** on grids. `mytest.py` mirrors
  `get_all_progs`. `merger_tree.c`, `read_tree.c`, `get_limits.c`,
  `test_data_type.c` are small C experiments/prototypes around the tree HDF5 files.
- `create_output_list.py`: builds `MTNG_OutputList_Selected.txt`
  (only snaps 264, 232, 199, 179, 151, 129, 94 written out).

### `scripts/`
- `cross_match.py`, `large_halo_cross_match.py`: batch cross-matching of all 31 zooms.
- `measure/`: measure SMF & fgas (`measure_SMF_fgas.py`), sSFR, SMHM, size–mass,
  BHMF, MBH–M*, high-z SMF, and group/cluster gas+density **profiles**
  (`compute_profiles.py`, uses `halotools`).
- `train/`: `GP_models.py` (gpytorch ExactGP, ConstantMean + ScaleKernel(RBF, ARD
  over 8 dims = 1 mass/radius coord + 7 params), manual `initialize()` with random
  restarts) and one `train_GP_*.py` per observable; models trained on the
  31 runs, best of 10 restarts × 1000 Adam steps, saved to `gp_train_results/`.
- `cross-validation/`: k-fold (k=3) CV of the SMF/fgas emulators (+ plots).
- `mcmc_chain.py`: **emcee** MCMC fitting the emulators to observations:
  SMF from GAMA/SDSS stitched GSMF; fgas from Popesso+2024 or Kugel et al.;
  "smf", "fgas" or "joint" modes, optional "BF" extended range; free nuisance
  params `b_star` (mass shift) and `b_cv` (cosmic-variance/normalization);
  sim scatter from the `*_draws100` files enters as likelihood noise.
- `uncertainty/`: repeated random subsampling draws in the full MTNG (hydro and
  DM-only `MTNG-L500-1080-A`) to quantify emulator/simulation uncertainty.

### `data/`
Observational inputs: GAMA/SDSS stitched GSMF, Bernardi+2018, Li & White 2009,
Moustakas, Wang+2024; GAMA size–mass (Pakmor); Graham & Sahu 2023 group catalog.

### `Notebooks/`
Exploratory + paper-analysis notebooks: SMF/fgas rebuilds (`08_10_fgas_rebuild`,
`27_08_SMF_rebuild`, `SMF_fgas_LHs`), GP emulation many variants (`GP_*.ipynb`),
`SMF_fgas_fits.ipynb` (chains & best fits), `halo_selection.ipynb`, `LH.ipynb`,
kSZ/tSZ 3D profiles of zooms, CAMELS comparisons, `uncertainty_quantification`.
Subfolders: `validation/` (cross-matching, contamination fraction, large-scale
shift checks), `best_fit/`, `plots/` (paper figures incl. MCMC corner chains),
`profiles/`, `1pt_functions/`, `smhm_relation/`, `spectrum/`, `performance/`.

### `bug_tests/`, `projeto_victor/`
- `bug_tests/bug_comparison.ipynb`: regression checks around a (resolved) bug.
- `projeto_victor/`: collaborator notebooks (read params; gas-fraction plots).

## Key facts / caveats for future work

- Many absolute paths are hardcoded to `/cosmos_storage/...` (both
  `cosmos_storage/home/fgmaion` and `cosmos_storage/simulations/...`); the code is
  NOT portable without editing paths. **Phase 4 will replace these with imports
  from `src/paths.py`** (see refactor progress below) — the caveats below are
  pre-refactor state.
- Cross-match & measurement results are cached under `results/`,
  `halo_selection/`, `cross-match/`, `gp_train_results/` outside the repo.
- Default analysis snapshot: 264 (z=0); SMF variants exist at higher z.
- The number of SMF bins used with the GP is 15 (Nbins_smf=15), fgas bins 10.
- `src/utils.py` has a `__pycache__` from an editable install-like usage;
  `src` is appended to `sys.path` in scripts (no packaging).

---

# Refactor progress (session log, 2026-08-14)

Branch: `refactor-portability` (from tag `pre-refactor` @ f5581f8 on `master`).
No commits made; working tree only. Verified dead code: `src/{get_limits,
read_tree,merger_tree,test_data_type}.c` + `src/mytest.py` (zero live refs;
removal planned for Phase 4).

## Done

- **Phase 0** — `tests/smoke_test.py` (tier 1: AST API snapshot vs
  `tests/api_baseline.json` — 45/20/3 symbols in utils/merger_tree_tools/paths;
  tier 2: runtime checks, skips where `bacco` not importable) + PASS.
  Env note: this machine lacks a bacco-capable python (venv `jupyter_gpu` has
  numpy 2.2.4 vs numba<=2.1 conflict) → runtime verification must happen on the
  user's analysis environment; do NOT touch that venv.
- **Phase 1** — `src/paths.py`: single config module. Inputs as env-var overridable
  strings (`MTNG_BASE`, `MTNG_DM_BASE`, `MTNG_RESIMS_BASE`, `MTNG_TREE_BASE`,
  `MTNG_MIMIC_BASE`, `MTNG_LH_DESIGN_FILE`); repo-relative outputs
  (`RESULTS_DIR`, `GP_MODELS_DIR`, `MCMC_CHAINS_DIR`, `CROSSMATCH_DIR`, env-
  overridable); helpers `zoom_param_file`, `zoom_output_dir`,
  `halo_selection_file`; constants `COSMO`, `SNAP_Z0=264`, `NPART=4320`,
  `BOXSIZE=500`, `X_SHIFT=125`, `HIGHZ_SNAPS=(264,232,199,179,151,129,94)`.
  Tested: helpers match legacy strings; env override works.

## Known issues to fix in later phases

- ~~`src/utils.py` WIP bugs~~ — RESOLVED: `split_halos.__init__` and
  `get_parameters` fixed (Phase 2/3); `get_zoom_smf` and
  `large_halo_cross_match` `self` bugs fixed (Phase 3).
- ~~Duplicates~~ — RESOLVED in Phase 3: `cross_match` family via
  `_cross_match_mtng_impl`; `get_mstar_30kpc` family via `_mstar_within`;
  `q_pos` unified (3 copies → 1 canonical).
- ~12 copy-pasted cosmology + `bacco.Simulation` boilerplate blocks in
  `scripts/` → Phase 4 replaces with `src/loading.py` helpers; script file
  names must stay (cluster jobs depend on them).

## Next step

Phase 5 — DONE (see Phase 5 log below). Next is Phase 6 = README rewrite
(paper/public release) + LICENSE (user to pick). After that, remaining
niceties: `halo_selection/` currently appears in .gitignore although the two
selection .txt files are tracked — clarify intent before public release.

---

# Phase 5 log (2026-08-24) — repo hygiene

## Done

- `.gitignore`: recursive `__pycache__/`, `*.py[cod]`, and
  `**/.ipynb_checkpoints/`; dropped redundant `/results/fgas` and
  `best_models_k10_nbins10/*` sub-entries; everything else unchanged.
  Verified: nothing tracked under results/gp_train_results/mcmc_chains/
  cross-match.
- Tooling: new venv `~/venvs/strip_tools` (py3.11) with nbstripout 0.9.1 —
  touches neither jupyter_gpu nor .bacco2_venv.
- Bulk-stripped rendered outputs from all 58 tracked notebooks:
  75.9 MB → 1.07 MB on disk. Verified per notebook: JSON parses, cell count
  identical to HEAD, zero remaining outputs; spot-diff confirmed changes are
  only outputs/execution_count/media blobs.
- Ongoing prevention: tracked `.gitattributes` (`*.ipynb filter=nbstripout`
  + zpln + diff driver) and `nbstripout --install` ran in this repo (local
  `.git/config` filter points at `~/venvs/strip_tools`). Contributors on
  another checkout still need to `nbstripout --install` locally (call this
  out in the Phase-6 README).
- Nothing committed by the agent; history untouched (no rewrites).

Note: after `--install`, `git diff` on notebooks passes through the clean
filter, so output-only changes will no longer show in diffs — that is the
point, but be aware when reviewing.

---

# Phase 4 log (2026-08-24) — scripts ported, dead files removed

Branch `refactor-portability`, no commits by the agent (working tree only).

## Done

- `loading.py`: +`halo_selection_weights(mtng, sel=None)` (the 7x-duplicated
  h_frac block) and +`load_mtng_lite(snap)` (uncertainty scripts' diluted
  variant). baseline regen records both.
- All 22 scripts ported onto paths/loading with a uniform relative
  bootstraps: `sys.path.insert(0, <repo-relative src>)` via `__file__`
  (scripts/ root = one `..`; subdirs = two). Script file NAMES unchanged.
  - measure/{S_SF...}: 8 files — loader calls replace the bacco.Simulation
    boilerplate, halo-sel unified to the repo copy (verified identical),
    h_frac via halo_selection_weights, results via paths.RESULTS_DIR.
    compute_profiles keeps its variants via overrides
    (use_ids=False, numpart=4320**3, tree_file). measure_smf_high_z keeps
    its tree_file/use_ids=False variant + treedata lookup now via
    paths.zoom_output_dir.
  - scripts/cross_match.py + large_halo_cross_match.py: by-name loading
    RESTORED (per user decision; NAMES list + MTNG_RUNS env override, e.g.
    MTNG_RUNS="LH_3,fiducial").
  - scripts/uncertainty/{measure_smf,measure_fgas}.py: dm=True + lite
    loaders; note the DM box load is kept though the object was never used
    downstream (faithful port).
  - scripts/train/*.py (9): 60-line param-parse + local pars/mean_std ->
    get_lh_design()/pars/mean_std_without_zeros; results & GP model dirs via
    paths; observables/error models/restarts per script preserved verbatim.
    `import utils` dropped where it became unused.
  - scripts/cross-validation (4): same treatment; GP_models import kept via
    relative scripts/train insert; model save/load path built from
    __file__ (gitignored best_models_k*_nbins* dirs as before).
  - scripts/mcmc_chain.py: design['mean'/'sigma'] feed param_means/
    param_stds/prepare_theta; GSMF via paths.REPO_ROOT/data/; models via
    GP_MODELS_DIR; chains via MCMC_CHAINS_DIR (+makedirs).
- Dead files REMOVED: src/{get_limits,read_tree,merger_tree,test_data_type}.c
  + src/mytest.py. Also removed stale cached bytecode incl. the *tracked*
  scripts/train/__pycache__/GP_models.cpython-39.pyc (shows as D in git;
  Phase 5 should ignore __pycache__ recursively).
- utils.py DISCOVERED-FIX: `split_halos.halo_smf_draws` still built
  `self.zoom_base+"/{name}/hydro_output/"` directly (legacy layout) — found
  via the e2e measure run, now uses paths.zoom_output_dir(name). This was
  the last hardcoded layout in src/.

## Verified

- `python -m py_compile` over all edited scripts: OK.
- grep: zero remaining /cosmos_storage|/lscratch|/scratch in scripts/
  (outside comments).
- sbatch e2e on this machine:
  - scripts/cross_match.py (MTNG_RUNS=fiducial): fresh compute ==
    ceph cache bit-for-bit (ind equal, d maxdiff 0.0).
  - scripts/measure/measure_SMF_fgas.py (bestfit_run): full pipeline ran;
    produced results/smf|fgas bestfit_run npy; values sane
    (SMF 1.4e-6..1.1e-2 /Mpc^3/dex; fgas 0.022..0.137).
  - tests/test_loading.py real-data layer incl. halo_selection_weights
    (452 fractions, all in [0,1]) — PASS (both load_mtng variants too).
- suites: smoke_test PASS py3.6 + venv; test_loading PASS py3.6 + venv.

## NOT verifiable here (flagged for next cluster/data-available run)

- Full LH_0..29 loops (zoom outputs absent on this machine).
- train/cv/mcmc execution (no results npy on this machine AND gpytorch/torch
  only in venv (torch yes, gpytorch NO) -> install gpytorch into
  .bacco2_venv or use another env before rerunning those).
- measure_smf_high_z / compute_profiles / uncertainty scripts: runnable
  where tree/lite/DM data are complete.

---

# Phase 3 log (2026-08-24) — utils.py internals

Branch `refactor-portability`, no commits (working tree only). Plan
decisions by user: read_central_xmatch stale → removed; q_pos unified on
safe flavor; cross_match_zooms dead → removed.

## Done

- `paths.py`: +`BIAS_DIR` (MTNG_BIAS_DIR), `MIMIC_DM_BASE`
  (MTNG_MIMIC_DM_BASE; genuine legacy name 'MTNG-mimic' on scratch),
  `CAMELS_BASE` (CAMELS_BASE env).
- `split_halos.__init__`: defaults paths.MTNG_BASE / paths.RESIMS_BASE,
  isdir as boolean + raise outside try (None+"" TypeError bug gone);
  `_FIDUCIAL_COSMO` now aliases `paths.COSMO`.
- `_load_sim_at_snap` → delegates to loading.load_zoom(use_ids=False,
  tree_file=...).
- `get_zoom_smf`: self.mtng_base bug + hardcoded halo-sel fixed via
  loading loaders (function was previously un-runnable).
- `cross_match`/`large_halo_cross_match` → shared private
  `_cross_match_mtng_impl` (frozen public signatures). large_'s
  `self.mtng_base` NameError gone. Cache dir unified to
  paths.CROSSMATCH_DIR (read+write; old read-from-halo_selection/,
  write-to-cross-match/ asymmetry dropped; `resims_base` still honored with
  both its subdirs when passed). Saves now makedirs(exist_ok=True).
- `cross_match_zooms`: REMOVED (always crashed: metric called with 8/9
  args; unused anywhere).
- `read_central_xmatch`: REMOVED (stale /lscratch/kwalsen path).
- `q_pos`: single canonical implementation in utils (deepcopy + %BoxSize
  wrap = old mtt flavor). mtt.q_pos delegates; split_halos.q_pos method
  delegates + keeps its corr_fac x-shift. Deliberate behavior change vs old
  utils.q_pos: input no longer mutated; mtng=True outputs wrapped.
- `get_mstar_30kpc` family → shared private `_mstar_within(...,
  aperture=)`; comoving (legacy get_mstar_30kpc) vs physical (_general)
  semantics frozen bit-identically (verified below).
- `pars`: dead first def removed (was broken at runtime); surviving
  pars(i, mass) delegates to loading.pars (no deprecation spam).
- `get_bpo`/`camels_get_LH_pars` hardcoded paths → paths.
- Numpy-style docstrings: all public symbols now documented (was missing:
  __init__, total_smf, rhalf_m2half, sSFR_mstar, smhm_ratio, bh_mstar,
  get_bpo, read_cpu, dict2d_sum, camels_sSFR, camels_get_LH_pars, metric,
  get_zoom_smf, get_mtng_bhmf, redshift_from_snap).

## Verification (all PASS)

- Parity harness (/tmp, seeded, record-before/check-after on fiducial zoom):
  bit-identical outputs for q_pos plain flavors, split_halos.q_pos method,
  get_mstar_30kpc{,_general,_vec} at z=0/1, get_mstar_2halfrad_vec;
  unified-safe cases verified against their intended new expectations.
- cross_match cache-read path (ceph resims_info/cross-match): identical.
- cross_match fresh recompute (sbatch, MTNG hydro + fiducial zoom): matches
  the DIPC-produced cache bit-for-bit (ind equal, d maxdiff 0.0).
- tests/smoke_test.py: PASS on py3.6 (tier-2 skip) and on venv py3.11
  (tier-2 real PASS). tests/test_loading.py: PASS both interpreters (real
  layer incl. zoom loads).
- tests/api_baseline.json regenerated: records the 2 removals + new private
  helpers (_cross_match_mtng_impl, _mstar_within) + unchanged signatures.

## Notes

- scripts/ still contain the copy-pasted loading boilerplate — Phase 4.
- gpytorch still missing from .bacco2_venv (needed for scripts/train).

---

# Phase 2 log (2026-08-24) — src/loading.py created

Branch `refactor-portability`, no commits (working tree only). Plan drafted
with user, then executed.

## Environment corrections (supersede Phase-0 env note)

- This machine DOES have a bacco-capable python: venv
  `/mnt/home/fmaion/packages/2_bacco/.bacco2_venv/` (py3.11.11, numpy 2.1.3,
  bacco editable from `~/packages/2_bacco/baccogit`, h5py/scipy/numba/emcee
  3.1.1/torch 2.6.0; gpytorch MISSING — needed only for GP training, Phase 4
  concern). The numpy-2.2.4/numba conflict applies to `~/venvs/jupyter_gpu`
  only; do NOT touch that venv. Canonical module set for SLURM:
  `~/packages/2_bacco/modules` (gcc/13.3.0, python/3.11.11, fftw/3.3.10,
  gsl/2.7.1, hdf5/1.12.3).
- Data on this machine: `/mnt/ceph/users/fmaion/projects/MTNG_resims/` with
  `param_LH/` (31 param files), `resims_info/halo_selection/` (both sel
  files), `simulations/{fiducial,bestfit_run}/hydro_output`,
  `simulations/MTNG/groups_264`, `simulations/MTNG-L500-2160-A`
  (groups_265+snapdir_265; z=0 snap = 265 there), `simulations/DM_only`.
  **LH_0..29 zoom outputs are NOT on this machine** — full 31-zoom
  load_all_zooms() verification stays deferred to a machine that has them.
- Zoom layout differs here: `<RESIMS_BASE>/simulations/<name>/hydro_output`
  (extra `simulations/` level vs legacy DIPC layout). Handled via new env
  var `MTNG_ZOOM_LAYOUT` (see below).

## Done

- `src/paths.py`: new env-overridable `ZOOM_OUTPUT_LAYOUT`
  (`MTNG_ZOOM_LAYOUT`, default legacy `{name}/hydro_output`).
- `src/loading.py` (new): `LH_NAMES`/`ALL_NAMES`, `PARAM_NAMES` (legacy
  ARD column order!), `get_lh_design()` → dict raw/design/mean/sigma
  (log10 on rho_rec+ef_kin before all statistics, ddof=0; errstate-guarded;
  cached), `pars(i, coord)`, `mean_std_without_zeros`, `load_zoom` /
  `load_all_zooms` (bacco imported lazily; **overrides for variants like
  compute_profiles' use_ids=False/numpart/tree_file), `load_mtng(dm=...)`
  (dm=True: gadget4_hdf5 on MTNG_DM_BASE, numpart overridable),
  `load_halo_selection()`.
- `src/utils.py`: `get_parameters()` is now a DeprecationWarning wrapper
  around loading.get_lh_design (fixes `self.zoom_base` NameError; identical
  return contract). No other utils fixes (Phase 3).
- `tests/smoke_test.py`: MODULES += loading.py; baseline regenerated
  (tests/ is git-untracked, so review via file content: only loading.py and
  known-intended entries).
  - Fixed a latent Phase-0 test bug: tier-2 `metric()` expected (2,) but
    metric is PAIRWISE (m1[...,None]/m2) → (2,2). It had never actually run
    before this machine got bacco.
- `tests/test_loading.py` (new): hermetic sandbox layer (py3.6-safe:
  stdlib+numpy only; also exercises utils wrapper when bacco present) +
  real-data layer (env-knob gated; big-box loads WARN not FAIL).

## Data-side maintenance performed (user should know)

- Deleted two **0-byte** `snaplist.txt` files —
  `simulations/fiducial/hydro_output/snaplist.txt` and
  `simulations/MTNG/snaplist.txt` — which made bacco raise "Empty
  snaplist.txt ... delete it so it gets regenerated". No information lost;
  bacco regenerates them on next load.

## Verified (2026-08-24)

- `python3 tests/smoke_test.py` (system py3.6.8): PASS (tier-2 skip).
- Under .bacco2_venv: `python tests/smoke_test.py` → PASS incl. REAL tier-2.
- `python tests/test_loading.py` hermetic: PASS on both interpreters.
- Real data (ceph): design finite 31×7 + direct text re-parse match; halo
  selection 452; utils wrapper identical to legacy columns;
  `load_zoom('fiducial')` → 139956 halos; `load_zoom('bestfit_run')` →
  139582 halos. Big-box loads via a small sbatch job (partition gen):
  `load_mtng(dm=False)` → 98233510 halos;
  `load_mtng(dm=True)` on MTNG-L500-2160-A (snap=265, numpart=2160**3,
  total_snapshots=266) → 11273943 halos.
  Note: boxes that keep only their final snapshot need
  `total_snapshots = <snap>+1` so bacco's snaplist scan reaches it
  (test layer passes it automatically for the DM check).

## Caveats

- `LH_0..29` zoom outputs absent here → `load_all_zooms()` on the full
  suite not yet exercised on real data (logic is name-loop only).
- gpytorch absent from .bacco2_venv — flag for Phase 4 (GP scripts).
