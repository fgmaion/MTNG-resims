#!/usr/bin/env python3
"""Tests for src/loading.py.

Two layers:

Layer 1 (hermetic, always runs; safe on any machine, no simulation data):
    builds a sandbox with 31 fake Arepo parameter files, pointed to via the
    MTNG_RESIMS_BASE env var, and checks param parsing, standardization, the
    pars() design builder, mean_std_without_zeros, the (repo-tracked) halo
    selection reader, and -- if bacco is importable -- consistency of the
    deprecated utils.get_parameters wrapper.

Layer 2 (real data, runs only when the environment points at real data):
    re-runs the design/selection checks against the real param_LH files and
    attempts the actual simulation loaders. Knobs (all optional):

      MTNG_RESIMS_BASE     root containing param_LH/           (required)
      MTNG_ZOOM_LAYOUT     zoom layout template, e.g.
                           'simulations/{name}/hydro_output'   (zoom loaders)
      MTNG_BASE            parent MTNG hydro base              (best-effort)
      MTNG_DM_BASE         DM counterpart base                 (dm=True test)
      MTNG_DM_TEST_SNAP    snapshot number of the DM run       (dm=True test)
      MTNG_DM_TEST_NUMPART particle count of the DM run        (dm=True test)

    Missing knobs skip the corresponding check; bacco failures on the
    big-box loads are reported as WARN (data subsets may be incomplete on
    some machines) rather than FAIL.

Usage:
    python tests/test_loading.py            # layer 1 + attempt layer 2
    python tests/test_loading.py --real-data   # layer 2 only (used internally)

Exit code 0 = pass/skips only; 1 = at least one hard failure.
"""

import os
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")

FAILURES = []


def check(cond, msg):
    """Record a test outcome."""
    print("  {}: {}".format("ok" if cond else "FAIL", msg))
    if not cond:
        FAILURES.append(msg)
    return cond


# --------------------------------------------------------------------------
# Layer 1: hermetic sandbox
# --------------------------------------------------------------------------

def _write_fake_param_file(path, i):
    """Deterministic parameter values for fake run i (0..30)."""
    keys = [
        ("WindEnergyIn1e51erg",              1.5  * (1.0 + 0.02 * i)),
        ("VariableWindVelFactor",            3.7  * (1.0 + 0.02 * i)),
        ("WindFreeTravelDensFac",            0.05 * (1.0 + 0.02 * i)),
        ("MaxSfrTimescale",                  0.002 * (1.0 + 0.02 * i)),
        ("RadioFeedbackFactor",              0.02 * (1.0 + 0.02 * i)),
        ("BlackHoleFeedbackFactor",          0.1  * (1.0 + 0.02 * i)),
        ("RadioFeedbackReiorientationFactor", 20.0 * (1.0 + 0.02 * i)),
    ]
    with open(path, "w") as f:
        f.write("Omega0    0.3089\n")
        f.write("\n")
        first = True
        for key, val in keys:
            if key == "WindFreeTravelDensFac" and first:
                # duplicated keyword: parser must keep the first value
                f.write("{}    {}\n".format(key, val / 2.0))
                first = False
            f.write("%s    %.17g\n" % (key, val))
        f.write("SomeOtherKeyword    42\n")


def layer1_hermetic():
    print("Layer 1 (hermetic sandbox):")

    tmp = tempfile.mkdtemp(prefix="mtng_loading_test_")
    param_dir = os.path.join(tmp, "param_LH")
    os.makedirs(param_dir)
    for i in range(30):
        _write_fake_param_file(
            os.path.join(param_dir, "param_MTNG-hydro_{:d}.txt".format(i)), i)
    _write_fake_param_file(
        os.path.join(param_dir, "param_MTNG-hydro.txt"), 30)
    _write_fake_param_file(
        os.path.join(param_dir, "param_MTNG-hydro_bf.txt"), 31)

    os.environ["MTNG_RESIMS_BASE"] = tmp
    sys.path.insert(0, SRC_DIR)
    import paths
    import loading

    try:
        # -- naming conventions
        check(loading.LH_NAMES[0] == "LH_0"
              and loading.LH_NAMES[-1] == "LH_29"
              and len(loading.LH_NAMES) == 30,
              "LH_NAMES convention")
        check(loading.ALL_NAMES[-1] == "fiducial"
              and len(loading.ALL_NAMES) == 31,
              "ALL_NAMES = LH_* + fiducial")

        # -- design: shapes, order, standardization, log10 params
        d = loading.get_lh_design()
        check(d["raw"].shape == (31, 7) and d["design"].shape == (31, 7),
              "get_lh_design shapes (31 x 7)")
        check(d["names"] == loading.ALL_NAMES, "design names order")
        check(d["param_names"] == ("wind_en", "wind_vel", "rho_rec",
                                   "sf_ts", "ef_kin", "ef_high", "f_re"),
              "legacy parameter column order")
        colmean = d["design"].mean(axis=0)
        colstd = d["design"].std(axis=0)
        check(abs(colmean).max() < 1e-10 and abs(colstd - 1).max() < 1e-10,
              "design columns are z-scored (mean 0, std 1)")
        check(abs(d["mean"] - d["raw"].mean(axis=0)).max() < 1e-12
              and abs(d["sigma"] - d["raw"].std(axis=0)).max() < 1e-12,
              "mean/sigma match raw reference")
        # duplicated WindFreeTravelDensFac keyword: first value must win
        import numpy as np
        expect = np.log10(0.05 * (1.0 + 0.02 * 7) / 2.0)
        check(abs(d["raw"][7, 2] - expect) < 1e-12,
              "log10 param + first-occurrence parsing (LH_7 rho_rec)")
        check(loading.get_lh_design() is d, "get_lh_design caching")
        d_bf = loading.get_lh_design(names=["bestfit_run"])
        check(d_bf["raw"].shape == (1, 7),
              "get_lh_design(names=['bestfit_run']) via _bf param file")

        # -- pars builder
        coord = np.linspace(9.0, 12.0, 5)
        arr = loading.pars(3, coord)
        check(arr.shape == (5, 8), "pars shape (N, 8)")
        check(abs(arr[:, 0] - coord).max() < 1e-12, "pars column 0 = coord")
        check(abs(arr[:, 1:] - d["design"][3]).max() < 1e-12,
              "pars columns 1..7 = design row of run i")

        # -- mean_std_without_zeros
        a = np.array([[0.0, 2.0], [4.0, 0.0], [8.0, 6.0]])
        m, s = loading.mean_std_without_zeros(a)
        check(abs(m[0] - 6.0) < 1e-12 and abs(s[0] - 2.0) < 1e-12
              and abs(m[1] - 4.0) < 1e-12 and abs(s[1] - 2.0) < 1e-12,
              "mean_std_without_zeros ignores zeros")

        # -- halo selection (repo-tracked file)
        sel = loading.load_halo_selection("hydro")
        check(sel.dtype.kind == "i" and len(sel) == 452,
              "halo selection: 452 integer halo indices")

        # -- deprecated utils wrapper (only where utils is importable)
        try:
            import warnings
            import utils
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                legacy = utils.get_parameters()
            check(any(w.category is DeprecationWarning for w in caught),
                  "utils.get_parameters emits DeprecationWarning")
            ok = len(legacy) == 7
            for j in range(7):
                ok = ok and abs(legacy[j] - d["design"][:, j]).max() < 1e-12
            check(ok, "utils.get_parameters == loading design columns")
        except ImportError as e:
            print("  skip: utils wrapper check (no bacco: {})".format(e))

        # -- informative ImportError when bacco is missing
        try:
            import bacco  # noqa: F401
            print("  skip: bacco ImportError check (bacco available here)")
        except ImportError:
            try:
                loading.load_zoom("LH_0")
                check(False, "load_zoom without bacco should raise")
            except ImportError as e:
                check("bacco is required" in str(e),
                      "load_zoom ImportError message is informative")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def layer1_layout_override():
    print("Layer 1b (zoom layout env override):")
    env = dict(os.environ)
    env["MTNG_ZOOM_LAYOUT"] = "simulations/{name}/hydro_output"
    code = (
        "import sys; sys.path.insert(0, {0!r});"
        "import paths;"
        "p = paths.zoom_output_dir('LH_0');"
        "assert p.endswith('simulations/LH_0/hydro_output'), p;"
        "q = paths.zoom_param_file('LH_0');"
        "assert q.endswith('param_LH/param_MTNG-hydro_0.txt'), q;"
        "print('child-ok')"
    ).format(SRC_DIR)
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         universal_newlines=True)
    check(out.returncode == 0 and "child-ok" in out.stdout,
          "MTNG_ZOOM_LAYOUT override honored by zoom_output_dir")


# --------------------------------------------------------------------------
# Layer 2: real data (all pieces optional; skip when not configured)
# --------------------------------------------------------------------------

def layer2_real_data():
    print("Layer 2 (real data):")
    resims = os.environ.get("MTNG_RESIMS_BASE")
    param0 = None
    if resims:
        param0 = os.path.join(resims, "param_LH", "param_MTNG-hydro_0.txt")
    if not param0 or not os.path.isfile(param0):
        print("  skip: no real MTNG_RESIMS_BASE with param_LH configured")
        return

    sys.path.insert(0, SRC_DIR)
    import numpy as np
    import loading

    d = loading.get_lh_design()
    check(d["raw"].shape == (31, 7) and np.all(np.isfinite(d["design"]))
          and np.all(d["sigma"] > 0),
          "real design: finite 31 x 7, positive sigmas")
    # independent re-parse of LH_7 straight from the text file
    vals = {}
    with open(param0.replace("_0.txt", "_7.txt")) as f:
        for line in f:
            cols = line.split()
            if len(cols) >= 2 and cols[0] == "WindFreeTravelDensFac":
                vals["rho_rec"] = float(cols[1])
                break
    check(abs(d["raw"][7, 2] - np.log10(vals["rho_rec"])) < 1e-12,
          "real design matches direct text parse (LH_7 rho_rec)")

    sel = loading.load_halo_selection("hydro")
    check(len(sel) == 452, "real halo selection: 452 halos")

    try:
        import bacco  # noqa: F401
    except ImportError as e:
        print("  skip: simulation loader checks (no bacco: {})".format(e))
        return

    # deprecated utils wrapper against the REAL design
    import warnings
    import utils
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        legacy = utils.get_parameters()
    ok = len(legacy) == 7
    for j in range(7):
        ok = ok and abs(legacy[j] - d["design"][:, j]).max() == 0.0
    check(ok, "real data: utils.get_parameters == loading design columns")

    layout = os.environ.get("MTNG_ZOOM_LAYOUT")
    for name in ("fiducial", "bestfit_run"):
        outdir = loading.paths.zoom_output_dir(name)
        if not layout or not os.path.isdir(outdir):
            print("  skip: load_zoom('{}') (no dir {})".format(name, outdir))
            continue
        z = loading.load_zoom(name)
        nhalo = len(z.fof["halo_pos"])
        check(nhalo > 0, "load_zoom('{}') loaded ({} halos)".format(name, nhalo))

    if os.environ.get("MTNG_BASE") and os.path.isdir(os.environ["MTNG_BASE"]):
        try:
            m = loading.load_mtng(dm=False)
            print("  ok  : load_mtng(dm=False) loaded ({} halos)"
                  .format(len(m.fof["halo_pos"])))
        except Exception as e:  # best-effort: data subsets may be partial
            print("  WARN: load_mtng(dm=False) failed: {}".format(e))
    else:
        print("  skip: load_mtng(dm=False) (MTNG_BASE not set/found)")

    dm_base = os.environ.get("MTNG_DM_BASE")
    dm_snap = os.environ.get("MTNG_DM_TEST_SNAP")
    dm_npart = os.environ.get("MTNG_DM_TEST_NUMPART")
    if dm_base and dm_snap and dm_npart and os.path.isdir(dm_base):
        try:
            mdm = loading.load_mtng(dm=True, snap=int(dm_snap),
                                    numpart=int(dm_npart),
                                    total_snapshots=int(dm_snap) + 1)
            print("  ok  : load_mtng(dm=True) loaded ({} halos)"
                  .format(len(mdm.fof["halo_pos"])))
        except Exception as e:
            print("  WARN: load_mtng(dm=True) failed: {}".format(e))
    else:
        print("  skip: load_mtng(dm=True) (set MTNG_DM_BASE / "
              "MTNG_DM_TEST_SNAP / MTNG_DM_TEST_NUMPART to enable)")


def main():
    print("Repo root: {}".format(REPO_ROOT))
    if "--real-data" in sys.argv:
        layer2_real_data()
    else:
        orig_env = dict(os.environ)
        layer1_hermetic()
        layer1_layout_override()
        env = orig_env
        code = subprocess.call(
            [sys.executable, os.path.abspath(__file__), "--real-data"],
            env=env)
        if code != 0:
            FAILURES.append("real-data layer exited with {}".format(code))
    if FAILURES:
        print("LOADING TEST: FAIL ({} failures)".format(len(FAILURES)))
        return 1
    print("LOADING TEST: PASS (non-checked pieces were skipped/warned above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
