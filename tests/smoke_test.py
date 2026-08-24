#!/usr/bin/env python3
"""Smoke test guarding the MTNG-resims refactor.

Two tiers:

Tier 1 (always runs, stdlib only -- safe on any machine):
    AST-parses src/utils.py and src/merger_tree_tools.py, extracts the public
    API (functions, classes, methods and argument names) and compares it
    against tests/api_baseline.json. Use --write-baseline to (re)generate the
    baseline. Run --write-baseline ONCE before the refactor and commit the
    json; afterwards any missing/renamed/argument-changed symbol fails.

Tier 2 (runs only if 'bacco' is importable, e.g. on the analysis cluster):
    imports the two modules and exercise a few pure helper functions
    (metric, q_pos, build_lookup_index) that do not need simulation data.

Usage:
    python tests/smoke_test.py [--write-baseline] [-v]

Exit code 0 = pass (tier 2 may be skipped); 1 = failure.
"""

import ast
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "api_baseline.json")

MODULES = ("utils.py", "merger_tree_tools.py", "paths.py", "loading.py")

VERBOSE = "-v" in sys.argv


def _args_signature(node):
    """Return a normalized argument signature for a FunctionDef node."""
    args = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
    n_pos_defaults = len(node.args.defaults)
    if node.args.vararg is not None:
        args.append("*" + node.args.vararg.arg)
    if node.args.kwarg is not None:
        args.append("**" + node.args.kwarg.arg)
    return {"args": args, "n_pos_defaults": n_pos_defaults,
            "n_kwonly_defaults": len(node.args.kw_defaults)}


def extract_api(path):
    """Extract public API of a module from its AST.

    Returns
    -------
    dict
        {symbol_name: {"kind": "function"|"method"|"class", plus signature
        for callables}}. Methods appear under "<Class>.<method>". Dunder
        methods other than __init__ are ignored.
    """
    with open(path) as f:
        tree_ = ast.parse(f.read(), filename=path)

    api = {}
    for node in tree_.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            entry = {"kind": "function"}
            entry.update(_args_signature(node))
            api[node.name] = entry
        elif isinstance(node, ast.ClassDef):
            api[node.name] = {"kind": "class"}
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if sub.name.startswith("__") and sub.name != "__init__":
                        continue
                    entry = {"kind": "method"}
                    entry.update(_args_signature(sub))
                    api["{}.{}".format(node.name, sub.name)] = entry
    return api


def tier1(verbose=False):
    ok = True
    current = {}
    for mod in MODULES:
        path = os.path.join(SRC_DIR, mod)
        if not os.path.isfile(path):
            print("FAIL: missing {}".format(path))
            return False
        current[mod] = extract_api(path)
        if verbose:
            print("  [{}] {} symbols".format(mod, len(current[mod])))

    if "--write-baseline" in sys.argv:
        with open(BASELINE, "w") as f:
            json.dump(current, f, indent=1, sort_keys=True)
        print("Wrote baseline to {}".format(os.path.relpath(BASELINE)))
        return True

    if not os.path.isfile(BASELINE):
        print("FAIL: no baseline {} (run with --write-baseline first)"
              .format(os.path.relpath(BASELINE)))
        return False

    with open(BASELINE) as f:
        baseline = json.load(f)

    for mod in MODULES:
        base_syms, cur_syms = baseline.get(mod, {}), current[mod]
        for name in sorted(set(base_syms) - set(cur_syms)):
            print("REMOVED: {}:{}".format(mod, name))
            ok = False
        for name in sorted(set(cur_syms) & set(base_syms)):
            if base_syms[name] != cur_syms[name]:
                print("CHANGED: {}:{}\n  was: {}\n  now: {}"
                      .format(mod, name, base_syms[name], cur_syms[name]))
                ok = False
        for name in sorted(set(cur_syms) - set(base_syms)):
            print("added : {}:{} (ok)".format(mod, name))

    if ok:
        print("Tier 1 (API snapshot): PASS")

    # paths.py is stdlib-only: runtime check always possible.
    sys.path.insert(0, SRC_DIR)
    import paths
    assert paths.REPO_ROOT == REPO_ROOT, "REPO_ROOT mismatch"
    for name in ("LH_0", "LH_29", "fiducial", "bestfit_run"):
        p = paths.zoom_param_file(name)
        assert os.path.basename(p).startswith("param_MTNG-hydro"), \
            "zoom_param_file({!r}) -> {!r}".format(name, p)
    assert paths.zoom_output_dir("LH_0").endswith(
        os.path.join("LH_0", "hydro_output"))
    print("Tier 1 (paths runtime): PASS")
    return ok


def tier2():
    try:
        import bacco  # noqa: F401
    except ImportError:
        print("Tier 2 (runtime): SKIP ('bacco' not importable on this machine)")
        return True

    import numpy as np
    sys.path.insert(0, SRC_DIR)
    import utils
    import merger_tree_tools as mtt

    # utils.metric: pairwise distance-like metric matrix (n x m)
    d = utils.metric(np.array([1.0, 2.0]), np.array([1.1, 1.9]),
                     np.array([0.9, 0.9]), np.array([1.0, 1.1]),
                     np.array([0.01, 0.02]), 1, 1, 1, 1)
    d = np.asarray(d)
    assert d.shape == (2, 2) and np.all(np.isfinite(d)), \
        "metric() sanity failed"

    # utils.q_pos: Lagrangian positions from particle IDs
    q = utils.q_pos(np.array([1, 4320, 4320 ** 2]))
    assert q.shape == (3, 3) and np.all(q >= 0) and np.all(q < 500), \
        "q_pos() sanity failed"

    # merger_tree_tools.build_lookup_index
    arr = np.array([(1, 5, 9), (1, 5, 10)],
                   dtype=[("snap", "i4"), ("subhalo", "i4"), ("prog", "i4")])
    idx = mtt.build_lookup_index(arr)
    assert sorted(idx[1][5]) == [9, 10], "build_lookup_index() sanity failed"

    print("Tier 2 (runtime): PASS")
    return True


def main():
    write_mode = "--write-baseline" in sys.argv
    print("Repo root: {}".format(REPO_ROOT))
    ok = tier1(verbose=VERBOSE)
    if not write_mode:
        ok = tier2() and ok
    print("SMOKE TEST: {}".format("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
