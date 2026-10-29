'''
    Locations of the external data used by utils.py and merger_tree_tools.py.

    Every location can be overridden with an environment variable of the same
    name (e.g. `export MTNG_SIM=/data/MTNG/`); these are the same variables read
    by prob-bias/MTNG/src/data_paths.py. The defaults are the locations on the
    Flatiron cluster; anything not available there is left unset and raises a
    clear error when used.

    All directories end with a slash.
'''
import os

class _Unset:
    '''
        Placeholder for a location that is not available on this machine
    '''
    def __init__(self, name):
        self.name = name

    def _fail(self, *args, **kwargs):
        raise FileNotFoundError("{0} is not set on this machine: export {0}=/path/to/it".format(self.name))

    __add__ = __radd__ = __fspath__ = __str__ = __format__ = __truediv__ = _fail

    def __repr__(self):
        return "<unset {}>".format(self.name)

def _get(name, default=None):
    path = os.environ.get(name) or default
    if path is None:
        return _Unset(name)
    return os.path.join(path, '')

_FI = "/mnt/ceph/users/fmaion/"

RESIMS_DIR       = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '')
PROB_BIAS_MTNG   = _get("PROB_BIAS_MTNG", _FI + "fgmaion/prob-bias/MTNG/")
PROB_BIAS_BIASES = PROB_BIAS_MTNG + "biases/"
TREE_DATA        = _get("MTNG_TREE_DATA", PROB_BIAS_MTNG + "tree_data/")

MTNG_SIM       = _get("MTNG_SIM", _FI + "projects/MTNG_resims/simulations/MTNG/")  # only snap 264 + lite snaps here
MTNG_MIMIC_SIM = _get("MTNG_MIMIC_SIM")   # MTNG-mimic DM-only run (the "output/" directory)
MN5_RESIMS     = _get("MN5_RESIMS", _FI + "projects/MTNG_resims/LH_sims/")  # contains {name}/hydro_output/, param_LH/
CAMELS_DIR     = _get("CAMELS_DIR")       # CAMELS 1P/LH/CV catalogs
KWALSEN_XMATCH = _get("KWALSEN_XMATCH")   # Kai's xmatch/ directory (contains 264/<sim_name>/)
