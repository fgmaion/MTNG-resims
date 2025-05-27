import ctypes
import numpy as np

# Load the shared library
lib = ctypes.CDLL('./libmerger_tree.so')

# Define the function signature
lib.get_all_progs.argtypes = [
    ctypes.POINTER(ctypes.c_int),  # roots
    ctypes.c_int,                  # snap_0
    ctypes.c_int,                  # depth
    ctypes.POINTER(ctypes.c_int),  # firstprog
    ctypes.POINTER(ctypes.c_int),  # nextprog
]

def get_all_progs(roots, snap_0, depth, firstprog, nextprog):

    # Convert inputs to C-compatible arrays
    roots = np.ascontiguousarray(roots, dtype=np.int32)
    firstprog = np.ascontiguousarray(firstprog, dtype=np.int32)
    nextprog = np.ascontiguousarray(nextprog, dtype=np.int32)

    # Call the C function
    lib.get_all_progs(
        roots.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(snap_0),
        ctypes.c_int(depth),
        firstprog.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        nextprog.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )

    return None