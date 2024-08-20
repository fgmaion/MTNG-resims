import numpy as np
import h5py

alist = np.zeros(265)
for i in range(265):
    with h5py.File("/cosmos_storage/simulations/MTNG/single_files/file9/fof_subhalo_tab_{:03d}.9.hdf5".format(i), 'r') as f:
        alist[i] = f['Header'].attrs['Time']

with open("/lscratch/fgmaion/MTNG-resims/MTNG_OutputList_Selected.txt", 'w') as f:
    for i in range(265):
        if i in [264,232,199,179,151,129,94]:
            f.write(str(alist[i])+"  1\n")
        else:   
            f.write(str(alist[i])+"  0\n")

