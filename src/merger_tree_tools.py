import os
import numpy as np
import h5py
import functools
import bacco

class tree:

    def __init__(self, snap_0=264, treebase="/cosmos_storage/simulations/MTNG/", tree_format='MTNG'):
        self.snap_0 = snap_0
        self.treebase = treebase
        self.tree_format = tree_format

        self.mtng_snap0 = bacco.utils.load_MTNG(adr="/cosmos_storage/simulations/MTNG/", snap=self.snap_0)


    def get_tree_props(self, halo_index):

        """
        Retrieves properties of trees associated with the provided halo_index.

        Parameters
        ----------
        halo_index : array_like
            Indices of halos to retrieve tree properties for, relative to the group-file at snap_0.

        Returns
        -------
        mass : dict
            Dictionary containing mass of each tree corresponding to halo_index.
        firstprog : dict
            Dictionary containing first progenitor of each tree corresponding to halo_index.
        group_nr : dict
            Dictionary containing group number of each tree corresponding to halo_index.

        Notes
        -----
        This function accesses multiple hdf5 files and can be quite slow.

        """

        main_subs = self.mtng_snap0.fof['halo_firstsub'][halo_index]

        # get finelumbers and offsets for each of these subhalos
        ifile, offset = self.find_sub_filenum(main_subs)

        self.treeID = np.array([ h5py.File(self.treebase+"groups_{:03d}/subhalo_treelink_{:03d}.{:d}.hdf5".format(self.snap_0, self.snap_0, ifile[i]))['Subhalo']['TreeID'][main_subs[i]-offset[i]] for i in range(len(main_subs)) ])
        self.treeIndex = np.array([ h5py.File(self.treebase+"groups_{:03d}/subhalo_treelink_{:03d}.{:d}.hdf5".format(self.snap_0, self.snap_0, ifile[i]))['Subhalo']['TreeIndex'][main_subs[i]-offset[i]] for i in range(len(main_subs)) ])

        # find the files that contain these trees
        self.tree_arr = self.find_tree_filenum(self.treeID)

        offset = self.load_tree_prop(['TreeTable', 'StartOffset'], 639)
        tree_ids = self.load_tree_prop(['TreeTable', 'TreeID'], 639)
        length = self.load_tree_prop(['TreeTable', 'Length'], 639)
        self.prop_dic = {tree_ids[i]:{'Offset':offset[i], 'Len':length[i]} for i in range(len(tree_ids))}

        tree_file = self.find_tree_filenum(self.treeID)

        self.mass = {}
        self.firstprog = {}
        self.group_nr = {}

        for n in range(len(halo_index)):
            print("Done with halo {:d}".format(n), end='\r')
            nsubs1 = int(0)
            nsubs2 = int(0)
            i = int(0)

            off = self.prop_dic[self.treeID[n]]['Offset'] 
            lenn = self.prop_dic[self.treeID[n]]['Len']

            while i < (tree_file[self.treeID[n]][0]+1):
                with h5py.File("/cosmos_storage/simulations/MTNG/treedata/trees.{:d}.hdf5".format(i), 'r') as f:
                    if i < tree_file[self.treeID[n]][0]:
                        nsubs1 += int(f['Header'].attrs['Nhalos_ThisFile'])
                    nsubs2 += int(f['Header'].attrs['Nhalos_ThisFile'])
                i += 1

            if off+ lenn < nsubs2:
                with h5py.File("/cosmos_storage/simulations/MTNG/treedata/trees.{:d}.hdf5".format(tree_file[self.treeID[n]][0]), 'r') as f:
                    self.mass[n] = f['TreeHalos']['SubhaloMass'][(off - nsubs1):(off - nsubs1)+lenn]
                    self.firstprog[n] = f['TreeHalos']['TreeMainProgenitor'][(off - nsubs1):(off - nsubs1)+lenn]
                    self.group_nr[n] = f['TreeHalos']['GroupNr'][(off - nsubs1):(off - nsubs1)+lenn]
            else:
                print("Tree incomplete") # This happens when the tree is divided in 2 different files, does not indicate a problem
                with h5py.File("/cosmos_storage/simulations/MTNG/treedata/trees.{:d}.hdf5".format(tree_file[self.treeID[n]][0]), 'r') as f:
                    main_1 = f['TreeHalos']['SubhaloMass'][(off - nsubs1):]
                    main_2 = f['TreeHalos']['TreeMainProgenitor'][(off - nsubs1):]
                    main_3 = f['TreeHalos']['GroupNr'][(off - nsubs1):]

                with h5py.File("/cosmos_storage/simulations/MTNG/treedata/trees.{:d}.hdf5".format(tree_file[self.treeID[n]][0]+1), 'r') as f:
                    rest_1 = f['TreeHalos']['SubhaloMass'][:(self.prop_dic[self.treeID[n]]['Offset'] - nsubs2)]
                    rest_2 = f['TreeHalos']['TreeMainProgenitor'][:(self.prop_dic[self.treeID[n]]['Offset'] - nsubs2)]
                    rest_3 = f['TreeHalos']['GroupNr'][:(self.prop_dic[self.treeID[n]]['Offset'] - nsubs2)]

                self.mass[n] = np.concatenate((main_1, rest_1))
                self.firstprog[n] = np.concatenate((main_2, rest_2))
                self.group_nr[n] = np.concatenate((main_3, rest_3))

    def walk_tree(self):
        '''
            Walk the tree to get all progenitors of a given set of halos
        '''

        idx = np.zeros((len(self.treeIndex), self.snap_0+1), dtype=int)
        for ii in range(len(self.treeIndex)):

            idx_temp = []
            
            fp = self.treeIndex[ii]
            # walk tree until null pointer
            while fp!=-1:
                idx_temp.append( fp )
                fp = self.firstprog[ii][fp]

            idx[ii][:len(idx_temp)] = idx_temp
            idx[ii][len(idx_temp):] = -1 * np.ones(self.snap_0+1-len(idx_temp))
            
        return idx


    def find_tree_filenum(self, treeID):
        '''
            Given a set of tree-IDs, this function finds the treefiles which contain these trees. This is very relevant
            because loading the full tree is unfeasible
        '''
        _treeID = treeID

        # get number of files
        with h5py.File(self.treebase+"treedata/trees.0.hdf5") as f:
            numfiles = f['Header'].attrs['NumFiles']

        tree_filenum = {}
        for i in range(numfiles):
            with h5py.File(self.treebase+"treedata/trees.{:d}.hdf5".format(i)) as f:
                tree_ids = f['TreeTable']['TreeID'][...]
                for j, tid in enumerate(treeID):
                    if tid in tree_ids:
                        tree_filenum.setdefault(tid, []).append(i)

        return tree_filenum


    def find_sub_filenum(self, sub_index):
        '''
            This function finds in which of the group files each of the provided halos is in.

            sub_index -> Index of the subhalos we want, relative to the group-file at snap_0
        '''

        with h5py.File("/cosmos_storage/simulations/MTNG/groups_264/fof_subhalo_tab_{:03d}.0.hdf5".format(self.snap_0), "r") as f:
            numfiles = f['Header'].attrs['NumFiles']

        nsubs = np.array([h5py.File("/cosmos_storage/simulations/MTNG/groups_264/fof_subhalo_tab_{:03d}.{:d}.hdf5".format(self.snap_0,i), "r")['Header'].attrs['Nsubhalos_ThisFile'] for i in range(numfiles)], dtype=int)

        ifile = np.zeros(len(sub_index), dtype=int) # group-file number of each subhalo
        offset = np.zeros(len(sub_index), dtype=int) # subhalo offset corresponding to group-file
        for i in range(len(sub_index)):
            ifile[i] = np.where(np.cumsum(nsubs) < sub_index[i])[0][-1] + 1
            offset[i] = np.cumsum(nsubs)[ifile[i]-1]

        return ifile, offset

    def load_tree_prop(self, field, N):
        '''
        Load one particular property of the trees. This loads the properties in treefiles sequentially until the (N+1)th treefile
        Start the docum

        '''

        # Use list comprehension to directly read data
        data_list = [
            h5py.File(self.treebase + f"treedata/trees.{n}.hdf5", 'r')[field[0]][field[1]][...]
            for n in range(N+1)
        ]

        # Concatenate all arrays in the list
        data = np.concatenate(data_list, axis=0)

        return data

#    def get_sublinks():
