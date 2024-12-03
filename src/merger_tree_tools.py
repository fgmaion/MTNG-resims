import os
import numpy as np
import h5py
import functools
import bacco

class tree:

    def __init__(self, snap_0=264, tree_format='MTNG', name=None):
        self.snap_0 = snap_0
        self.tree_format = tree_format

        if tree_format == 'MTNG':
            self.sim_base = "/cosmos_storage/simulations/TNG_Family/MTNG/"
            self.sim_0 = bacco.utils.load_MTNG(adr="/cosmos_storage/simulations/TNG_Family/MTNG/", snap=self.snap_0)

        elif tree_format == 'zoom':
            self.sim_base = "/cosmos_storage/data_sharing/MN5_resims/"+name+"/hydro_output/"
            self.sim_0 = bacco.utils.load_zoom(snap=self.snap_0, name=name)


    def get_tree_props(self, halo_index, N_tree=639):

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
        # Get indices of subhalos of interest
        main_subs = self.sim_0.fof['halo_firstsub'][halo_index]

        # Get finelumbers and offsets for each of these subhalos
        ifile, offset = self.find_sub_filenum(main_subs)

        # Get treeID for the subhalos we are interested in
        self.treeID = np.array([ h5py.File(self.sim_base+"groups_{:03d}/subhalo_treelink_{:03d}.{:d}.hdf5".format(self.snap_0, self.snap_0, ifile[i]))['Subhalo']['TreeID'][main_subs[i]-offset[i]] for i in range(len(main_subs)) ])
        # Get the index of these subhalos in the tree
        self.treeIndex = np.array([ h5py.File(self.sim_base+"groups_{:03d}/subhalo_treelink_{:03d}.{:d}.hdf5".format(self.snap_0, self.snap_0, ifile[i]))['Subhalo']['TreeIndex'][main_subs[i]-offset[i]] for i in range(len(main_subs)) ])

        offset = self.load_tree_prop(['TreeTable', 'StartOffset'], N_tree)
        tree_ids = self.load_tree_prop(['TreeTable', 'TreeID'], N_tree)
        length = self.load_tree_prop(['TreeTable', 'Length'], N_tree)
        self.prop_dic = {tree_ids[i]:{'Offset':offset[i], 'Len':length[i]} for i in range(len(tree_ids))}
        
        # Find the files that contain these trees
        tree_file = self.find_tree_filenum(self.treeID)

        self.mass = {}
        self.firstprog = {}
        self.group_nr = {}
        self.snapnum = {}
        self.nextprog = {}
        self.subhalo_nr = {}

        for n in range(len(halo_index)):
            print("Done with halo {:d}".format(n), end='\r')
            nsubs1 = int(0)
            nsubs2 = int(0)
            i = int(0)

            off = self.prop_dic[self.treeID[n]]['Offset'] 
            lenn = self.prop_dic[self.treeID[n]]['Len']

            while i < (tree_file[self.treeID[n]][0]+1):
                with h5py.File(self.sim_base+"treedata/trees.{:d}.hdf5".format(i), 'r') as f:
                    # number of subhalos of the tree in this file
                    if i < tree_file[self.treeID[n]][0]:
                        nsubs1 += int(f['Header'].attrs['Nhalos_ThisFile'])
                    # number of subhalos of the tree in next file
                    nsubs2 += int(f['Header'].attrs['Nhalos_ThisFile'])
                i += 1

            if off+lenn < nsubs2:
                with h5py.File(self.sim_base+"treedata/trees.{:d}.hdf5".format(tree_file[self.treeID[n]][0]), 'r') as f:
                    self.mass[n] = f['TreeHalos']['SubhaloMass'][(off - nsubs1):(off - nsubs1)+lenn]
                    self.firstprog[n] = f['TreeHalos']['TreeMainProgenitor'][(off - nsubs1):(off - nsubs1)+lenn]
                    self.nextprog[n] = f['TreeHalos']['TreeNextProgenitor'][(off - nsubs1):(off - nsubs1)+lenn]
                    self.group_nr[n] = f['TreeHalos']['GroupNr'][(off - nsubs1):(off - nsubs1)+lenn]
                    self.snapnum[n] = f['TreeHalos']['SnapNum'][(off - nsubs1):(off - nsubs1)+lenn]
                    self.subhalo_nr[n] = f['TreeHalos']['SubhaloNr'][(off - nsubs1):(off - nsubs1)+lenn]
            else:
                print("Tree incomplete") # This happens when the tree is divided in 2 different files, does not indicate a problem
                with h5py.File(self.sim_base+"treedata/trees.{:d}.hdf5".format(tree_file[self.treeID[n]][0]), 'r') as f:
                    main_1 = f['TreeHalos']['SubhaloMass'][(off - nsubs1):]
                    main_2 = f['TreeHalos']['TreeMainProgenitor'][(off - nsubs1):]
                    main_3 = f['TreeHalos']['GroupNr'][(off - nsubs1):]
                    main_4 = f['TreeHalos']['SnapNum'][(off - nsubs1):]
                    main_5 = f['TreeHalos']['TreeNextProgenitor'][(off - nsubs1):]
                    main_6 = f['TreeHalos']['SubhaloNr'][(off - nsubs1):]

                with h5py.File(self.sim_base+"treedata/trees.{:d}.hdf5".format(tree_file[self.treeID[n]][0]+1), 'r') as f:
                    rest_1 = f['TreeHalos']['SubhaloMass'][:(self.prop_dic[self.treeID[n]]['Offset'] - nsubs2)]
                    rest_2 = f['TreeHalos']['TreeMainProgenitor'][:(self.prop_dic[self.treeID[n]]['Offset'] - nsubs2)]
                    rest_3 = f['TreeHalos']['GroupNr'][:(self.prop_dic[self.treeID[n]]['Offset'] - nsubs2)]
                    rest_4 = f['TreeHalos']['SnapNum'][:(self.prop_dic[self.treeID[n]]['Offset'] - nsubs2)]
                    rest_5 = f['TreeHalos']['TreeNextProgenitor'][:(self.prop_dic[self.treeID[n]]['Offset'] - nsubs2)]
                    rest_6 = f['TreeHalos']['SubhaloNr'][:(self.prop_dic[self.treeID[n]]['Offset'] - nsubs2)]

                self.mass[n] = np.concatenate((main_1, rest_1))
                self.firstprog[n] = np.concatenate((main_2, rest_2))
                self.group_nr[n] = np.concatenate((main_3, rest_3))
                self.snapnum[n] = np.concatenate((main_4, rest_4))
                self.nextprog[n] = np.concatenate((main_5, rest_5))
                self.subhalo_nr[n] = np.concatenate((main_5, rest_5))

    def walk_tree(self):
        '''
            Walk the tree to get first progenitors of a given set of halos
        '''

        self.fp_idx = np.zeros((len(self.treeIndex), self.snap_0+1), dtype=int)
        for ii in range(len(self.treeIndex)):

            idx_temp = []
            
            fp = self.treeIndex[ii]
            # walk tree until null pointer
            while fp!=-1:
                idx_temp.append( fp )
                fp = self.firstprog[ii][fp]

            self.fp_idx[ii][:len(idx_temp)] = idx_temp
            self.fp_idx[ii][len(idx_temp):] = -1 * np.ones(self.snap_0+1-len(idx_temp))

    def get_sub_tree_props(self, files=[0,1]):

        self.file_tree_ID = []
        self.file_tree_index = []
        self.ifile = []

        for file_number in files:
            treelink=self.sim_base+"single_files/file{0:01}/subhalo_treelink_{1:03}.{2:01}.hdf5".format(file_number, self.snap_0, file_number)

            with h5py.File(treelink) as file:
                self.file_tree_ID.extend( file['Subhalo']['TreeID'][...] )
                self.file_tree_index.extend( file['Subhalo']['TreeIndex'][...] )
                self.ifile.extend( file_number * np.ones(len(file['Subhalo']['TreeID'][...])) )

        self.file_tree_index = np.array(self.file_tree_index)
        self.file_tree_ID = np.array(self.file_tree_ID)
        self.ifile = np.array(self.ifile)
        
        # Dictionary to hold the main progenitors for each tree
        sub_mainprog = np.empty(0, dtype=int) 
        tree_offsets = np.empty(0, dtype=int)

        tree_max = 0
        i = 0
        while tree_max < np.max(self.file_tree_ID) + 1:
            with h5py.File(self.sim_base+"treedata/trees.{:d}.hdf5".format(i)) as f:
                tree_ids = f['TreeTable']['TreeID'][...]
                tree_max = tree_ids[-1]

                tree_offsets = np.hstack( (tree_offsets, f['TreeTable']['StartOffset'][...] ) )
                sub_mainprog = np.hstack( (sub_mainprog, f['TreeHalos']['TreeMainProgenitor'][...] ) )
            i+=1

        self.tree_offsets = tree_offsets
        self.sub_mainprog = sub_mainprog

    def walk_subs(self):

        self.fp_idx = np.zeros((self.snap_0, len(self.file_tree_index)), dtype=int)
        
        print("Walking the Tree")

        tree_indices = self.file_tree_index
        self.fp_idx[self.snap_0-1] = tree_indices

        for j in range(len(tree_indices)):
            
            fp = tree_indices[j]
            i = 1
            while fp != -1:
                fp = self.sub_mainprog[fp + self.tree_offsets[self.file_tree_ID[j]]]
                self.fp_idx[self.snap_0 - i - 1,j] = fp + self.tree_offsets[self.file_tree_ID[j]]
                i+=1
            
            self.fp_idx[:(self.snap_0 - i),j] = -1 * np.ones(self.snap_0 - i)

    def get_offsets(self, treeID):

        tree_offsets = np.empty(0, dtype=int)

        tree_max = 0
        i = 0
        while tree_max < np.max(treeID) + 1:
            with h5py.File(self.sim_base+"treedata/trees.{:d}.hdf5".format(i)) as f:
                tree_ids = f['TreeTable']['TreeID'][...]
                tree_max = tree_ids[-1]

                tree_offsets = np.hstack( (tree_offsets, f['TreeTable']['StartOffset'][...] ) )
            i+=1

        return tree_offsets

    def _sub_treeindex(self, snap, files):

        file_tree_ID = []
        file_tree_index = []
        ifile = []
        
        for file_number in files:
            treelink=self.sim_base+"single_files/file{0:01}/subhalo_treelink_{1:03}.{2:01}.hdf5".format(file_number, snap, file_number)

            with h5py.File(treelink) as file:
                file_tree_ID.extend( file['Subhalo']['TreeID'][...] )
                file_tree_index.extend( file['Subhalo']['TreeIndex'][...] )
                ifile.extend( file_number * np.ones(len(file['Subhalo']['TreeID'][...])) )

        file_tree_ID = np.array(file_tree_ID, dtype=int)
        ifile = np.array(ifile, dtype=int)

        tree_offsets = self.get_offsets(file_tree_ID)

        file_tree_index = np.array(file_tree_index + tree_offsets[file_tree_ID], dtype=int)

        return file_tree_ID, file_tree_index, ifile

    # def get_sub_file_props(self, files=[0,1]):
    
    #     self.get_sub_tree_props(files=files)
    #     self.walk_subs()

    #     self.sub_mass = np.zeros((self.snap_0, len(self.file_tree_index)), dtype=int)


    #     for i in range(self.snap_0-1):

            
    #         treeID, treeIndex, Nfile = self._sub_treeindex(snap=self.snap_0-i-1, files=files)
    #         self.sub_mass[self.snap_0-i-1] =

        # Get the tree-indexes of the subhalos that are in higher-redshift snaps


    def get_all_progs(self, snap_0, depth):

        self.all_idx = {}
        for ii in range(len(self.treeIndex)):
            print('Done for halo {:d}'.format(ii), end='\r')
            self.all_idx[ii] = {}

            # starting at snap_0
            roots = [self.treeIndex[ii]]
            self.all_idx[ii][snap_0] = roots
            
            for i in range(depth):
                snap = snap_0 - (i+1)
                self.all_idx[ii][snap] = []
                
                for root in roots:
                    fp = self.firstprog[ii][root] # first prog of root
                    if fp!=-1:
                        self.all_idx[ii][snap].append(fp)
                        Np = self.nextprog[ii][fp]

                        while Np != -1:
                            self.all_idx[ii][snap].append(Np)
                            Np = self.nextprog[ii][Np]

                roots = self.all_idx[ii][snap]
                    

    def find_tree_filenum(self, treeID):
        '''
            Given a set of tree-IDs, this function finds the treefiles which contain these trees. This is very relevant
            because loading the full tree is unfeasible
        '''
        _treeID = treeID

        # get number of files
        with h5py.File(self.sim_base+"treedata/trees.0.hdf5") as f:
            numfiles = f['Header'].attrs['NumFiles']

        tree_filenum = {}
        for i in range(numfiles):
            with h5py.File(self.sim_base+"treedata/trees.{:d}.hdf5".format(i)) as f:
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

        with h5py.File(self.sim_base+"/groups_264/fof_subhalo_tab_{:03d}.0.hdf5".format(self.snap_0), "r") as f:
            numfiles = f['Header'].attrs['NumFiles']

        nsubs = np.array([h5py.File(self.sim_base+"/groups_264/fof_subhalo_tab_{:03d}.{:d}.hdf5".format(self.snap_0,i), "r")['Header'].attrs['Nsubhalos_ThisFile'] for i in range(numfiles)], dtype=int)

        ifile = np.zeros(len(sub_index), dtype=int) # group-file number of each subhalo
        offset = np.zeros(len(sub_index), dtype=int) # subhalo offset corresponding to group-file
        for i in range(len(sub_index)):
            ifile[i] = np.where(np.cumsum(nsubs) > sub_index[i])[0][0]
            if ifile[i] !=0:
                offset[i] = np.cumsum(nsubs)[ifile[i]-1]
            else:
                offset[i] = 0
        return ifile, offset

    def load_tree_prop(self, field, N):
        '''
        Load one particular property of the trees. This loads the properties in treefiles sequentially until the (N+1)th treefile
        Start the docum

        '''

        # Use list comprehension to directly read data
        data_list = [
            h5py.File(self.sim_base + f"treedata/trees.{n}.hdf5", 'r')[field[0]][field[1]][...]
            for n in range(N+1)
        ]

        # Concatenate all arrays in the list
        data = np.concatenate(data_list, axis=0)

        return data

#    def get_sublinks():
