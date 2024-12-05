import os
import numpy as np
import h5py
import functools
import bacco

def q_pos(mbID, npart=4320, BoxSize=500, mtng=False, idstart=0):
    
    if mtng:
        mbID[np.where(mbID>=1)] -= 20155392000
        mbID[np.where(mbID<1)] += 80621568000

    q = np.zeros(mbID.shape + (3,), dtype=np.float32)

    q[..., 0] = (mbID - idstart) // npart**2
    q[..., 1] = ( (mbID - idstart) // npart) % npart
    q[..., 2] = (mbID - idstart) % npart

    # normalize correctly
    q *= (BoxSize / npart)

    return q


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
        while tree_max < np.max(self.file_tree_ID) + 1 and i < 640:
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
        self.fp_idx[self.snap_0-1] = tree_indices + self.tree_offsets[self.file_tree_ID]

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

            try:
                with h5py.File(treelink) as file:
                    file_tree_ID.extend( file['Subhalo']['TreeID'][...] )
                    file_tree_index.extend( file['Subhalo']['TreeIndex'][...] )
                    ifile.extend( file_number * np.ones(len(file['Subhalo']['TreeID'][...])) )
            except:
                continue
                
        file_tree_ID = np.array(file_tree_ID, dtype=int)
        ifile = np.array(ifile, dtype=int)

        tree_offsets = self.get_offsets(file_tree_ID)

        file_tree_index = np.array(file_tree_index + tree_offsets[file_tree_ID], dtype=int)

        return file_tree_ID, file_tree_index, ifile

    def get_sub_file_props(self, files=[0,1]):
    
        self.get_sub_tree_props(files=files)
        self.walk_subs()
        print("Done Walking Tree")

        self.sub_tree_prop = {}
        self.sub_tree_prop['SubhaloMass'] =  np.zeros( (self.snap_0, len(self.file_tree_index)) )
        self.sub_tree_prop['SubhaloIsCen'] =  np.zeros( (self.snap_0, len(self.file_tree_index)) )
        self.sub_tree_prop['SubhaloMassType'] =  np.zeros( (self.snap_0, len(self.file_tree_index), 6) )
        self.sub_tree_prop['SubhaloIDMostbound'] =  np.zeros( (self.snap_0, len(self.file_tree_index)), dtype=int )
        self.sub_tree_prop['SubhaloPos'] =  np.zeros( (self.snap_0, len(self.file_tree_index), 3) )
        self.sub_tree_prop['SubhaloIntertiaTensorStars'] =  np.zeros( (self.snap_0, len(self.file_tree_index), 6) )
        self.sub_tree_prop['SubhaloRotationalEnergyStars'] =  np.zeros( (self.snap_0, len(self.file_tree_index)) )
        self.sub_tree_prop['SubhaloSFR'] =  np.zeros( (self.snap_0, len(self.file_tree_index)) )

        # loading all data of the single-files
        print("Starting to load single-files")
        for s in range(self.snap_0-15): 
            # This factor of 15 is to avoid trying to load the very early snapshots 
            # -- those will need to climb the tree very high, and are thus costly, but to a minimal gain since their information is basically useless and for very few subhalos

            # get the tree-indexes of file-subhalos in higher-redshift files -- uses treelink files
            treeID, treeIndex, Nfile = self._sub_treeindex(snap=self.snap_0-s, files=files)

            # intersect these tree-indexes with those of the tree which we have walked
            _, fileCommon, treeCommon = np.intersect1d( treeIndex, self.fp_idx[self.snap_0-s-1,:], return_indices=True)

            # get all the interesting properties stored in the single-files, for all snapshots
            ################ READING SINGLE-FILES ###################
            total_Nsub = int(0)
            for i in files:
                file_i = np.load(self.sim_base+'single_files/file{:d}/reduced_fof_subhalo_tab_{:03d}.{:d}.npy'.format(i,self.snap_0-s,i), allow_pickle=True)[0]
                total_Nsub += int(file_i['Header']['Nsubhalos_ThisFile'])

            _mass = np.empty(total_Nsub)
            _is_cen = np.zeros(total_Nsub)
            _mass_type = np.empty((total_Nsub, 6))
            _id_mostbound = np.empty(total_Nsub, dtype=int)
            _pos = np.empty((total_Nsub, 3))
            _intertia_tensor = np.empty((total_Nsub, 6))
            _rotational_energy = np.empty(total_Nsub)
            _sfr = np.empty(total_Nsub)

            cumsub = 0
            for i in files:
                file_i = np.load(self.sim_base+'single_files/file{:d}/reduced_fof_subhalo_tab_{:03d}.{:d}.npy'.format(i,self.snap_0-s,i), allow_pickle=True)[0]
                nsub = int(file_i['Header']['Nsubhalos_ThisFile'])

                try:
                    file_i['Subhalo']['SubhaloMass']
                except:
                    continue

                _mass[cumsub:cumsub+nsub]              = file_i['Subhalo']['SubhaloMass']
                sel = file_i['Group']['GroupFirstSub'] < total_Nsub
                _is_cen[file_i['Group']['GroupFirstSub'][sel]] = np.ones(len(file_i['Group']['GroupFirstSub'][sel]))
                _mass_type[cumsub:cumsub+nsub,:]       = file_i['Subhalo']['SubhaloMassType']
                _id_mostbound[cumsub:cumsub+nsub]      = file_i['Subhalo']['SubhaloIDMostbound']
                _pos[cumsub:cumsub+nsub,:]             = file_i['Subhalo']['SubhaloPos']
                _intertia_tensor[cumsub:cumsub+nsub,:] = file_i['Subhalo']['SubhaloIntertiaTensorStars']
                _rotational_energy[cumsub:cumsub+nsub] = file_i['Subhalo']['SubhaloRotationalEnergyStars']
                _sfr[cumsub:cumsub+nsub]               = file_i['Subhalo']['SubhaloSFR']

                cumsub += nsub
            ################# DONE WITH SINGLE FILES ###################
            print("Done with snap {:d}".format(s), end='\r')

            self.sub_tree_prop['SubhaloMass'][self.snap_0-s-1,treeCommon] = _mass[fileCommon]
            self.sub_tree_prop['SubhaloIsCen'][self.snap_0-s-1,treeCommon] = _is_cen[fileCommon]
            self.sub_tree_prop['SubhaloMassType'][self.snap_0-s-1,treeCommon,...] = _mass_type[fileCommon,...]
            self.sub_tree_prop['SubhaloIDMostbound'][self.snap_0-s-1,treeCommon] = _id_mostbound[fileCommon]
            self.sub_tree_prop['SubhaloPos'][self.snap_0-s-1,treeCommon,...] = _pos[fileCommon,...]
            self.sub_tree_prop['SubhaloIntertiaTensorStars'][self.snap_0-s-1,treeCommon,...] = _intertia_tensor[fileCommon,...]
            self.sub_tree_prop['SubhaloRotationalEnergyStars'][self.snap_0-s-1,treeCommon] = _rotational_energy[fileCommon]
            self.sub_tree_prop['SubhaloSFR'][self.snap_0-s-1,treeCommon] = _sfr[fileCommon]

    def get_d_bias_history(self, ngrid=192, damping_scale=0.1, files=None, recompute=False):
        '''
            In this function we wish to apply the probabilistic bias-estimators to the
            subhalos we have
        '''

        import bacco
        import bacco.probabilistic_bias as pb

        # lagrangian positions of the galaxies
        lag_pos = q_pos(prop['mbID'][:,0], mtng=True, idstart=0)

        # MTNG Mimic
        dir_name_dm = "/cosmos_storage/simulations/TNG_Family/MTNG-mimic/output/"

        dm_mtng = bacco.Simulation(basedir=dir_name_dm, halo_file="groups_001/fof_subhalo_history_tab_orph_wweight_001", sim_format='gadget_hdf5',
                            ngenic_phases=True, phase_type=2, fixedPk=True)

        dm_mtng.header['Seed'] = 100672

        # These are the variables that need to be measured on a Lagrangian grid
        variables = ("J2", "J2=2", "J4", "J4=4", "J2=4")
        terms = ("J2", "J22", "J2=2")

        pbm = pb.ProbabilisticBiasManager(dm_mtng, variables=variables, damping_scale=damping_scale, ngrid=ngrid, verbose=2)
        D_model = pbm.setup_bias_model(pb.TensorBiasND, terms=terms, spatial_order=4)

        # total number of subhalos considering several files
        nsub = 0
        for fn in files:
            nsub += len(group_file_index[self.snap_0][fn])

        # set the arrays to receive properties
        b1_bpo = -1 * np.ones((nsub, self.depth, 3))

        for i in range(self.depth):
            pbm.set_reference_expfactor( 1 / ( 1 + self.redshift[i]))

            for fn in files:
                # Note if you pass the parameter  cachedir="path/to/some/empty/directory"
                # you may save some time, at the cost of storing some extra files

                tr_q, tr_value, tr_mask = pbm._define_tracers(tracer_q=lag_pos[group_tree_index[self.snap_0-i][fn]])
                b1_bpo[group_tree_index[self.snap_0-i][fn], self.depth-i-1] = D_model.bias_per_object(tr_value)

        return b1_bpo

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
