import os
import numpy as np
import h5py
import functools
import bacco

def q_pos(mbID, npart=4320, BoxSize=500, mtng=False, idstart=0):
    
    import copy

    _mbID = copy.deepcopy(mbID)

    if mtng:
        _mbID[np.where(mbID>=1)] -= 20155392000
        _mbID[np.where(mbID<1)] += 80621568000

    q = np.zeros(_mbID.shape + (3,), dtype=np.float32)

    q[..., 0] = (_mbID - idstart) // npart**2
    q[..., 1] = ( (_mbID - idstart) // npart) % npart
    q[..., 2] = (_mbID - idstart) % npart

    # normalize correctly
    q *= (BoxSize / npart)
    q = q % BoxSize

    return q


class tree:

    def __init__(self, snap_0=264, tree_format='MTNG', name=None, to_read=None):
        self.snap_0 = snap_0
        self.tree_format = tree_format

        if tree_format == 'MTNG':
            self.sim_base = "/cosmos_storage/simulations/TNG_Family/MTNG/"
            self.sim_0 = bacco.utils.load_MTNG(adr="/cosmos_storage/simulations/TNG_Family/MTNG/", snap=self.snap_0)

        elif tree_format == 'zoom':
            self.sim_base = "/cosmos_storage/data_sharing/MN5_resims/"+name+"/hydro_output/"
            self.sim_0 = bacco.utils.load_zoom(snap=self.snap_0, name=name)


        self.to_read = to_read
        self.get_redshift()

    def get_redshift(self):

        with h5py.File(self.sim_base+"treedata/trees.{:d}.hdf5".format(0), 'r') as f:
            redshift = f['TreeTimes']['Redshift'][...]
        self.redshift = redshift[1:]
        self.expfactor = 1. / (1. + self.redshift)

    def get_sub_tree_props(self):
        '''
            This function returns
            the tree-relative ID and Index of the subhalos there contained.
            
        '''

        self.sub_tree_ID = []
        self.sub_tree_index = []
        self.ifile = []
        
        print( "Reading tree-relevant ID and index of subhalos in group {:d}".format(self.snap_0) )
        for file_number in range(640):
            print("Done with file {:d}".format(file_number), end="\r")
            treelink = self.sim_base+"groups_{0:03}/subhalo_treelink_{1:03}.{2:01}.hdf5".format(self.snap_0, self.snap_0, file_number)

            with h5py.File(treelink) as file:
                self.sub_tree_ID.extend( file['Subhalo']['TreeID'][...] )
                self.sub_tree_index.extend( file['Subhalo']['TreeIndex'][...] )
                self.ifile.extend( file_number * np.ones(len(file['Subhalo']['TreeID'][...]), dtype=int) )
        
        if self.to_read is None:
            self.sub_tree_index = np.array(self.sub_tree_index, dtype=np.int64)
            self.sub_tree_ID = np.array(self.sub_tree_ID, dtype=np.int64)
            self.ifile = np.array(self.ifile)

        else:
            self.sub_tree_index = np.array(self.sub_tree_index, dtype=np.int64)[self.to_read]
            self.sub_tree_ID = np.array(self.sub_tree_ID, dtype=np.int64)[self.to_read]
            self.ifile = np.array(self.ifile)[self.to_read]

    def read_tree(self):
        '''
            This function reads relevant information of the tree, including the progenitors 
            and tree offsets.
        '''

        #TODO: We should add a key, prog_type='Main', which should also take
        # other values that allow us to change the branch of the tree that
        # we wish to follow

        # Dictionary to hold the main progenitors for each tree
        sub_mainprog = np.empty(0, dtype=int) 
        tree_offsets = np.empty(0, dtype=int)

        global_max = np.max(self.sub_tree_ID)
        local_max = 0
        i = 0
        while (local_max <= global_max and i < 640):
            print("Now reading file {:d}".format(i), end='\r')
            with h5py.File(self.sim_base+"treedata/trees.{:d}.hdf5".format(i)) as f:
                tree_ids = f['TreeTable']['TreeID'][...]
                local_max = tree_ids[-1]

                tree_offsets = np.hstack( (tree_offsets, f['TreeTable']['StartOffset'][...] ) )
                sub_mainprog = np.hstack( (sub_mainprog, f['TreeHalos']['TreeMainProgenitor'][...] ) )

            i+=1

        self.tree_offsets = tree_offsets
        self.sub_mainprog = sub_mainprog
    def read_tree_opt(self):
        
        self.tree_offsets = np.fromfile("/cosmos_storage/home/fgmaion/prob-bias/MTNG/tree_data/offsets.bin", dtype=np.int64)
        self.sub_mainprog = np.fromfile("/cosmos_storage/home/fgmaion/prob-bias/MTNG/tree_data/main_progs.bin", dtype=np.int64)


    def walk_subs(self):
        '''
            This function takes the indices in self.sub_tree_index and walks
            them following the main progenitor.

            It uses information stored in self.tree_offsets, self.sub_tree_ID and self.sub_mainprog

        '''
        #TODO: We should add a key, prog_type='Main', which should also take
        # other values that allow us to change the branch of the tree that
        # we wish to follow. This key should also be added to the function read_tree above,
        # so that we read the correct type of progenitors.

        self.fp_idx = np.zeros((self.snap_0, len(self.sub_tree_index)), dtype=np.int64)
        
        print("Walking the Tree")

        tree_indices = self.sub_tree_index
        self.fp_idx[self.snap_0-1] = tree_indices + self.tree_offsets[self.sub_tree_ID] 

        for j in range(len(tree_indices)):
            
            fp = tree_indices[j]
            i = 1
            while fp != -1 and i < 264:
                fp = self.sub_mainprog[fp + self.tree_offsets[self.sub_tree_ID[j]]]
                self.fp_idx[self.snap_0 - i - 1,j] = fp + self.tree_offsets[self.sub_tree_ID[j]]
                i+=1
            
            self.fp_idx[:(self.snap_0 - i),j] = -1 * np.ones(self.snap_0 - i)

    def _sub_treeindex(self, snap):

        sub_tree_ID = []
        sub_tree_index = []
        ifile = []
        
        for file_number in range(640):
            treelink=self.sim_base+"groups_{0:03}/subhalo_treelink_{1:03}.{2:01}.hdf5".format(snap, snap, file_number)

            with h5py.File(treelink) as file:
                sub_tree_ID.extend( file['Subhalo']['TreeID'][...] )
                sub_tree_index.extend( file['Subhalo']['TreeIndex'][...] )
                ifile.extend( file_number * np.ones(len(file['Subhalo']['TreeID'][...]), dtype=int) )

        sub_tree_ID = np.array(sub_tree_ID, dtype=np.int64)
        sub_tree_index = np.array(sub_tree_index, dtype=np.int64)
        ifile = np.array(ifile, dtype=int)
                
        sub_tree_index = sub_tree_index[sub_tree_ID <= 6885892]
        ifile = ifile[sub_tree_ID <= 6885892]
        sub_tree_ID = sub_tree_ID[sub_tree_ID <= 6885892]

        sub_tree_index = sub_tree_index + self.tree_offsets[sub_tree_ID]

        return sub_tree_ID, sub_tree_index, ifile

    def get_sub_file_props(self):
        '''
            This function is responsible for getting the file-pertinent information
            for the subhalos which we have walked up the tree.
        '''
        
        # Get tree-relative ID and index of subhalos at Snap_0
        try:
            self.sub_tree_ID
            self.sub_tree_index
        except:
            self.get_sub_tree_props()

        print("Reading tree")
        # Read the tree
        try:
            self.tree_offsets
            self.sub_mainprog
        except:
            self.read_tree_opt()
   
        print("Done reading tree")
        
        print("Walking Tree")
        # Walk the tree
        try:
            self.fp_idx
        except:
            self.walk_subs()
        print("Done Walking Tree")

        self.sub_tree_prop = {}
        self.sub_tree_prop['SubhaloMass'] =  np.zeros( (self.snap_0, len(self.sub_tree_index)) )
        self.sub_tree_prop['SubhaloIsCen'] =  np.zeros( (self.snap_0, len(self.sub_tree_index)), dtype=int )
        self.sub_tree_prop['SubhaloMassType'] =  np.zeros( (self.snap_0, len(self.sub_tree_index), 6) )
        self.sub_tree_prop['SubhaloSpinType'] = np.zeros( (self.snap_0, len(self.sub_tree_index), 18) )
        self.sub_tree_prop['SubhaloIDMostbound'] =  np.zeros( (self.snap_0, len(self.sub_tree_index)), dtype=np.uint64 )
        self.sub_tree_prop['SubhaloPos'] =  np.zeros( (self.snap_0, len(self.sub_tree_index), 3) )
        self.sub_tree_prop['SubhaloIntertiaTensorStars'] =  np.zeros( (self.snap_0, len(self.sub_tree_index), 6) )
        self.sub_tree_prop['SubhaloRotationalEnergyStars'] =  np.zeros( (self.snap_0, len(self.sub_tree_index)) )
        self.sub_tree_prop['SubhaloSFR'] =  np.zeros( (self.snap_0, len(self.sub_tree_index)) )
        self.sub_tree_prop['SubhaloSfrInHalfRad'] =  np.zeros( (self.snap_0, len(self.sub_tree_index)) )

        # loading all data
        print("Starting to load data")
        for s in range(0, 260, 10): 
            print("Getting the tree-relevant IDs and Indexes of subhalos in high-redshift snapshot")
            treeID, treeIndex, Nfile = self._sub_treeindex(snap=self.snap_0-s)

            fp_index = self.fp_idx[self.snap_0-s-1,:]

            print("Intersecting indexes of walked subhalos and those at the high-redshift snapshot")
            _, fileCommon, treeCommon = np.intersect1d( treeIndex, fp_index, return_indices=True)
            # Now we find the positions of unique and non-unique elements of the tree-indices
            uni_els, uni_id, counts = np.unique(fp_index, return_index=True, return_counts=True)
            nuni_id = np.where(counts > 1)[0]
            
            print()
            # Now for each id in the non-unique list we enter a loop
            for i in range(len(nuni_id)):
                if uni_els[nuni_id[i]] == -1:
                    continue
                w_i = np.where(fp_index==uni_els[nuni_id[i]])[0]
                # Now we loop over all elements of w_i
                for j in range(1, len(w_i)):
                    try:
                        fileCommon = np.append( fileCommon, fileCommon[np.where(treeCommon==w_i[0])[0][0]] )
                    except:
                        continue
                    treeCommon = np.append( treeCommon, w_i[j] )


            # These guys not always match, sometimes treeCommon has
            print(treeCommon.shape, fp_index.shape)

            
            # get all the interesting properties stored in the group-files, for all available snapshots
            ################ READING SINGLE-FILES ###################
            total_Nsub = int(0)
            for i in range(640):
                print("Reading file {:d} of group {:d}".format(i, self.snap_0-s), end='\r')
                with h5py.File(self.sim_base+'groups_{:03d}/fof_subhalo_tab_{:03d}.{:d}.hdf5'.format(self.snap_0-s,self.snap_0-s,i), 'r') as f:
                    total_Nsub += int(f['Header'].attrs['Nsubhalos_ThisFile'])

            _mass = np.empty(total_Nsub)
            _is_cen = np.zeros(total_Nsub)
            _mass_type = np.empty((total_Nsub, 6))
            _spin_type = np.empty((total_Nsub, 18))
            _id_mostbound = np.empty(total_Nsub, dtype=np.uint64)
            _pos = np.empty((total_Nsub, 3))
            _intertia_tensor = np.empty((total_Nsub, 6))
            _rotational_energy = np.empty(total_Nsub)
            _sfr = np.empty(total_Nsub)
            _sfr_half_rad = np.empty(total_Nsub)

            cumsub = 0
            for i in range(640):
                print("Reading file {:d} of group {:d}".format(i, self.snap_0-s), end='\r')

                with h5py.File(self.sim_base+'groups_{:03d}/fof_subhalo_tab_{:03d}.{:d}.hdf5'.format(self.snap_0-s,self.snap_0-s,i), 'r') as f:
                    nsub = int(f['Header'].attrs['Nsubhalos_ThisFile'])

                    try:
                        f['Subhalo']['SubhaloMass']
                    except:
                        continue

                    _mass[cumsub:cumsub+nsub]                 = f['Subhalo']['SubhaloMass']
#                    sel                                       = f['Group']['GroupFirstSub'][...] < total_Nsub
                    _is_cen[f['Group']['GroupFirstSub']] = np.ones(len(f['Group']['GroupFirstSub']))
                    _mass_type[cumsub:cumsub+nsub,:]       = f['Subhalo']['SubhaloMassType']
                    _spin_type[cumsub:cumsub+nsub,:]       = f['Subhalo']['SubhaloSpinType']
                    _id_mostbound[cumsub:cumsub+nsub]      = f['Subhalo']['SubhaloIDMostbound']
                    _pos[cumsub:cumsub+nsub,:]             = f['Subhalo']['SubhaloPos']
                    _intertia_tensor[cumsub:cumsub+nsub,:] = f['Subhalo']['SubhaloIntertiaTensorStars']
                    _rotational_energy[cumsub:cumsub+nsub] = f['Subhalo']['SubhaloRotationalEnergyStars']
                    _sfr[cumsub:cumsub+nsub]               = f['Subhalo']['SubhaloSFR']
                    _sfr_half_rad[cumsub:cumsub+nsub]      = f['Subhalo']['SubhaloSfrInHalfRad']

                cumsub += nsub
            ################# DONE WITH THIS SNAPSHOT  ###################
            print("Done with snap {:d}".format(s))

            self.sub_tree_prop['SubhaloMass'][self.snap_0-s-1,treeCommon] = _mass[fileCommon]
            self.sub_tree_prop['SubhaloIsCen'][self.snap_0-s-1,treeCommon] = _is_cen[fileCommon]
            self.sub_tree_prop['SubhaloMassType'][self.snap_0-s-1,treeCommon,...] = _mass_type[fileCommon,...]
            self.sub_tree_prop['SubhaloSpinType'][self.snap_0-s-1,treeCommon,...] = _spin_type[fileCommon,...]
            self.sub_tree_prop['SubhaloIDMostbound'][self.snap_0-s-1,treeCommon] = _id_mostbound[fileCommon]
            self.sub_tree_prop['SubhaloPos'][self.snap_0-s-1,treeCommon,...] = _pos[fileCommon,...]
            self.sub_tree_prop['SubhaloIntertiaTensorStars'][self.snap_0-s-1,treeCommon,...] = _intertia_tensor[fileCommon,...]
            self.sub_tree_prop['SubhaloRotationalEnergyStars'][self.snap_0-s-1,treeCommon] = _rotational_energy[fileCommon]
            self.sub_tree_prop['SubhaloSFR'][self.snap_0-s-1,treeCommon] = _sfr[fileCommon]
            self.sub_tree_prop['SubhaloSfrInHalfRad'][self.snap_0-s-1,treeCommon] = _sfr[fileCommon]

    def get_d_bias_history(self, ngrid=192, damping_scale=0.1, recompute=False):
        '''
            In this function we wish to apply the probabilistic bias-estimators to the
            subhalos we have
        '''

        import bacco
        import bacco.probabilistic_bias as pb

        # lagrangian positions of the galaxies
        lag_pos = q_pos(self.sub_tree_prop['SubhaloIDMostbound'], mtng=True)

        # load the mtng-mimic simulation at very early stages
        dir_name_dm = "/cosmos_storage/simulations/TNG_Family/MTNG_mimic/output/"

        dm_mtng = bacco.Simulation(basedir=dir_name_dm, halo_file="groups_001/fof_subhalo_history_tab_orph_wweight_001", sim_format='gadget_hdf5',
                            ngenic_phases=True, phase_type=2, fixedPk=True)

        dm_mtng.header['Seed'] = 100672

        # These are the variables that need to be measured on a Lagrangian grid
        variables = ("J2", "J2=2", "J4", "J4=4", "J2=4")
        terms = ("J2", "J22", "J2=2")

        pbm = pb.ProbabilisticBiasManager(dm_mtng, variables=variables, damping_scale=damping_scale, ngrid=ngrid, verbose=2)
        D_model = pbm.setup_bias_model(pb.TensorBiasND, terms=terms, spatial_order=2)

        self.sub_tree_prop['d_bias'] =  np.zeros( (self.snap_0, len(self.sub_tree_index), 3) )

        for s in range(0, 260, 10):
            print("Doing Snapshot {:d}".format(self.snap_0-s))
            pbm.set_reference_expfactor( 1 / ( 1 + self.redshift[self.snap_0-s-1]) )

            tr_q, tr_value, tr_mask = pbm._define_tracers(tracer_q=lag_pos[self.snap_0-s-1,...])
            self.sub_tree_prop['d_bias'][self.snap_0-s-1,...] = D_model.bias_per_object(tr_value)

    def get_IA_bias_history(self, ngrid=192, damping_scale=0.1, recompute=False):
        '''
            In this function we wish to apply the probabilistic bias-estimators to the
            subhalos we have
        '''

        import bacco
        import bacco.probabilistic_bias as pb

        # lagrangian positions of the galaxies
        lag_pos = q_pos(self.sub_tree_prop['SubhaloIDMostbound'], mtng=True)

        # load the mtng-mimic simulation at very early stages
        dir_name_dm = "/cosmos_storage/simulations/TNG_Family/MTNG_mimic/output/"

        dm_mtng = bacco.Simulation(basedir=dir_name_dm, halo_file="groups_001/fof_subhalo_history_tab_orph_wweight_001", sim_format='gadget_hdf5',
                            ngenic_phases=True, phase_type=2, fixedPk=True)

        dm_mtng.header['Seed'] = 100672

        # These are the variables that need to be measured on a Lagrangian grid
        variables = ("Txx", "Txy", "Txz", "Tyy", "Tyz", "Tzz",)
        terms =  ("J2=2", "J22=2", "J2-2-2-")

        pbm = pb.ProbabilisticBiasManager(dm_mtng, variables=variables, damping_scale=damping_scale, ngrid=ngrid, verbose=2)
        IA_model = pbm.setup_bias_model(pb.IA_TensorBiasND, terms=terms, spatial_order=2)

        self.sub_tree_prop['IA_bias'] =  np.zeros( (self.snap_0, len(self.sub_tree_index), 3) )

        for s in range(0, 260, 10):
            print("Doing Snapshot {:d}".format(self.snap_0-s))
            pbm.set_reference_expfactor( 1 / ( 1 + self.redshift[self.snap_0-s-1]) )

            tr_q, tr_value, tr_mask = pbm._define_tracers(tracer_q=lag_pos[self.snap_0-s-1,...])
            shape_tensor = bacco.utils.I_to_S(self.sub_tree_prop['SubhaloIntertiaTensorStars'][self.snap_0-s-1,...])
            
            self.sub_tree_prop['IA_bias'][self.snap_0-s-1,...] = IA_model.bias_per_object(tr_value, I=shape_tensor)

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
