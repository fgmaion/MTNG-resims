import os
import numpy as np
import h5py
import functools
import bacco
import copy

try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle

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

    def __init__(self, snap_0=264, tree_format='MTNG', name=None, to_read=None, SNAP_INT=None):
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

        self.SNAP_INT = SNAP_INT

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
        self.sub_firstprog = np.fromfile("/cosmos_storage/home/fgmaion/prob-bias/MTNG/tree_data/first_progs.bin", dtype=np.int64)
        self.sub_nextprog = np.fromfile("/cosmos_storage/home/fgmaion/prob-bias/MTNG/tree_data/next_progs.bin", dtype=np.int64)


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

    def get_sub_file_props(self, recompute=False):
        '''
            This function is responsible for getting the file-pertinent information
            for the subhalos which we have walked up the tree.
        '''
        
        TREE_BASE = "/cosmos_storage/home/fgmaion/prob-bias/MTNG/tree_data/"
        if recompute==False:
            with open(TREE_BASE+"props_main_prog_10_SNAP_INT.p", 'rb') as fp:
                tree.sub_tree_prop = pickle.load(fp)
        else:
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
            #self.sub_tree_prop['SubhaloSpinType'] = np.zeros( (self.snap_0, len(self.sub_tree_index), 18) )
            self.sub_tree_prop['SubhaloIDMostbound'] =  np.zeros( (self.snap_0, len(self.sub_tree_index)), dtype=np.uint64 )
            self.sub_tree_prop['SubhaloPos'] =  np.zeros( (self.snap_0, len(self.sub_tree_index), 3) )
            self.sub_tree_prop['SubhaloIntertiaTensorStars'] =  np.zeros( (self.snap_0, len(self.sub_tree_index), 6) )
            self.sub_tree_prop['SubhaloRotationalEnergyStars'] =  np.zeros( (self.snap_0, len(self.sub_tree_index)) )
            self.sub_tree_prop['SubhaloSFR'] =  np.zeros( (self.snap_0, len(self.sub_tree_index)) )
            self.sub_tree_prop['SubhaloSfrInHalfRad'] =  np.zeros( (self.snap_0, len(self.sub_tree_index)) )

            # loading all data
            print("Starting to load data")
            for s in range(0, self.snap_0-self.SNAP_INT, self.SNAP_INT): 
                print("Getting the tree-relevant IDs and Indexes of subhalos in high-redshift snapshot")
                treeID, treeIndex, Nfile = self._sub_treeindex(snap=self.snap_0-s)

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
    #                    _spin_type[cumsub:cumsub+nsub,:]       = f['Subhalo']['SubhaloSpinType']
                        _id_mostbound[cumsub:cumsub+nsub]      = f['Subhalo']['SubhaloIDMostbound']
                        _pos[cumsub:cumsub+nsub,:]             = f['Subhalo']['SubhaloPos']
                        _intertia_tensor[cumsub:cumsub+nsub,:] = f['Subhalo']['SubhaloIntertiaTensorStars']
                        _rotational_energy[cumsub:cumsub+nsub] = f['Subhalo']['SubhaloRotationalEnergyStars']
                        _sfr[cumsub:cumsub+nsub]               = f['Subhalo']['SubhaloSFR']
                        _sfr_half_rad[cumsub:cumsub+nsub]      = f['Subhalo']['SubhaloSfrInHalfRad']

                    cumsub += nsub

                ################# DONE WITH THIS SNAPSHOT  ###################
                print("Done with snap {:d}".format(s))

                # Map values to their positions in treeIndex
                treeIndex = np.array(treeIndex)
                val_to_idx = {val: idx for idx, val in enumerate(treeIndex)}

                # Get the progenitor list
                fp_index = self.fp_idx[self.snap_0-s-1,:]

                # Create mask and array of positions in treeIndex
                mask = np.isin(fp_index, treeIndex)
                positions = np.array([val_to_idx[val] for val in fp_index[mask]])

                self.sub_tree_prop['SubhaloMass'][self.snap_0-s-1,mask] = _mass[positions]
                self.sub_tree_prop['SubhaloIsCen'][self.snap_0-s-1,mask] = _is_cen[positions]
                self.sub_tree_prop['SubhaloMassType'][self.snap_0-s-1,mask,...] = _mass_type[positions,...]
    #            self.sub_tree_prop['SubhaloSpinType'][self.snap_0-s-1,mask,...] = _spin_type[positions,...]
                self.sub_tree_prop['SubhaloIDMostbound'][self.snap_0-s-1,mask] = _id_mostbound[positions]
                self.sub_tree_prop['SubhaloPos'][self.snap_0-s-1,mask,...] = _pos[positions,...]
                self.sub_tree_prop['SubhaloIntertiaTensorStars'][self.snap_0-s-1,mask,...] = _intertia_tensor[positions,...]
                self.sub_tree_prop['SubhaloRotationalEnergyStars'][self.snap_0-s-1,mask] = _rotational_energy[positions]
                self.sub_tree_prop['SubhaloSFR'][self.snap_0-s-1,mask] = _sfr[positions]
                self.sub_tree_prop['SubhaloSfrInHalfRad'][self.snap_0-s-1,mask] = _sfr[positions]

            with open(TREE_BASE+"sec_prog_10_SNAP_INT.p", "wb") as fp: 
                pickle.dump(tree.sub_secprog_prop, fp, protocol=pickle.HIGHEST_PROTOCOL)


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

    def get_mp_secondary_progenitors(self):

        print("Reading tree")
        # Read the tree
        try:
            self.tree_offsets
            self.sub_mainprog
            self.sub_nextprog
            self.sub_firstprog
        except:
            self.read_tree_opt()

        # Get tree-relative ID and index of subhalos at Snap_0
        try:
            self.sub_tree_ID
            self.sub_tree_index
        except:
            self.get_sub_tree_props()

        # For loop over all the possible halos at a certain snapshot
        print("Starting the Loop")
        self.sec_prog = {self.snap_0 - 1 - self.SNAP_INT*i:{} for i in range(1, int(self.snap_0/self.SNAP_INT))} # This will then start at 253
        for ii in range(len(self.sub_tree_index)):
            tree_offset = self.tree_offsets[self.sub_tree_ID[ii]]
            print('Done for halo {:d}'.format(ii), end='\r')
            
            snap_i = self.snap_0
            roots = [ self.sub_tree_index[ii] ]
            mp = roots[0]
            while snap_i > 34: # at such high redshifts the code will break down due to lack of subhalos anyway
                new_roots = [] # this will store all the progenitors during the loop
                for i in range(len(roots)):
                    Np = self.sub_firstprog[roots[i] + tree_offset]
                    while Np != -1:
                        new_roots.append( np.int64(Np) )
                        Np = self.sub_nextprog[Np + tree_offset]

                roots = new_roots
                snap_i -= 1

                mp = self.sub_mainprog[mp + self.tree_offsets[self.sub_tree_ID[ii]]] # pass to the next main progenitor

                if ( (self.snap_0 - snap_i)%self.SNAP_INT==0 ):
                    self.sec_prog[snap_i-1][ii] = new_roots
                    roots = [ mp ] # reset the root as being just the main progenitor

        return None

    def get_secprog_props(self):
        '''
            This function is responsible for getting the file-pertinent information
            for the secondary progenitors of subhalos
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
        
        print("Walking Tree (Secondary Progenitors Included)")
        # Walk the tree
        try:
            self.sec_prog
        except:
            self.get_mp_secondary_progenitors(self.snap_0)
        print("Done Walking Tree (Secondary Progenitors Included)")

        self.sub_secprog_prop = {}
        self.sub_secprog_prop['SubhaloMass'] = {}
        self.sub_secprog_prop['SubhaloMassType'] = {}

        # loading all data
        print("Starting to load data")
        for s in range(self.SNAP_INT, self.snap_0-self.SNAP_INT, self.SNAP_INT): 
            print("Getting the tree-relevant IDs and Indexes of subhalos in snapshot")
            treeID, treeIndex, Nfile = self._sub_treeindex(snap=self.snap_0-s)

            # get all the interesting properties stored in the group-files, for all available snapshots
            ################ READING SINGLE-FILES ###################
            total_Nsub = int(0)
            for i in range(640):
                print("Reading file {:d} of group {:d}".format(i, self.snap_0-s), end='\r')
                with h5py.File(self.sim_base+'groups_{:03d}/fof_subhalo_tab_{:03d}.{:d}.hdf5'.format(self.snap_0-s,self.snap_0-s,i), 'r') as f:
                    total_Nsub += int(f['Header'].attrs['Nsubhalos_ThisFile'])

            _mass = np.empty(total_Nsub)
            _mass_type = np.empty((total_Nsub, 6))
            _pos = np.empty((total_Nsub, 3))

            cumsub = 0
            for i in range(640):
                print("Reading file {:d} of group {:d}".format(i, self.snap_0-s), end='\r')

                with h5py.File(self.sim_base+'groups_{:03d}/fof_subhalo_tab_{:03d}.{:d}.hdf5'.format(self.snap_0-s,self.snap_0-s,i), 'r') as f:
                    nsub = int(f['Header'].attrs['Nsubhalos_ThisFile'])

                    try:
                        f['Subhalo']['SubhaloMass']
                    except:
                        continue

                    _mass[cumsub:cumsub+nsub]              = f['Subhalo']['SubhaloMass']
                    _mass_type[cumsub:cumsub+nsub,:]       = f['Subhalo']['SubhaloMassType']
                    _pos[cumsub:cumsub+nsub,:]             = f['Subhalo']['SubhaloPos']

                cumsub += nsub
            ################# DONE WITH THIS SNAPSHOT  ###################
            print("Done with snap {:d}".format(s))

            self.sub_secprog_prop['SubhaloMass'][self.snap_0-s-1] = {}
            self.sub_secprog_prop['SubhaloMassType'][self.snap_0-s-1] = {}

            # Map values to their positions in treeIndex
            treeIndex = np.array(treeIndex)
            val_to_idx = {val: idx for idx, val in enumerate(treeIndex)}

            # Flatten the progenitor list
            sp_index = [np.array(self.sec_prog[self.snap_0-s-1][ii], dtype=int)  for ii in range(10)] #for ii in range(len(self.sub_tree_index))]
            sp_1d = np.concatenate(sp_index)

            # Create mask and array of positions in treeIndex
            mask = np.isin(sp_1d, treeIndex)
            positions = np.array([val_to_idx[val] for val in sp_1d[mask]])

            # Lengths of original arrays
            len_array = np.array([len(x) for x in sp_index])
            cut_idx = np.r_[0, len_array.cumsum()]

            # Count how many were kept in each chunk
            len_mask = np.add.reduceat(mask, cut_idx[:-1])
            cl_m = np.r_[0, len_mask.cumsum()]

            # Rebuild output with treeIndex positions instead of values
            self.out = [positions[i:j] for i, j in zip(cl_m[:-1], cl_m[1:])]

#            for ii in range(len(self.sub_tree_index)):
            for ii in range(10):
                self.sub_secprog_prop['SubhaloMass'][self.snap_0-s-1][ii] = _mass[self.out[ii]]
                self.sub_secprog_prop['SubhaloMassType'][self.snap_0-s-1][ii] = _mass_type[self.out[ii],...]

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
