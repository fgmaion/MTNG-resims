import os
import numpy as np
import h5py
import functools
import bacco
import copy
import gc

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

from collections import defaultdict

def build_lookup_index(struct_array, name1='snap', name2='subhalo', name3='prog', name4=None):
    """
    Build a fast (name1, name2) -> list-of-name3 lookup dictionary
    from a structured NumPy array.

    Parameters
    ----------
    struct_array : np.ndarray
        Structured array with fields 'name1', 'name2', and 'name3'.

    Returns
    -------
    dict
        Nested dict of form index[name1][name2] = list of name3 values
    """
    if name4 is None:
        index = defaultdict(lambda: defaultdict(list))
        for row in struct_array:
            index[row[name1]][row[name2]].append(row[name3])
    else:
        print("Entering the 4th name")
        index = defaultdict(lambda: defaultdict( lambda: defaultdict(list)))
        counter=0
        for row in struct_array:
            if counter%1000000==0:
                print("Done with {:d} rows".format(counter))
            index[row[name1]][row[name2]][row[name3]].append(row[name4])
            counter+=1
    return index

class tree:

    def __init__(self, snap_0=264, tree_format='MTNG', name=None, to_read=None, SNAP_INT=None):
        self.TREE_BASE = "/cosmos_storage/home/fgmaion/prob-bias/MTNG/tree_data/"

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
        self.min_snap = 44

    def get_redshift(self):

        with h5py.File(self.sim_base+"treedata/trees.{:d}.hdf5".format(0), 'r') as f:
            redshift = f['TreeTimes']['Redshift'][...]
        self.redshift = redshift[1:]
        self.expfactor = 1. / (1. + self.redshift)

    def get_root_info(self, recompute=True, save=True):
        '''
            This function reads the tree-relevant ID and index of subhalos in group snap_0.
            It stores the information in self.root_tree_ID, self.root_index and self.ifile.
            If to_read is not None, it will only read the indices specified in to_read.            
        '''

        self.root_tree_ID = []
        self.root_index = []
        self.ifile = []
        
        if recompute==False:
            print("Function get_root_info is being run with recompute=False\n")
            print("Now loading the saved file of subhalo tree-relevant IDs and indices\n")
            with open(self.TREE_BASE+"root_info_{:d}.p".format(self.snap_0), 'rb') as fp:
                self.root_tree_ID, self.root_index, self.ifile = pickle.load(fp)
            print("Done\n")
            return None
        else:
            print( "Reading tree-relevant ID and index of subhalos in group {:d}".format(self.snap_0) )
            for file_number in range(640):
                print("Done with file {:d}".format(file_number), end="\r")
                treelink = self.sim_base+"groups_{0:03}/subhalo_treelink_{1:03}.{2:01}.hdf5".format(self.snap_0, self.snap_0, file_number)

                with h5py.File(treelink) as file:
                    self.root_tree_ID.extend( file['Subhalo']['TreeID'][...] )
                    self.root_index.extend( file['Subhalo']['TreeIndex'][...] )
                    self.ifile.extend( file_number * np.ones(len(file['Subhalo']['TreeID'][...]), dtype=int) )
            
            if self.to_read is None:
                self.root_index = np.array(self.root_index, dtype=np.int64)
                self.root_tree_ID = np.array(self.root_tree_ID, dtype=np.int64)
                self.ifile = np.array(self.ifile)

            else:
                self.root_index = np.array(self.root_index, dtype=np.int64)[self.to_read]
                self.root_tree_ID = np.array(self.root_tree_ID, dtype=np.int64)[self.to_read]
                self.ifile = np.array(self.ifile)[self.to_read]
            print("Done reading tree-relevant IDs and indices of subhalos in group {:d}\n".format(self.snap_0))

        if save:
            print("Now saving the subhalo tree-relevant IDs and indices in "+self.TREE_BASE+"root_info_{:d}.p\n".format(self.snap_0))
            with open(self.TREE_BASE+"root_info_{:d}.p".format(self.snap_0), "wb") as fp: 
                pickle.dump((self.root_tree_ID, self.root_index, self.ifile), fp, protocol=pickle.HIGHEST_PROTOCOL)
            print("Done!\n")

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

        global_max = np.max(self.root_tree_ID)
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
        print("Done with offsets")
        self.sub_mainprog = np.fromfile("/cosmos_storage/home/fgmaion/prob-bias/MTNG/tree_data/main_progs.bin", dtype=np.int64)
        print("Done with Mainprog")
        self.sub_firstprog = np.fromfile("/cosmos_storage/home/fgmaion/prob-bias/MTNG/tree_data/first_progs.bin", dtype=np.int64)
        print("Done with FirstProg")
        self.sub_nextprog = np.fromfile("/cosmos_storage/home/fgmaion/prob-bias/MTNG/tree_data/next_progs.bin", dtype=np.int64)
        print("Done with SecondaryProg")

    def clean_tree(self):
        '''
           This function deletes some properties of the class that are no longer
           being used and take large portions of RAM
        '''

        try:
            del self.sub_mainprog
        except:
            print("sub_mainprog does not exist, skipping deletion")
        try:
            del self.sub_firstprog
        except:
            print("sub_firstprog does not exist, skipping deletion")
        try:
            del self.sub_nextprog
        except:
            print("sub_nextprog does not exist, skipping deletion")

        gc.collect()

        print("Tree cleaned!\n")

    def walk_subs(self, recompute=False, save=False):
        '''
            This function takes the indices in self.root_index and walks
            them following the main progenitor.

            It uses information stored in self.tree_offsets, self.root_tree_ID and self.sub_mainprog

        '''
        #TODO: We should add a key, prog_type='Main', which should also take
        # other values that allow us to change the branch of the tree that
        # we wish to follow. This key should also be added to the function read_tree above,
        # so that we read the correct type of progenitors.

        if recompute==False:
            print("Function walk_subs is being run with recompute=False\n")
            print("Now loading the saved file of main progenitors \n")
            with open(self.TREE_BASE+"main_progs.p", 'rb') as fp:
                self.mp_idx = pickle.load(fp)
            print("Done\n")
        else:
            try:
                self.sub_mainprog
                self.tree_offsets
            except:
                print("sub_mainprog does not exist, reading tree")
                self.read_tree_opt()
                
            self.mp_idx = np.zeros((self.snap_0, len(self.root_index)), dtype=np.int64)
            
            print("Walking the Tree")

            tree_indices = self.root_index
            self.mp_idx[self.snap_0-1] = tree_indices + self.tree_offsets[self.root_tree_ID]

            for j in range(len(tree_indices)):
                
                fp = tree_indices[j]
                i = 1
                while fp != -1 and i < 264:
                    fp = self.sub_mainprog[fp + self.tree_offsets[self.root_tree_ID[j]]]
                    self.mp_idx[self.snap_0 - i - 1,j] = fp + self.tree_offsets[self.root_tree_ID[j]]
                    i+=1
                
                self.mp_idx[:(self.snap_0 - i),j] = -1 * np.ones(self.snap_0 - i)

            print("Done walking the tree")
            self.clean_tree()

            if save:
                print("Now saving the main progenitors in "+self.TREE_BASE+"main_progs.p\n")
                with open(self.TREE_BASE+"main_progs.p", "wb") as fp: 
                    pickle.dump(self.mp_idx, fp, protocol=pickle.HIGHEST_PROTOCOL)
                print("Done!\n")

    def _sub_treeindex(self, snap):

        root_tree_ID = []
        root_index = []
        ifile = []
        
        for file_number in range(640):
            print("Loading file number ${:d}\n".format(file_number))
            treelink=self.sim_base+"groups_{0:03}/subhalo_treelink_{1:03}.{2:01}.hdf5".format(snap, snap, file_number)

            with h5py.File(treelink) as file:
                root_tree_ID.extend( file['Subhalo']['TreeID'][...] )
                root_index.extend( file['Subhalo']['TreeIndex'][...] )
                ifile.extend( file_number * np.ones(len(file['Subhalo']['TreeID'][...]), dtype=int) )

        root_tree_ID = np.array(root_tree_ID, dtype=np.int64)
        root_index = np.array(root_index, dtype=np.int64)
        ifile = np.array(ifile, dtype=int)
                
        root_index = root_index[root_tree_ID <= 6885892]
        ifile = ifile[root_tree_ID <= 6885892]
        root_tree_ID = root_tree_ID[root_tree_ID <= 6885892]

        root_index = root_index + self.tree_offsets[root_tree_ID]

        return root_tree_ID, root_index, ifile

    def get_sub_file_props(self, recompute=False, save=False):
        '''
            This function is responsible for getting the file-pertinent information
            for the subhalos which we have walked up the tree.
        '''
        
        if recompute==False:
            print("Function get_sub_file_props is being run with recompute=False\n")
            print("Now loading the saved file of secondary progenitor properties\n")
            with open(self.TREE_BASE+"props_main_prog_10_SNAP_INT.p", 'rb') as fp:
                self.sub_tree_prop = pickle.load(fp)
            print("Done\n")
        else:
            # Get tree-relative ID and index of subhalos at Snap_0
            try:
                self.root_tree_ID
                self.root_index
            except:
                self.get_root_info()

    
            print("Done reading tree")
            
            print("Walking Tree")
            # Walk the tree
            try:
                self.mp_idx
            except:
                self.walk_subs()
            print("Done Walking Tree")

            self.sub_tree_prop = {}
            self.sub_tree_prop['SubhaloMass'] =  np.zeros( (self.snap_0, len(self.root_index)) )
            self.sub_tree_prop['SubhaloIsCen'] =  np.zeros( (self.snap_0, len(self.root_index)), dtype=int )
            self.sub_tree_prop['SubhaloMassType'] =  np.zeros( (self.snap_0, len(self.root_index), 6) )
            #self.sub_tree_prop['SubhaloSpinType'] = np.zeros( (self.snap_0, len(self.root_index), 18) )
            self.sub_tree_prop['SubhaloIDMostbound'] =  np.zeros( (self.snap_0, len(self.root_index)), dtype=np.uint64 )
            self.sub_tree_prop['SubhaloPos'] =  np.zeros( (self.snap_0, len(self.root_index), 3) )
            self.sub_tree_prop['SubhaloIntertiaTensorStars'] =  np.zeros( (self.snap_0, len(self.root_index), 6) )
            self.sub_tree_prop['SubhaloRotationalEnergyStars'] =  np.zeros( (self.snap_0, len(self.root_index)) )
            self.sub_tree_prop['SubhaloSFR'] =  np.zeros( (self.snap_0, len(self.root_index)) )
            self.sub_tree_prop['SubhaloSfrInHalfRad'] =  np.zeros( (self.snap_0, len(self.root_index)) )

            # loading all data
            print("Starting to load data")
            for s in range(0, self.snap_0-self.min_snap+1, self.SNAP_INT): 
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
                
                try:
                    # Map values to their positions in treeIndex
                    treeIndex = np.array(treeIndex)
                    val_to_idx = {val: idx for idx, val in enumerate(treeIndex)}

                    # Get the progenitor list
                    fp_index = self.mp_idx[self.snap_0-s-1,:]

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
                
                except:
                    print("FAILED for snap {:d}\n".format(self.snap_0-s))

                print("Done with snap {:d}".format(s))

            if save:
                with open(self.TREE_BASE+"props_main_prog_10_SNAP_INT.p", "wb") as fp: 
                    pickle.dump(self.sub_tree_prop, fp, protocol=pickle.HIGHEST_PROTOCOL)


    def get_d_bias_history(self, ngrid=192, damping_scale=0.1, recompute=False, save=True):
        '''
            In this function we wish to apply the probabilistic bias-estimators to the
            subhalos we have
        '''

        import bacco
        import bacco.probabilistic_bias as pb

        if recompute==False:
            with open(self.TREE_BASE+"props_main_prog_10_SNAP_INT.p", 'rb') as fp:
                self.sub_tree_prop = pickle.load(fp)
            try:
                self.sub_tree_prop['d_bias']
            except:
                print("Code was run with recompute=FALSE, yet density biases have not been computed\n")
        else:
            try:
                self.root_tree_ID
                self.root_index
            except:
                self.get_root_info()

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

            self.sub_tree_prop['d_bias'] =  np.zeros( (self.snap_0, len(self.root_index), 3) )

            for s in range(0, self.snap_0, self.SNAP_INT):
                try:
                    print("Doing Snapshot {:d}\n".format(self.snap_0-s))
                    pbm.set_reference_expfactor( 1 / ( 1 + self.redshift[self.snap_0-s-1]) )

                    tr_q, tr_value, tr_mask = pbm._define_tracers(tracer_q=lag_pos[self.snap_0-s-1,...])
                    self.sub_tree_prop['d_bias'][self.snap_0-s-1,...] = D_model.bias_per_object(tr_value)
                except:
                    print("Failed for SNAP {:d}\n".format(self.snap_0-s))

            if save:
                with open(self.TREE_BASE+"props_main_prog_10_SNAP_INT.p", "wb") as fp: 
                    pickle.dump(self.sub_tree_prop, fp, protocol=pickle.HIGHEST_PROTOCOL)



    def get_IA_bias_history(self, ngrid=192, damping_scale=0.1, recompute=False, save=True):
        '''
            In this function we wish to apply the probabilistic bias-estimators to the
            subhalos we have
        '''

        import bacco
        import bacco.probabilistic_bias as pb
        
        if recompute==False:
            with open(self.TREE_BASE+"props_main_prog_10_SNAP_INT.p", 'rb') as fp:
                self.sub_tree_prop = pickle.load(fp)
            try:
                self.sub_tree_prop['IA_bias']
            except:
                print("Code was run with recompute=FALSE, yet density biases have not been computed\n")
        else:
            try:
                self.root_tree_ID
                self.root_index
            except:
                self.get_root_info()

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

            self.sub_tree_prop['IA_bias'] =  np.zeros( (self.snap_0, len(self.root_index), 3) )

            for s in range(0, self.snap_0, self.SNAP_INT):
                try:
                    print("Doing Snapshot {:d}".format(self.snap_0-s))
                    pbm.set_reference_expfactor( 1 / ( 1 + self.redshift[self.snap_0-s-1]) )

                    tr_q, tr_value, tr_mask = pbm._define_tracers(tracer_q=lag_pos[self.snap_0-s-1,...])
                    shape_tensor = bacco.utils.I_to_S(self.sub_tree_prop['SubhaloIntertiaTensorStars'][self.snap_0-s-1,...])
                    
                    self.sub_tree_prop['IA_bias'][self.snap_0-s-1,...] = IA_model.bias_per_object(tr_value, I=shape_tensor)
                except:
                    print("Failed for SNAP {:d}\n".format(self.snap_0-s))

            if save:
                with open(self.TREE_BASE+"props_main_prog_10_SNAP_INT.p", "wb") as fp: 
                    pickle.dump(self.sub_tree_prop, fp, protocol=pickle.HIGHEST_PROTOCOL)

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

    def get_mp_secondary_progenitors(self, recompute=False, save=False, sep_int=None):

        if sep_int is None:
            sep_int = self.SNAP_INT

        if recompute==False:
            print("Function get_mp_secondary_progenitors is being run with recompute=False\n")
            print("Now loading the saved file of secondary progenitors\n")
            self.sec_prog = np.load(self.TREE_BASE+"sec_prog_{:d}_SNAP_INT_v1.npy".format(sep_int), allow_pickle=True)
            print("Done\n")
        else:
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
                self.root_tree_ID
                self.root_index
            except:
                self.get_root_info()

            # For loop over all the possible halos at a certain snapshot
            print("Starting the Loop")
            _sec_prog = {self.snap_0 - 1 - sep_int*i:{} for i in range(1, int(self.snap_0/sep_int))} # This will then start at 262
            for ii in range(300): #len(self.root_index)):
                tree_offset = self.tree_offsets[self.root_tree_ID[ii]]
                print('Done for halo {:d}'.format(ii), end='\r')
                
                snap_i = self.snap_0
                mp = self.root_index[ii]
                while snap_i > 34: # at such high redshifts the code will break down due to lack of subhalos anyway
                    f_a_s = [] # this will store all the progenitors during the loop
                    Np = self.sub_firstprog[mp + tree_offset]
                    while Np != -1:
                        print(Np, end=',')
                        f_a_s.append( np.int64(Np) )
                        Np = self.sub_nextprog[Np + tree_offset]

                    snap_i -= 1
                    
                    mp = self.sub_mainprog[mp + tree_offset] # pass to the next main progenitor

                    if len(f_a_s)==0:
                        _sec_prog[snap_i-1][ii] = [-1]
                    else:
                        _sec_prog[snap_i-1][ii] = f_a_s
                    if mp == -1:
                        break
            print("Finished")

            # print("Preparing data for being saved\n")
            # rows = []
            # for snap, sub_dict in _sec_prog.items():
            #     for subhalo_index, prog_list in sub_dict.items():
            #         for prog in prog_list:
            #             rows.append( (snap, subhalo_index, prog) )

            # # Convert to a NumPy structured array
            # dtype = np.dtype([('snap', np.int32), ('subhalo', np.int32), ('prog', np.int64)])
            # print("Converting to a NumPy structured array\n")
            # flat_array = np.array(rows, dtype=dtype)

            # self.sec_prog = flat_array

            # print("Done with secondary progenitors!\n")
            # self.clean_tree()
            # print("Tree cleaned!\n")

            # if save:
            #     print("Saving data at "+self.TREE_BASE+"sec_prog_{:d}_SNAP_INT_v1.npy\n".format(sep_int))
            #     np.save(self.TREE_BASE+"sec_prog_{:d}_SNAP_INT_v1.npy".format(sep_int), flat_array, allow_pickle=True)
            #     print("Done!\n")

        return _sec_prog

    def get_main_prog_props(self, recompute=False, save=False):
        '''
            This function is responsible for getting the tree-pertinent information
            for the subhalos in the main-progenitor branch, starting from the roots 
            stored at self.root_index at snap_0.
        '''

        if recompute==False:
            print("Function get_sub_tree_props is being run with recompute=False\n")
            print("Now loading the saved file of subhalo tree-pertinent properties of main progenitors\n")
            with open(self.TREE_BASE+"props_main_prog_10_SNAP_INT.p", 'rb') as fp:
                self.main_prop = pickle.load(fp)
            print("Done\n")
            
        else:
            # Get tree-relative ID and index of subhalos at Snap_0.
            # These are the roots of the trees that we will employ for all computations
            try:
                self.root_tree_ID
                self.root_index
            except:
                self.get_root_info()

            print("Loading the number of stars in each subhalo\n")
            self.sub_lenstars = np.fromfile("/cosmos_storage/home/fgmaion/prob-bias/MTNG/tree_data/len_stars.bin", dtype=np.int32)
            print("Done\n")

            # Walk the tree
            try:
                self.mp_idx
            except:
                self.walk_subs()

            self.main_prop['LenStars'] = np.zeros( (self.snap_0, len(self.root_index)) )
            print("Dimension of root_index is {:d}\n".format(len(self.root_index)))
            for s in range(self.snap_0-self.min_snap+1):
                print("Now going through snap {:d} of the merger tree\r".format(self.snap_0-s))
                self.main_prop['LenStars'][self.snap_0-s-1,:] = self.sub_lenstars[self.mp_idx[self.snap_0-s-1]]
                zero_mask = self.mp_idx[self.snap_0-s-1]==-1
                print("Dimension of zero mask is {:d}\n".format(np.where(zero_mask)[0].shape[0]))
                self.main_prop['LenStars'][self.snap_0-s-1,zero_mask] = np.zeros(np.where(zero_mask)[0].shape[0])
            
            if save:
                print("Now saving the subhalo tree-pertinent properties in "+self.TREE_BASE+"props_main_prog_10_SNAP_INT.p\n")
                with open(self.TREE_BASE+"props_main_prog_10_SNAP_INT.p", "wb") as fp: 
                    pickle.dump(self.main_prop, fp, protocol=pickle.HIGHEST_PROTOCOL)
                print("Done!\n")

            print("Done with the main progenitors!\n")

        return None

    def get_sec_prog_props(self, recompute=False, save=False):
        '''
            This function is responsible for getting the tree-pertinent information
            for the subhalos in the main-progenitor branch, starting from the roots 
            stored at self.root_index at snap_0.
        '''

        try:
            self.sec_prog
        except:
            try:
                self.get_mp_secondary_progenitors(recompute=False, save=False, sep_int=1)
            except:
                self.get_mp_secondary_progenitors(recompute=False, save=False, sep_int=1)

        print("Cleaning unused tree properties\n")
        self.clean_tree()
        print("Done Cleaning\n")

        try:
            self.sub_secprog_prop
        except:
            self.get_secprog_props(recompute=False, save=False)

        print("Building lookup index for secondary progenitors\n")
        index_dict = build_lookup_index(self.sec_prog, name1='snap', name2='subhalo', name3='prog')

        _lenstars = {'LenStars': {}}
        for s in range(1,self.snap_0-self.min_snap+1):
            snap_index = self.snap_0 - s - 1
            _lenstars['LenStars'][snap_index] = {}
            print("Now going through snap {:d} of the merger tree\n".format(snap_index+1))

            for ii in range(len(self.root_index)):
                print("Now going through subhalo {:d} of the merger tree\r".format(ii), end='\r')
                sec_prog_snap = index_dict[snap_index].get(ii, [])
                _lenstars['LenStars'][snap_index][ii] = self.sub_lenstars[sec_prog_snap]

        rows = []
        for field_name, sub_dict in _lenstars.items():
            for snap, ss_dict in sub_dict.items():
                for subhalo_index, prop_list in ss_dict.items():
                    for prop in prop_list:
                        rows.append((field_name, snap, subhalo_index, float(prop)))      

        flat_array = np.array(rows, dtype=self.sub_secprog_prop.dtype)

        combine = np.concatenate((self.sub_secprog_prop, flat_array))

        if save:
            print("Now saving the subhalo tree-pertinent properties in "+self.TREE_BASE+"props_sec_prog_{:d}_SNAP_INT_v1.npy\n".format(self.SNAP_INT))
            np.save(self.TREE_BASE+"props_sec_prog_{:d}_SNAP_INT_v1.npy".format(self.SNAP_INT), combine, allow_pickle=True)
            print("Done!\n")

        return None
 
    # def get_sub_tree_props(self, recompute=False, save=False):
    #     '''
    #         This function is responsible for getting the tree-pertinent information
    #         for the subhalos which we have walked up the tree.
    #     '''

    #     if recompute==False:
    #         print("Function get_sub_tree_props is being run with recompute=False\n")
    #         print("Now loading the saved file of subhalo tree-pertinent properties of main progenitors\n")
    #         with open(self.TREE_BASE+"props_main_prog_10_SNAP_INT.p", 'rb') as fp:
    #             self.sub_tree_prop = pickle.load(fp)
    #         print("Done\n")
            
    #         print("Now loading the saved file of secondary progenitor properties\n")
    #         self.sub_secprog_prop = np.load(self.TREE_BASE+"props_sec_prog_{:d}_SNAP_INT_v1.npy".format(self.SNAP_INT), allow_pickle=True)
    #         print("Done\n")
    #     else:
    #         # Get tree-relative ID and index of subhalos at Snap_0
    #         try:
    #             self.root_tree_ID
    #             self.root_index
    #         except:
    #             self.get_root_info()

    #         print("Loading the number of stars in each subhalo\n")
    #         self.sub_lenstars = np.fromfile("/cosmos_storage/home/fgmaion/prob-bias/MTNG/tree_data/len_stars.bin", dtype=np.int32)
    #         print("Done\n")

    #         try:
    #             self.sub_tree_prop['LenStars']
    #         except:
    #             print("Subhalo File properties are not yet computed so we must get them")
    #             self.get_sub_file_props(recompute=False, save=False)
    #             print("Done!\n")

    #         # Walk the tree
    #         try:
    #             self.mp_idx
    #         except:
    #             self.walk_subs()

    #         self.sub_tree_prop['LenStars'] = np.zeros( (self.snap_0, len(self.root_index)) )
    #         print("Dimension of root_index is {:d}\n".format(len(self.root_index)))
    #         for s in range(self.snap_0-self.min_snap+1):
    #             print("Now going through snap {:d} of the merger tree\r".format(self.snap_0-s))
    #             self.sub_tree_prop['LenStars'][self.snap_0-s-1,:] = self.sub_lenstars[self.mp_idx[self.snap_0-s-1]]
    #             zero_mask = self.mp_idx[self.snap_0-s-1]==-1
    #             print("Dimension of zero mask is {:d}\n".format(np.where(zero_mask)[0].shape[0]))
    #             self.sub_tree_prop['LenStars'][self.snap_0-s-1,zero_mask] = np.zeros(np.where(zero_mask)[0].shape[0])
            
    #         if save:
    #             print("Now saving the subhalo tree-pertinent properties in "+self.TREE_BASE+"props_main_prog_10_SNAP_INT.p\n")
    #             with open(self.TREE_BASE+"props_main_prog_10_SNAP_INT.p", "wb") as fp: 
    #                 pickle.dump(self.sub_tree_prop, fp, protocol=pickle.HIGHEST_PROTOCOL)
    #             print("Done!\n")

    #         print("Done with the main progenitors!\n")

    #         try:
    #             self.sec_prog
    #         except:
    #             try:
    #                 self.get_mp_secondary_progenitors(recompute=False, save=False, sep_int=1)
    #             except:
    #                 self.get_mp_secondary_progenitors(recompute=False, save=False, sep_int=1)

    #         print("Cleaning unused tree properties\n")
    #         self.clean_tree()
    #         print("Done Cleaning\n")

    #         try:
    #             self.sub_secprog_prop
    #         except:
    #             self.get_secprog_props(recompute=False, save=False)

    #         print("Building lookup index for secondary progenitors\n")
    #         index_dict = build_lookup_index(self.sec_prog, name1='snap', name2='subhalo', name3='prog')

    #         _lenstars = {'LenStars': {}}
    #         for s in range(1,self.snap_0-self.min_snap+1):
    #             snap_index = self.snap_0 - s - 1
    #             _lenstars['LenStars'][snap_index] = {}
    #             print("Now going through snap {:d} of the merger tree\n".format(snap_index+1))

    #             for ii in range(len(self.root_index)):
    #                 print("Now going through subhalo {:d} of the merger tree\r".format(ii), end='\r')
    #                 sec_prog_snap = index_dict[snap_index].get(ii, [])
    #                 _lenstars['LenStars'][snap_index][ii] = self.sub_lenstars[sec_prog_snap]

    #         rows = []
    #         for field_name, sub_dict in _lenstars.items():
    #             for snap, ss_dict in sub_dict.items():
    #                 for subhalo_index, prop_list in ss_dict.items():
    #                     for prop in prop_list:
    #                         rows.append((field_name, snap, subhalo_index, float(prop)))      

    #         flat_array = np.array(rows, dtype=self.sub_secprog_prop.dtype)

    #         combine = np.concatenate((self.sub_secprog_prop, flat_array))

    #     if save:
    #         print("Now saving the subhalo tree-pertinent properties in "+self.TREE_BASE+"props_sec_prog_{:d}_SNAP_INT_v1.npy\n".format(self.SNAP_INT))
    #         np.save(self.TREE_BASE+"props_sec_prog_{:d}_SNAP_INT_v1.npy".format(self.SNAP_INT), combine, allow_pickle=True)
    #         print("Done!\n")

    #     return None

    def get_exsitu_stellar_mass(self, recompute=False, save=False):
        '''
            This function is responsible for getting the ex-situ stellar mass of subhalos
            at different snapshots.

            This is done by summing up the masses of all secondary progenitors (non-main progenitors).
        '''

        # Check that LenStars is computed
        try:
            self.main_prop['LenStars']
        except:
            print("Main progenitor properties are not yet computed")
            self.get_main_prog_props(recompute=False, save=False)
            print("Done!\n")

        # Sum up the stellar mass of all secondary progenitors
        for s in range(self.snap_0):
            for ii in range(len(self.root_index)):
                _len_stars = np.sum(self.sub_secprog_prop['LenStars'][self.snap_0-s-1][ii])
        

    # def get_secprog_props(self, recompute=False, save=False, sep_int=None):
    #     '''
    #         This function is responsible for getting the file-pertinent information
    #         for the secondary progenitors of subhalos
    #     '''

    #     if sep_int is None:
    #         sep_int = self.SNAP_INT

    #     if recompute==False:
    #         print("Function get_secprog_props is being run with recompute=False\n")
    #         print("Now loading the saved file of secondary progenitor properties\n")
    #         self.sub_secprog_prop = np.load(self.TREE_BASE+"props_sec_prog_{:d}_SNAP_INT_v1.npy".format(sep_int), allow_pickle=True)
    #         print("Done\n")
    #     else:
    #         # Get tree-relative ID and index of subhalos at Snap_0
    #         try:
    #             self.root_tree_ID
    #             self.root_index
    #         except:
    #             self.get_root_info()
            
    #         print("Walking Tree (Secondary Progenitors Included)\n")
    #         # Walk the tree
    #         try:
    #             self.sec_prog
    #         except:
    #             self.get_mp_secondary_progenitors(recompute=False, save=False)

    #         print("Done Walking Tree (Secondary Progenitors Included)\n")

    #         _sub_secprog_prop = {}
    #         _sub_secprog_prop['SubhaloMass'] = {}
    #         _sub_secprog_prop['SubhaloMassType'] = {}

    #         # loading all data
    #         print("Starting to load data\n")
    #         for s in range(sep_int, self.snap_0-self.min_snap+1, sep_int): 
    #             print("Getting the tree-relevant IDs and Indexes of subhalos in snapshot\n")
    #             treeID, treeIndex, Nfile = self._sub_treeindex(snap=self.snap_0-s)

    #             # get all the interesting properties stored in the group-files, for all available snapshots
    #             ################ READING SINGLE-FILES ###################
    #             total_Nsub = int(0)
    #             for i in range(640):
    #                 print("Reading file {:d} of group {:d}\n".format(i, self.snap_0-s), end='\r')
    #                 with h5py.File(self.sim_base+'groups_{:03d}/fof_subhalo_tab_{:03d}.{:d}.hdf5'.format(self.snap_0-s,self.snap_0-s,i), 'r') as f:
    #                     total_Nsub += int(f['Header'].attrs['Nsubhalos_ThisFile'])

    #             _mass = np.empty(total_Nsub)
    #             _mass_type = np.empty((total_Nsub, 6))
    #             _pos = np.empty((total_Nsub, 3))

    #             cumsub = 0
    #             for i in range(640):
    #                 print("Reading file {:d} of group {:d}\n".format(i, self.snap_0-s), end='\r')

    #                 with h5py.File(self.sim_base+'groups_{:03d}/fof_subhalo_tab_{:03d}.{:d}.hdf5'.format(self.snap_0-s,self.snap_0-s,i), 'r') as f:
    #                     nsub = int(f['Header'].attrs['Nsubhalos_ThisFile'])

    #                     try:
    #                         f['Subhalo']['SubhaloMass']
    #                     except:
    #                         continue

    #                     _mass[cumsub:cumsub+nsub]              = f['Subhalo']['SubhaloMass']
    #                     _mass_type[cumsub:cumsub+nsub,:]       = f['Subhalo']['SubhaloMassType']
    #                     _pos[cumsub:cumsub+nsub,:]             = f['Subhalo']['SubhaloPos']

    #                 cumsub += nsub
    #             ################# DONE WITH THIS SNAPSHOT  ###################
    #             print("Done with snap {:d}\n".format(self.snap_0-s))

    #             try:
    #                 _sub_secprog_prop['SubhaloMass'][self.snap_0-s-1] = {}
    #                 _sub_secprog_prop['SubhaloMassType'][self.snap_0-s-1] = {}

    #                 # Map values to their positions in treeIndex
    #                 treeIndex = np.array(treeIndex)
    #                 val_to_idx = {val: idx for idx, val in enumerate(treeIndex)}

    #                 # Flatten the progenitor list
    #                 sel = (self.sec_prog['snap'] == self.snap_0-s-1) & (self.sec_prog['subhalo'] == ii)
    #                 sp_index = [np.array(self.sec_prog['prog'][sel], dtype=int) for ii in range(len(self.root_index))]
    #                 sp_1d = np.concatenate(sp_index)

    #                 # Create mask and array of positions in treeIndex
    #                 mask = np.isin(sp_1d, treeIndex)
    #                 positions = np.array([val_to_idx[val] for val in sp_1d[mask]])

    #                 # Lengths of original arrays
    #                 len_array = np.array([len(x) for x in sp_index])
    #                 cut_idx = np.r_[0, len_array.cumsum()]

    #                 # Count how many were kept in each chunk
    #                 len_mask = np.add.reduceat(mask, cut_idx[:-1])
    #                 cl_m = np.r_[0, len_mask.cumsum()]

    #                 # Rebuild output with treeIndex positions instead of values
    #                 self.out = [positions[i:j] for i, j in zip(cl_m[:-1], cl_m[1:])]

    #                 for ii in range(len(self.root_index)):
    #                     _sub_secprog_prop['SubhaloMass'][self.snap_0-s-1][ii] = _mass[self.out[ii]]
    #                     _sub_secprog_prop['SubhaloMassType'][self.snap_0-s-1][ii] = _mass_type[self.out[ii],...]

    #             except:
    #                 print("FAILED for snapshot {:d}\n".format(self.snap_0-s))

    #         if save:
    #             print("Starting to flatten data and prepare it for saving \n")
    #             rows = []
    #             for field_name, sub_dict in _sub_secprog_prop.items():
    #                 for snap, ss_dict in sub_dict.items():
    #                     for subhalo_index, prop_list in ss_dict.items():
    #                         for prop in prop_list:
    #                             if isinstance(prop, (list, tuple, np.ndarray)):
    #                                 for i, p in enumerate(prop):
    #                                     rows.append((f"{field_name}_{i}", snap, subhalo_index, float(p)))
    #                             else:
    #                                 rows.append((field_name, snap, subhalo_index, float(prop)))                    

    #             # Convert to a NumPy structured array
    #             dtype = np.dtype([('name', '<U50'), ('snap', np.int32), ('subhalo', np.int32), ('prop', np.float64)])
    #             flat_array = np.array(rows, dtype=dtype)

    #             self.sub_secprog_prop = flat_array

    #             np.save(self.TREE_BASE+"props_sec_prog_{:d}_SNAP_INT_v1.npy".format(sep_int), self.sub_secprog_prop, allow_pickle=True)