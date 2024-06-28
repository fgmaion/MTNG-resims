import numpy as np
import bacco
import h5py

class split_halos():

    def __init__(self, sim):
        self.sim = sim

    def q_pos(self, sim, npart=4320, BoxSize=500, corr_fac=125):
        
        gal_mbid = sim.sub['IDMostbound']

        q = np.zeros(gal_mbid.shape + (3,), dtype=np.float32)

        q[..., 0] = gal_mbid // npart**2
        q[..., 1] = (gal_mbid // npart) % npart
        q[..., 2] = gal_mbid % npart

        # normalize correctly
        q *= (BoxSize / npart)

        q[..., 0] = ( q[...,0] - corr_fac ) % BoxSize # Correction due to differences between MTNG and MTNG-mimic

        return q

    def sample_halos(self, Nhalos=None, gal_sel=None, sel_mask=None):
        '''
            int:Nhalos
            Amount of halos we wish to randomly select from total population

            int array:sel_mask
            Array of indices of the pre-selected subhalos. To be used as a mask for subhalo quantities

        '''

        index_mask = self.sim.sub['fof_index'][gal_sel][sel_mask]

        unique_indices = np.unique(index_mask)
        if Nhalos is not None:
            fof_choice = np.random.choice(unique_indices, min(Nhalos, len(unique_indices)), replace=False)
        else: 
            fof_choice = unique_indices
        
        mask = np.isin(index_mask, fof_choice)
        halo_frac = len(fof_choice) / len(unique_indices)
        
        return mask, fof_choice, halo_frac

    def subhalo_sel(self, mhalo_edges=None, vmax_sel=False, Nhalos=10, DM_only=False):
        '''
        Function to select subhalos belonging to the Nhalos sampled halos in mass range delimited by mhalo_edges.

        array of floats:mass_edges
        Contains the edges of the mass-bin in which we wish to select our halo

        bool:vmax_sel
        Whether to split the selection not only by mass, but by concentration as well
        '''
        # Typical galaxy selection. May have to change particle limit
        gal_sel = np.where( (self.sim.sub['LenType'][:,4]>200) & (np.sum(self.sim.sub['MassType'], axis=1)>1) )[0]
        
        # Get the galaxy parent-halo masses
        parent_mass = self.sim.sub['parent_halo']['mfof'][gal_sel] * 1e10

        sel_mask = {}
        # subselection by concentration
        if vmax_sel is True:
            fof_choice = {}
            fof_choice['high_c'] = []
            fof_choice['low_c'] = []

            halo_frac = {}
            mbin_tot = {}
            mbin_tot['high_c'] = np.zeros(mhalo_edges.shape[0])
            halo_frac['high_c'] = np.zeros(mhalo_edges.shape[0])
            mbin_tot['low_c'] = np.zeros(mhalo_edges.shape[0])
            halo_frac['low_c'] = np.zeros(mhalo_edges.shape[0])

            sel_mask = {}
            sel_mask['high_c'] = {}
            sel_mask['low_c'] = {}
            
            halo_vmax = self.sim.sub['vmax'][self.sim.fof['halo_firstsub']]
            parent_vmax = halo_vmax[self.sim.sub['fof_index'][gal_sel]]

            for m in range(mhalo_edges.shape[0]):
                avg_vmax = np.mean( parent_vmax[np.where( (parent_mass>10**mhalo_edges[m,0]) & (parent_mass<10**mhalo_edges[m,1]) )] )

                # select galaxies in halos of given mass/concentration
                sel_temp_high = np.where( (parent_mass>10**mhalo_edges[m,0]) & (parent_mass<10**mhalo_edges[m,1]) & (parent_vmax > avg_vmax ) )[0]
                sel_temp_low = np.where( (parent_mass>10**mhalo_edges[m,0]) & (parent_mass<10**mhalo_edges[m,1]) & (parent_vmax <= avg_vmax) )[0]

                # sample those halos randomly
                if Nhalos is None or isinstance(Nhalos, int):
                    mask_h, fof_temp_h, halo_frac['high_c'][m] = self.sample_halos(Nhalos=Nhalos//2, gal_sel=gal_sel, sel_mask=sel_temp_high)
                    mask_l, fof_temp_l, halo_frac['low_c'][m] = self.sample_halos(Nhalos=Nhalos//2, gal_sel=gal_sel, sel_mask=sel_temp_low)
                else:
                    mask_h, fof_temp_h, halo_frac['high_c'][m] = self.sample_halos(Nhalos=Nhalos[m]//2, gal_sel=gal_sel, sel_mask=sel_temp_high)
                    mask_l, fof_temp_l, halo_frac['low_c'][m] = self.sample_halos(Nhalos=Nhalos[m]//2, gal_sel=gal_sel, sel_mask=sel_temp_low)
                
                fof_choice['high_c'].extend(fof_temp_h)
                fof_choice['low_c'].extend(fof_temp_l)

                # get the mass in the selected halos
                mbin_tot['high_c'][m] = np.sum( self.sim.fof['halo_mfof'][fof_temp_h] * 1e10 )
                mbin_tot['low_c'][m] = np.sum( self.sim.fof['halo_mfof'][fof_temp_l] * 1e10 )

                # get the galaxies in the selected halos
                sel_mask['high_c'][m] = sel_temp_high[np.where(mask_h)[0]]
                sel_mask['low_c'][m] = sel_temp_low[np.where(mask_l)[0]]

            fof_choice['high_c'] = np.array(fof_choice['high_c'])
            fof_choice['low_c'] = np.array(fof_choice['low_c'])

            mhalos_tot = np.sum( (self.sim.fof['halo_mfof'][fof_choice['high_c']]+self.sim.fof['halo_mfof'][fof_choice['low_c']]) *1e10)
            nhalos = len(fof_choice['high_c']) + len(fof_choice['low_c'])

        else:
            fof_choice = []
            mbin_tot = np.zeros(mhalo_edges.shape[0])
            halo_frac = np.zeros(mhalo_edges.shape[0])
            for m in range(mhalo_edges.shape[0]):
                # select galaxies/subhalos in halos of a given mass
                if DM_only is False:
                    sel_temp = np.where( (parent_mass>10**mhalo_edges[m,0]) & (parent_mass <10**mhalo_edges[m,1]) )[0]
                else:
                    sel_temp = np.where( (parent_mass>10**mhalo_edges[m,0]) & (parent_mass<10**mhalo_edges[m,1]) & (self.sim.sub['len']>200) )[0]

                # sample those halos randomly
                if Nhalos is None or isinstance(Nhalos, int):
                    mask, fof_temp, halo_frac[m] = self.sample_halos(Nhalos=Nhalos, gal_sel=gal_sel, sel_mask=sel_temp)
                else:
                    mask, fof_temp, halo_frac[m] = self.sample_halos(Nhalos=Nhalos[m], gal_sel=gal_sel, sel_mask=sel_temp)
                fof_choice.extend(fof_temp)

                # get the mass in the selected halos
                mbin_tot[m] = np.sum( self.sim.fof['halo_mfof'][fof_temp] * 1e10 )

                # get the galaxies in the selected halos
                sel_mask[m] = sel_temp[np.where(mask)[0]]

            fof_choice = np.array(fof_choice)

            mhalos_tot = np.sum(self.sim.fof['halo_mfof'][fof_choice]*1e10)
            nhalos = len(fof_choice)

        return {'sel':sel_mask, 'h_idx':fof_choice, 'mh_tot':mhalos_tot, 'mh_bin':mbin_tot, 'Nh':nhalos, 'h_frac':halo_frac, 'gal_sel':gal_sel}

    def stellar_mf(self, gal_sel=None, sel_mask=None, mhalo_edges=None, nbins=100, Nhalos=None, vmax_sel=None):
        '''
            Function to compute stellar mass function from the chosen population of halos

            Object:sim
            This is a bacco.simulation object, representing a simulation from which we will load all the information

            array of floats:mass_edges
            Edges of the halo mass-bin over which we will compute the stellar mass-function

            int:nbins
            Number of bins over which to build the histogram

            int:Nhalos
            Number of halos that we wish to sample at this mass-bin

            Bool:vmax_sel
            Whether to split by concentration besides the mass selection
        '''
        
        if sel_mask is None:
            sel_mask = self.subhalo_sel(mhalo_edges=mhalo_edges, vmax_sel=vmax_sel, Nhalos=Nhalos)
        
        bins = np.logspace(8, 13, nbins)
        hist = np.zeros(nbins-1)
        if vmax_sel is True:
            length = len(sel_mask['sel']['high_c'])
        else:
            length = len(sel_mask['sel'])
        for m in range(length):
            if vmax_sel is True:
                mstar_h = self.sim.sub['MassType'][:,4][gal_sel][sel_mask['sel']['high_c'][m]] * 1e10
                mstar_l = self.sim.sub['MassType'][:,4][gal_sel][sel_mask['sel']['low_c'][m]] * 1e10

                if sel_mask['h_frac']['high_c'][m]!=0:
                    hist += np.histogram(mstar_h, bins=bins)[0] / sel_mask['h_frac']['high_c'][m]
                if sel_mask['h_frac']['low_c'][m]!=0:
                    hist += np.histogram(mstar_l, bins=bins)[0] / sel_mask['h_frac']['low_c'][m]
            else:
                if sel_mask['h_frac'][m]!=0:
                    mstar = self.sim.sub['MassType'][:,4][gal_sel][sel_mask['sel'][m]] * 1e10
                    hist += np.histogram(mstar, bins=bins)[0] / sel_mask['h_frac'][m]


        return  {'smf':hist, 'bins':bins}

    def get_bpo(
            self, recompute=False, IA_terms=("J2=2", "J222=", "J2-2-2-")
            ):
        
        z = 1 / self.sim.Cosmology.expfactor - 1
        
        if recompute:

            import bacco.probabilistic_bias as pb

            # MTNG Mimic
            dir_name_dm = "/scratch/cosmosims/TNG_Family/MTNG-mimic/output/"

            dm_mtng = bacco.Simulation(
                basedir=dir_name_dm, halo_file="groups_085/fof_subhalo_history_tab_orph_wweight_085",
                sim_format='gadget_hdf5', ngenic_phases=True, phase_type=2, fixedPk=True
                )

            dm_mtng.header['Seed'] = 100672

            # These are the variables that need to be measured on a Lagrangian grid
            variables = ("Txx", "Txy", "Txz", "Tyy", "Tyz", "Tzz",)

            pbm = pb.ProbabilisticBiasManager(dm_mtng, variables=variables, damping_scale=0.1, ngrid=384)
            # Note if you pass the parameter  cachedir="path/to/some/empty/directory"
            # you may save some time, at the cost of storing some extra files

            IA_model = pbm.setup_bias_model(pb.IA_TensorBiasND, terms=IA_terms, spatial_order=2)

            q = self.q_pos(self.sim)
            I = bacco.utils.S_to_I(self.sim.sub['subhalo_stellar_MOI'])

            bias = pbm.fit_bias(model=IA_model, tracer_q=q, error='qjack4', tracer_properties={'I':I})
            bpo = np.float32(IA_model.bpo)
            np.save("/cosmos_storage/home/fgmaion/prob-bias/MTNG/biases/IA_bias_so0_mtng_z{:.2f}".format(z), [{'bias':bias, 'bpo':bpo}])
        
        else:
            load_bias = np.load("/cosmos_storage/home/fgmaion/prob-bias/MTNG/biases/IA_bias_so0_mtng_z{:.2f}.npy".format(z), allow_pickle=True)[0]
            bpo = load_bias['cs_bpo']

        return bpo

    def bias_sm(self, gal_sel=None, sel_mask=None, bpo=None, mhalo_edges=None, Nhalos=None, vmax_sel=None, recompute=False):
        '''
        Function to get the bias of a certain selection sel_mask, binned as a function of stellar masses
        '''
        
        if bpo is None:
            bpo = self.get_bpo(recompute=recompute)

        if sel_mask is None:
            # subhalos that belong to halos of mass in mass_edges
            sel_mask = self.subhalo_sel(mhalo_edges=mhalo_edges, vmax_sel=vmax_sel, Nhalos=Nhalos)
                
            # # selection of central galaxies
            # sel_mask = sel_mask[np.where(self.sim.sub['central'][sel_mask])]

        sel_comb = []
        for m in range(len(sel_mask['sel'])):
            sel_comb.extend(sel_mask['sel'][m])
        sel_comb = np.array(sel_comb)

        # stellar mass of final selected subhalos
        mstar = self.sim.sub['MassType'][:,4][gal_sel][sel_comb] * 1e10

        # bias per object of final selected subhalos
        bpo_sel = bpo[sel_comb]

        mstar = np.array(mstar, dtype=np.float)
        bpo_sel = np.array(bpo_sel)            

        # selection of galaxies per stellar-mass
        D = 0.5
        ms_edges = np.arange(9.5,12.5,D)
        idx = np.digitize(np.log10(mstar), bins=ms_edges)
        
        bias = np.zeros((len(ms_edges)-1, 5))
        for i in range(len(ms_edges)-1):
            sel_idx = np.where(idx==i+1)[0]
            if len(sel_idx)!=0:
                bias[i] = np.mean( bpo_sel[sel_idx], axis=0 )
            else:
                bias[i] = 0

        return  {'bias':bias, 'm_edges':ms_edges}

    # def cs_rad_sat(self, bpo=None, mass_edges=[12.5,13], nbins=100, Nhalos=None, vmax_sel=None, recompute=False):
        
    #     sel_mask         = self.subhalo_sel(mass_edges=mass_edges, vmax_sel=vmax_sel)
    #     mask, fof_choice = self.sample_halos(Nhalos=Nhalos, sel_mask=sel_mask)

    #     if bpo is None:
    #         bpo = self.get_cs_bpo(recompute=recompute)

    #     bpo_sel = bpo[mask]

    #     mhalos_tot  = np.sum(self.sim.fof['halo_mfof'][fof_choice]*1e10)
    #     nhalos = len(fof_choice)

    #     return  {'smf':hist, 'mh_tot':mhalos_tot, 'Nh':nhalos, 'h_idx':fof_choice}

def read_zoom(base=None, filebase="snapshot_ics_000"):

    files = [filebase+".{:d}.hdf5".format(ifile) for ifile in range(32)]

    with h5py.File(base+files[0], 'r') as f:
        NpartTotal = np.uint64(f['Header'].attrs['NumPart_Total'])

    pos1 = np.zeros((NpartTotal[1], 3), dtype=np.float32)
    pos2 = np.zeros((NpartTotal[2], 3), dtype=np.float32)
    pos3 = np.zeros((NpartTotal[3], 3), dtype=np.float32)

    istart1 = np.uint64(0); istart2 = np.uint64(0); istart3 = np.uint64(0)
    for ffname in files:
        with h5py.File(base+ffname, 'r') as f:
            npts1 = np.uint64(f['Header'].attrs['NumPart_ThisFile'][1])
            npts2 = np.uint64(f['Header'].attrs['NumPart_ThisFile'][2])
            npts3 = np.uint64(f['Header'].attrs['NumPart_ThisFile'][3])

            print(
                'Read data for {0}/{1} {1}/{2} {3}/{4} particles...'.format(
                    istart1+npts1, NpartTotal[1], istart2+npts2, NpartTotal[2], istart3+npts3, NpartTotal[3]))

            if npts1 >0:
                pos1[istart1:istart1 + npts1] = f[u'PartType1']['Coordinates'][()]
            if npts2 >0:
                pos2[istart2:istart2 + npts2] = f[u'PartType2']['Coordinates'][()]
            if npts3 >0:
                pos3[istart3:istart3 + npts3] = f[u'PartType3']['Coordinates'][()]

        istart1 += npts1
        istart2 += npts2
        istart3 += npts3

    return {'pos1':pos1, 'pos2':pos2, 'pos3':pos3}

def read_cpu(filename=None, skiprows=[]):
    with open(filename) as f:
        lines = f.readlines()

    d = {}

    i=0
    for line in lines:
        line.strip()

        if i == 0:
            columns = [item.strip() for item in line.split(',')]
            for index, elem in enumerate(columns[:-1]):
                d[columns[index]] = []
        elif i in skiprows:
            i = i + 1
            continue
        else:
            data = [item.strip() for item in line.split(',')]
            for index, elem in enumerate(data[:-1]):
                d[columns[index]].append(float(data[index]))
        i = i + 1

    return d