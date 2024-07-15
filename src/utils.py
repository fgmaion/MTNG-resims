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

    def halo_sel(self, mhalo_edges=None, Nhalos=None, vmax_sel=False):
        '''
        Function to select subhalos belonging to the Nhalos sampled halos in mass range delimited by mhalo_edges.

        array of floats:mass_edges
        Contains the edges of the mass-bin in which we wish to select our halo

        bool:vmax_sel
        Whether to split the selection not only by mass, but by concentration as well
        '''
        # Typical galaxy selection. May have to change particle limit
        m200b = 1e10 * self.sim.fof['halo_m200b']
        
        # Total Mass
        mhalos_tot = 0
        nhalos = 0

        if vmax_sel is True:
            fof_choice = {}
            halo_frac = {}
            
            halo_vmax = self.sim.sub['vmax'][self.sim.fof['halo_firstsub']]
            m200b = 1e10 * self.sim.fof['halo_m200b']
            r200b = self.sim.fof['halo_r200b']
            G_newton = 4.3009172706e-9 #Mpc/M_sun * (km/s)**2
            halo_v200 = np.sqrt(G_newton*m200b/r200b)
            vmax_v200 = halo_vmax / halo_v200

            for m in range(mhalo_edges.shape[0]):
                fof_choice[m] = []
                halo_frac[m] = []

                vratio_m = vmax_v200[np.where( (m200b>10**mhalo_edges[m,0]) & (m200b<10**mhalo_edges[m,1]) )]
                vratio_bins = np.linspace(vratio_m.min(), vratio_m.max(), Nhalos+1)

                for v in range(Nhalos):
                    # select galaxies in halos of given mass/concentration
                    sel_temp = np.where( (m200b>10**mhalo_edges[m,0]) & (m200b<10**mhalo_edges[m,1]) & (vmax_v200 > vratio_bins[v] ) & (vmax_v200 < vratio_bins[v+1] ) )[0]
                    
                    if len(sel_temp)==0:
                        continue

                    fof_temp = np.random.choice(sel_temp, 1, replace=False)
                    
                    fof_choice[m].append(fof_temp[0])
                    halo_frac[m].append(1 / len(sel_temp))

        else:
            fof_choice = []
            halo_frac = []
            mbin_tot = np.zeros(mhalo_edges.shape[0])
            
            for m in range(mhalo_edges.shape[0]):
                # select halos in mass-bin
                sel_temp = np.where( (m200b>10**mhalo_edges[m,0]) & (m200b <10**mhalo_edges[m,1]) )[0]

                if Nhalos is not None:
                    # sample those halos randomly
                    fof_temp = np.random.choice(sel_temp, Nhalos, replace=False)
                    fof_choice.extend(fof_temp)

                    halo_frac.append( Nhalos / len(sel_temp))
                else:
                    fof_temp=sel_temp
                    fof_choice.extend(fof_temp)

                    halo_frac.append(1)

            fof_choice = np.array(fof_choice)

        return {'sel':fof_choice, 'h_frac':halo_frac}

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
        m200b = self.sim.fof['halo_m200b']
        if DM_only is False:
            gal_sel = np.where(m200b[self.sim.sub['parent_halo']['index']]>0)[0]
#            gal_sel = np.where( (self.sim.sub['LenType'][:,4]>200) & (np.sum(self.sim.sub['MassType'], axis=1)>1) )[0]
        else:
            gal_sel = np.where( (self.sim.sub['len'] > 13) & (m200b[self.sim.sub['parent_halo']['index']] > 0) )[0]
        
        # Get the galaxy parent-halo masses
        parent_mass = self.sim.sub['parent_halo']['mfof'][gal_sel] * 1e10

        # Total Mass
        mhalos_tot = 0
        nhalos = 0
        
        sel_mask = {}
        # subselection by concentration
        if vmax_sel is True:
            fof_choice = {}
            halo_frac = {}
            mbin_tot = {}
            sel_mask = {}
            
            halo_vmax = self.sim.sub['vmax'][self.sim.fof['halo_firstsub']]
            m200b = self.sim.fof['halo_m200b']
            r200b = self.sim.fof['halo_r200b']
            G_newton = 4.3009172706e-9 #Mpc/M_sun * (km/s)**2
            parent_v200 = np.sqrt(G_newton*m200b/r200b)[self.sim.sub['parent_halo']['index'][gal_sel]]
            parent_vmax = halo_vmax[self.sim.sub['fof_index'][gal_sel]]
            vmax_v200 = parent_vmax / parent_v200

            for m in range(mhalo_edges.shape[0]):
                halo_frac[m] = {}
                fof_choice[m] = {}
                mbin_tot[m] = {}
                sel_mask[m] = {}
 
                vratio_m = vmax_v200[np.where( (parent_mass>10**mhalo_edges[m,0]) & (parent_mass<10**mhalo_edges[m,1]) )]
                if Nhalos is None or isinstance(Nhalos, int):
                    vratio_bins = np.linspace(vratio_m.min(), vratio_m.max(), Nhalos+1)
                else:
                    vratio_bins = np.linspace(vratio_m.min(), vratio_m.max(), Nhalos[m]+1)

                for v in range(len(vratio_bins)-1):
                    fof_choice[m][v] = []

                    # select galaxies in halos of given mass/concentration
                    sel_temp = np.where( (parent_mass>10**mhalo_edges[m,0]) & (parent_mass<10**mhalo_edges[m,1]) & (vmax_v200 > vratio_bins[v] ) & (vmax_v200 < vratio_bins[v+1] ) )[0]
                    
                    if len(sel_temp)==0:
                        sel_temp = np.where( (parent_mass>10**mhalo_edges[m,0]) & (parent_mass<10**mhalo_edges[m,1]) )[0]

                    mask, fof_temp, halo_frac[m][v] = self.sample_halos(Nhalos=1, gal_sel=gal_sel, sel_mask=sel_temp)
                    
                    fof_choice[m][v].extend(fof_temp)

                    # get the mass in the selected halos
                    mbin_tot[m][v] = np.sum( self.sim.fof['halo_mfof'][fof_temp] * 1e10 )

                    # get the galaxies in the selected halos
                    sel_mask[m][v] = sel_temp[np.where(mask)[0]]

                    fof_choice[m][v] = np.array(fof_choice[m][v])

                    mhalos_tot += 1e10 * np.sum( self.sim.fof['halo_mfof'][fof_choice[m][v]] )
                    nhalos += len(fof_choice[m][v])

        else:
            fof_choice = []
            mbin_tot = np.zeros(mhalo_edges.shape[0])
            halo_frac = np.zeros(mhalo_edges.shape[0])
            for m in range(mhalo_edges.shape[0]):
                # select galaxies/subhalos in halos of a given mass
                if DM_only is False:
                    sel_temp = np.where( (parent_mass>10**mhalo_edges[m,0]) & (parent_mass <10**mhalo_edges[m,1]) )[0]
                else:
                    sel_temp = np.where( (parent_mass>10**mhalo_edges[m,0]) & (parent_mass<10**mhalo_edges[m,1]) )[0]

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
        counts     = np.zeros(nbins-1)
        hist       = np.zeros(nbins-1)
        mstar_mean = np.zeros(nbins-1)

        for m in range(len(sel_mask['sel'])):
            if vmax_sel is True:
                for v in range(len(sel_mask['sel'][m])):
                    mstar = self.sim.sub['MassType'][:,4][gal_sel][sel_mask['sel'][m][v]] * 1e10 / self.sim.Cosmology.pars['hubble']

                    if sel_mask['h_frac'][m][v]!=0:
                        ids = np.digitize(mstar, bins)
                        counts_i = [np.sum( np.ones(len(mstar))[np.where(ids==i)]) for i in range(1,len(bins))]
                        counts += counts_i
                        hist   += np.array(counts_i) / sel_mask['h_frac'][m][v]
                        mstar_mean += [np.sum(mstar[np.where(ids==i)]) for i in range(1,len(bins))]

            else:
                if sel_mask['h_frac'][m]!=0:
                    mstar = self.sim.sub['MassType'][:,4][gal_sel][sel_mask['sel'][m]] * 1e10 / self.sim.Cosmology.pars['hubble']
                    ids = np.digitize(mstar, bins)
                    counts_i = [np.sum( np.ones(len(mstar))[np.where(ids==i)]) for i in range(1,len(bins))]
                    counts += counts_i
                    hist   += np.array(counts_i) / sel_mask['h_frac'][m]
                    mstar_mean += [np.sum(mstar[np.where(ids==i)]) for i in range(1,len(bins))]

        bin_width = np.log10(bins[1:])-np.log10(bins[:-1])
        norm = 1 / ( ( self.sim.header['BoxSize'] / self.sim.Cosmology.pars['hubble'] )**3 * bin_width )

        return  {'smf':norm * hist, 'bins':bins, 'mstar':mstar_mean / counts}

    def halo_smf(self, sel_mask=None, mhalo_edges=None, nbins=100, Nhalos=None, vmax_sel=None):
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
        
        bins = np.logspace(8, 13, nbins)
        counts     = np.zeros(nbins-1)
        hist       = np.zeros(nbins-1)
        mstar_mean = np.zeros(nbins-1)

        for m in range(len(sel_mask['sel'])):
            if vmax_sel is True:
                for v in range(len(sel_mask['sel'][m])):
                    gal_sel = np.where(self.sim.sub['parent_halo']['index']==sel_mask['sel'][m][v])
                    mstar = self.sim.sub['MassType'][:,4][gal_sel] * 1e10 / self.sim.Cosmology.pars['hubble']

                    if sel_mask['h_frac'][m][v]!=0:
                        ids = np.digitize(mstar, bins)
                        counts_i = [np.sum( np.ones(len(mstar))[np.where(ids==i)]) for i in range(1,len(bins))]
                        counts += counts_i
                        hist   += np.array(counts_i) / sel_mask['h_frac'][m][v]
                        mstar_mean += [np.sum(mstar[np.where(ids==i)]) for i in range(1,len(bins))]

            else:
                if sel_mask['h_frac'][m]!=0:
                    gal_sel = np.where(self.sim.sub['parent_halo']['index']==sel_mask['sel'][m])
                    mstar = self.sim.sub['MassType'][:,4][gal_sel] * 1e10 / self.sim.Cosmology.pars['hubble']
                    ids = np.digitize(mstar, bins)
                    counts_i = [np.sum( np.ones(len(mstar))[np.where(ids==i)]) for i in range(1,len(bins))]
                    counts += counts_i
                    hist   += np.array(counts_i) / sel_mask['h_frac'][m]
                    mstar_mean += [np.sum(mstar[np.where(ids==i)]) for i in range(1,len(bins))]

        bin_width = np.log10(bins[1:])-np.log10(bins[:-1])
        norm = 1 / ( ( self.sim.header['BoxSize'] / self.sim.Cosmology.pars['hubble'] )**3 * bin_width )

        return  {'smf':norm * hist, 'bins':bins, 'mstar':mstar_mean / counts}

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
            S = bacco.utils.I_to_S(self.sim.sub['subhalo_stellar_MOI'])

            bias = pbm.fit_bias(model=IA_model, tracer_q=q, error='qjack4', tracer_properties={'I':S})
            bpo = np.float32(IA_model.bpo)
            np.save("/cosmos_storage/home/fgmaion/prob-bias/MTNG/biases/IA_bias_so0_mtng_z{:.2f}".format(z), [{'bias':bias, 'bpo':bpo}])
        
        else:
            load_bias = np.load("/cosmos_storage/home/fgmaion/prob-bias/MTNG/biases/IA_bias_so0_mtng_z{:.2f}.npy".format(z), allow_pickle=True)[0]
            bpo = load_bias['bpo']

        return bpo

    def bias_sm(self, gal_sel=None, sel_mask=None, bpo=None, mhalo_edges=None, Nhalos=None, vmax_sel=None, bins=None, recompute=False):
        '''
        Function to get the bias of a certain selection sel_mask, binned as a function of stellar masses
        '''
        
        if bpo is None:
            bpo = self.get_bpo(recompute=recompute)[gal_sel]

        nbins      = len(bins)

        counts     = np.zeros(nbins-1)
        hist       = np.zeros(nbins-1)
        mstar_mean = np.zeros(nbins-1)

        bias = np.zeros((nbins-1, 3))

        for m in range(len(sel_mask['sel'])):
            if vmax_sel is True:
                for v in range(len(sel_mask['sel'][m])):
                    mstar = self.sim.sub['MassType'][:,4][gal_sel][sel_mask['sel'][m][v]] * 1e10 / self.sim.Cosmology.pars['hubble']

                    if sel_mask['h_frac'][m][v]!=0:
                        ids = np.digitize(mstar, bins)
                        counts_i = np.array([np.sum( np.ones(len(mstar))[np.where(ids==i)]) for i in range(1,len(bins))]) / sel_mask['h_frac'][m][v]
                        counts += counts_i
                        bias += np.array([np.sum( bpo[sel_mask['sel'][m][v]][np.where(ids==i)], axis=0) for i in range(1,len(bins))]) / sel_mask['h_frac'][m][v]
                        mstar_mean += np.array([np.sum(mstar[np.where(ids==i)]) for i in range(1,len(bins))]) / sel_mask['h_frac'][m][v]

            else:
                mstar = self.sim.sub['MassType'][:,4][gal_sel][sel_mask['sel'][m]] * 1e10 / self.sim.Cosmology.pars['hubble']

                if sel_mask['h_frac'][m]!=0:
                    ids = np.digitize(mstar, bins)
                    counts_i = [np.sum( np.ones(len(mstar))[np.where(ids==i)]) for i in range(1,len(bins))] / sel_mask['h_frac'][m]
                    counts += counts_i
                    bias += [np.sum( bpo[sel_mask['sel'][m]][np.where(ids==i)], axis=0) for i in range(1,len(bins))] / sel_mask['h_frac'][m]
                    mstar_mean += [np.sum(mstar[np.where(ids==i)]) for i in range(1,len(bins))] / sel_mask['h_frac'][m]

        return  {'bias':bias / counts[:,np.newaxis], 'mstar':mstar_mean / counts}

    # def cs_rad_sat(self, bpo=None, mass_edges=[12.5,13], nbins=100, Nhalos=None, vmax_sel=None, recompute=False):
        
    #     sel_mask         = self.subhalo_sel(mass_edges=mass_edges, vmax_sel=vmax_sel)
    #     mask, fof_choice = self.sample_halos(Nhalos=Nhalos, sel_mask=sel_mask)

    #     if bpo is None:
    #         bpo = self.get_cs_bpo(recompute=recompute)

    #     bpo_sel = bpo[mask]

    #     mhalos_tot  = np.sum(self.sim.fof['halo_mfof'][fof_choice]*1e10)
    #     nhalos = len(fof_choice)

    #     return  {'smf':hist, 'mh_tot':mhalos_tot, 'Nh':nhalos, 'h_idx':fof_choice}

    def gas_frac(self, m500_edges=None, sel_mask=None, vmax_sel=False):
        '''
        '''

        if vmax_sel is True:
            # Get the indices of the selected halos
            fof_idx = []
            for m in range(len(sel_mask['h_idx'])):
                for v in range(len(sel_mask['h_idx'][m])):
                    fof_idx.extend(sel_mask['h_idx'][m][v])
            
            fof_idx = np.unique(fof_idx)

            m500c = np.log10( 1e10 * self.sim.fof['halo_m500c'][fof_idx] )
            mgas = self.sim.fof['halo_mfof_type'][:,0][fof_idx]
            mfof = self.sim.fof['halo_mfof'][fof_idx]
            mstel = self.sim.fof['halo_mfof_type'][:,4][fof_idx]
        else:
            # Get the indices of the selected halos
            fof_idx = np.unique(sel_mask['h_idx'])

            m500c = np.log10( 1e10 * self.sim.fof['halo_m500c'][fof_idx] )
            mgas = self.sim.fof['halo_mfof_type'][:,0][fof_idx]
            mfof = self.sim.fof['halo_mfof'][fof_idx]
            mstel = self.sim.fof['halo_mfof_type'][:,4][fof_idx]
        
        f_gas = np.zeros(m500_edges.shape[0])
        f_stel = np.zeros(m500_edges.shape[0])
        m500c_mean = np.zeros(m500_edges.shape[0])

        for m in range(m500_edges.shape[0]):
            m_sel = np.where((m500c>m500_edges[m,0])&(m500c<m500_edges[m,1]))

            f_gas[m] = np.mean( mgas[m_sel] / mfof[m_sel] )

            f_stel[m] = np.mean( mstel[m_sel] / mfof[m_sel] )

            m500c_mean[m]= np.mean( 10**m500c[m_sel] )

        return {'f_gas':f_gas, 'f_stel':f_stel, 'm500c':m500c_mean}

def read_zoom(base=None, filebase="snapshot_ics_000"):

    files = [filebase+".{:d}.hdf5".format(ifile) for ifile in range(8)]

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


def dict2d_sum(dict):
    res_sum = np.zeros(len(dict))
    for i in range(len(dict)):
        for j in range(len(dict[i])):
            res_sum[i] += dict[i][j]

    return res_sum

def camels_stellar_mf(id_name, par, nbins=100, boxsize=25, hfac=0.6711):
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
    
    # catalog name
    catalog = '/scratch/fgmaion/CAMELS/1P/1P_p{:d}_'.format(par)+id_name+'/groups_090.hdf5'

    # value of the scale factor
    scale_factor = 1.0

    # open the catalogue
    f = h5py.File(catalog, 'r')

    # read the positions, black hole masses and stellar masses of the subhalos/galaxies
    mstar = f['Subhalo/SubhaloMassType'][:,4]*1e10 #stellar masses in Msun/h

    # close file
    f.close()
    
    bins = np.logspace(8, 13, nbins)
    counts     = np.zeros(nbins-1)
    mstar_mean = np.zeros(nbins-1)

    ids = np.digitize(mstar, bins)
    counts = np.array([np.sum( np.ones(len(mstar))[np.where(ids==i)]) for i in range(1,len(bins))])
    mstar_mean = np.array([np.sum(mstar[np.where(ids==i)]) for i in range(1,len(bins))])

    bin_width = np.log10(bins[1:])-np.log10(bins[:-1])
    norm = 1 / ( ( boxsize / hfac )**3 * bin_width )

    return  {'smf':norm * counts, 'bins':bins, 'mstar':mstar_mean / counts}

def camels_gas_frac(id_name, par, nbins=20):
    '''
    '''

    # catalog name
    catalog = '/scratch/fgmaion/CAMELS/1P/1P_p{:d}_'.format(par)+id_name+'/groups_090.hdf5'

    # value of the scale factor
    scale_factor = 1.0

    # open the catalogue
    f = h5py.File(catalog, 'r')

    # read the positions, black hole masses and stellar masses of the subhalos/galaxies
    m500c = f['Group/Group_M_Crit500'][()]*1e10  #M500c in log10 of Msun/h
    main_sub = f['Group/GroupFirstSub'][()]
    mgas = ( f['Subhalo/SubhaloMassType'][:,0]*1e10 )[main_sub] #M500c in log10 of Msun/h
    mtot = ( f['Subhalo/SubhaloMass'][()]*1e10 )[main_sub]

    # close file
    f.close()

    bins = np.logspace(10, 15, nbins)
    
    f_gas = np.zeros(nbins-1)
    m500c_mean = np.zeros(nbins-1)

    for m in range(nbins-1):
        m_sel = np.where(( m500c>bins[m])&( m500c<bins[m+1]) )
        sel = np.where(mtot[m_sel]!=0)[0]

        f_gas[m] = np.mean( mgas[m_sel][sel] / mtot[m_sel][sel] )

        m500c_mean[m]= np.mean( m500c[m_sel][sel] )

    return {'f_gas':f_gas, 'm500c':m500c_mean}
