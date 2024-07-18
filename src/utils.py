import numpy as np
import bacco
import h5py

class split_halos():

    def __init__(self, sim):
        self.sim = sim

    def halo_sel_setup(self,):
        '''
            Get the quantitities which are necessary for runnig the halo_sel function.
        '''

        self.m200b = 1e10 * self.sim.fof['halo_m200b']
        self.halo_vmax = self.sim.sub['vmax'][self.sim.fof['halo_firstsub']]
        self.r200b = self.sim.fof['halo_r200b']

        G_newton = 4.3009172706e-9 #Mpc/M_sun * (km/s)**2

        self.halo_v200 = np.sqrt(G_newton*self.m200b/self.r200b)
        self.vmax_v200 = self.halo_vmax / self.halo_v200


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

    def halo_sel(self, mhalo_edges=None, Nhalos=None, vmax_sel=False, draws=1):
        '''
        Function to sample halos in mass-bins

        array of floats:mass_edges
        Contains the edges of the mass-bin in which we wish to select our halo

        bool:vmax_sel
        Whether to split the selection not only by mass, but by concentration as well
        '''
        
        import time
        t0 = time.time()

        # Total Mass
        mhalos_tot = 0
        nhalos = 0
        
        presel = self.m200b>10**mhalo_edges[0,0]
        _m200b = self.m200b[presel]
        _vmax_v200 = self.vmax_v200[presel]

        print(time.time()-t0)

        if vmax_sel is True:
            fof_choice = {}
            halo_frac = {}
            
            for m in range(mhalo_edges.shape[0]):
                fof_choice[m] = {d:[] for d in range(draws)}
                halo_frac[m] = {d:[] for d in range(draws)}

                vratio_m = _vmax_v200[np.where( (_m200b>10**mhalo_edges[m,0]) & (_m200b<10**mhalo_edges[m,1]) )]
                vratio_bins = np.linspace(vratio_m.min(), vratio_m.max(), Nhalos+1)

                for v in range(Nhalos):
                    # select galaxies in halos of given mass/concentration
                    sel_temp = np.where( (_m200b>10**mhalo_edges[m,0]) & (_m200b<10**mhalo_edges[m,1]) & (_vmax_v200 > vratio_bins[v] ) & (_vmax_v200 < vratio_bins[v+1] ) )[0]
                    
                    if len(sel_temp)==0:
                        continue

                    for d in range(draws):
                        fof_temp = np.random.choice(sel_temp, 1, replace=False)
                        
                        fof_choice[m][d].append(fof_temp[0])
                        halo_frac[m][d].append(1 / len(sel_temp))

        else:
            fof_choice = {}
            halo_frac = {d:[] for d in range(draws)}
            mbin_tot = np.zeros(mhalo_edges.shape[0])

            print(time.time()-t0)
            
            for m in range(mhalo_edges.shape[0]):
                fof_choice[m] = {d:[] for d in range(draws)}

                # Filter halos in the current mass bin
                in_mass_bin = (_m200b > 10**mhalo_edges[m, 0]) & (_m200b < 10**mhalo_edges[m, 1])
                sel_temp = np.where(in_mass_bin)[0]

                if Nhalos is not None:
                    for d in range(draws):
                        # sample those halos randomly
                        fof_temp = np.random.choice(sel_temp, min(Nhalos, len(sel_temp)), replace=False)
                        fof_choice[m][d].extend(fof_temp)

                        halo_frac[d].append( min(Nhalos, len(sel_temp)) / len(sel_temp))
        
#                        fof_choice[m][d] = np.array(fof_choice[m])

                else:
                    fof_temp=sel_temp
                    fof_choice[m].extend(fof_temp)

                    halo_frac = np.ones(len(fof_temp))


        print(time.time()-t0)


        return {'sel':fof_choice, 'h_frac':halo_frac}

    def total_smf(self, nbins=100):

        bins = np.logspace(8, 13, nbins)
        counts     = np.zeros(nbins-1)
        hist       = np.zeros(nbins-1)
        mstar_mean = np.zeros(nbins-1)

        mstar = self.sim.sub['MassType'][:,4] * 1e10 / self.sim.Cosmology.pars['hubble']
        ids = np.digitize(mstar, bins)

        hist = np.array( [np.sum( np.ones(len(mstar))[np.where(ids==i)]) for i in range(1,len(bins))] )
        mstar_mean = np.array([np.sum(mstar[np.where(ids==i)]) for i in range(1,len(bins))])

        bin_width = np.log10(bins[1:])-np.log10(bins[:-1])
        norm = 1 / ( ( self.sim.header['BoxSize'] / self.sim.Cosmology.pars['hubble'] )**3 * bin_width )

        return  {'smf':norm * hist, 'bins':bins, 'mstar':mstar_mean / hist}

    def halo_smf(self, sel_mask=None, mhalo_edges=None, nbins=100, Nhalos=None, vmax_sel=None, draws=1):
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
        counts     = {d:np.zeros(nbins-1) for d in range(draws)}
        hist       = {d:np.zeros(nbins-1) for d in range(draws)}
        mstar_mean = {d:np.zeros(nbins-1) for d in range(draws)}

        presel = np.where(self.m200b>10**mhalo_edges[0,0])[0]
        first = self.sim.fof['halo_firstsub'][presel]
        nsubs = self.sim.fof['halo_nsubs'][presel]

        for m in range(len(sel_mask['sel'])):
            for d in range(draws):
                if vmax_sel is True:
                    if sel_mask['h_frac'][m][d]!=0:
                        mstar = []
                        for v in range(len(sel_mask['sel'][m][d])):
                            mstar = self.sim.sub['MassType'][:,4][first[sel_mask['sel'][m][d][v]]:first[sel_mask['sel'][m][d][v]]+nsubs[sel_mask['sel'][m][d][v]]] / self.sim.Cosmology.pars['hubble']
                        
                            mstar = 1e10 * np.array(mstar)

                            ids = np.digitize(mstar, bins)
                            counts_i = [np.sum( np.ones(len(mstar))[np.where(ids==i)]) for i in range(1,len(bins))]
                            
                            counts[d] += counts_i
                            hist[d]   += np.array(counts_i) / sel_mask['h_frac'][m][d][v]
                            mstar_mean[d] += [np.sum(mstar[np.where(ids==i)]) for i in range(1,len(bins))]

                else:
                    if sel_mask['h_frac'][d][m]!=0:
                        mstar = []
                        for i in range(len(sel_mask['sel'][m][d])):
                            mstar.extend(self.sim.sub['MassType'][:,4][first[sel_mask['sel'][m][d][i]]:first[sel_mask['sel'][m][d][i]]+nsubs[sel_mask['sel'][m][d][i]]] / self.sim.Cosmology.pars['hubble'])
                        
                        mstar = 1e10 * np.array(mstar)

                        ids = np.digitize(mstar, bins)
                        counts_i = [np.sum( np.ones(len(mstar))[np.where(ids==j)]) for j in range(1,len(bins))]

                        counts[d] += counts_i
                        hist[d]   += np.array(counts_i) / sel_mask['h_frac'][d][m]
                        mstar_mean[d] += [np.sum(mstar[np.where(ids==j)]) for j in range(1,len(bins))]

        bin_width = np.log10(bins[1:])-np.log10(bins[:-1])
        norm = 1 / ( ( self.sim.header['BoxSize'] / self.sim.Cosmology.pars['hubble'] )**3 * bin_width )

        for d in range(draws):
            hist[d] *= norm
            mstar_mean[d] = mstar_mean[d] / counts[d]

        return  {'smf':hist, 'bins':bins, 'mstar':mstar_mean}

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

    def halo_gas_frac(self, mhalo_edges=None, m500_edges=None, sel_mask=None, vmax_sel=False):
        '''
        '''

        presel = np.where(self.m200b>10**mhalo_edges[0,0])[0]
        _m500c = 1e10 * self.sim.fof['halo_m500c'][presel]
        _main_sub = self.sim.fof['halo_firstsub'][presel]
        _mgas  = 1e10 * self.sim.sub['MassType'][:,0][_main_sub]
        _mtot =  1e10 * np.sum(self.sim.sub['MassType'], axis=1)[_main_sub]

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

            f_gas = np.zeros(m500_edges.shape[0])
            m500c_mean = np.zeros(m500_edges.shape[0])

            for m in range(m500_edges.shape[0]):
                m_sel = np.where((_m500c>10**m500_edges[m,0])&(_m500c<10**m500_edges[m,1]))
                f_gas[m] = np.mean( _mgas[m_sel] / _mtot[m_sel] )
                m500c_mean[m]= np.mean( _m500c[m_sel] )

        return {'f_gas':f_gas, 'm500c':m500c_mean}

    def total_gas_frac(self, m500_edges=None, mass_edges=None):
        '''
            Get the gas-fraction in halos all across the simulation
        '''
        main_sub = self.sim.fof['halo_firstsub']
        mgas = self.sim.sub['MassType'][:,0][main_sub] * 1e10
        mtot = np.sum(self.sim.sub['MassType'][main_sub],axis=1) * 1e10
        m500c = 1e10 * self.sim.fof['halo_m500c']

        f_gas = np.zeros(m500_edges.shape[0])
        m500c_mean = np.zeros(m500_edges.shape[0])

        for m in range(m500_edges.shape[0]):
            m_sel = np.where((m500c>10**m500_edges[m,0])&(m500c<10**m500_edges[m,1]))

            f_gas[m] = np.mean( mgas[m_sel] / mtot[m_sel] )

            m500c_mean[m]= np.mean( m500c[m_sel] )

        return  {'f_gas':f_gas, 'm500':m500c_mean}

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
    mstar = f['Subhalo/SubhaloMassType'][:,4]*1e10 / hfac #stellar masses in Msun

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

def camels_gas_frac(id_name, par, nbins=20, hfac=0.6711):
    '''
    '''

    # catalog name
    catalog = '/scratch/fgmaion/CAMELS/1P/1P_p{:d}_'.format(par)+id_name+'/groups_090.hdf5'

    # value of the scale factor
    scale_factor = 1.0

    # open the catalogue
    f = h5py.File(catalog, 'r')

    # read the positions, black hole masses and stellar masses of the subhalos/galaxies
    m500c = f['Group/Group_M_Crit500'][()]*1e10 / hfac  #M500c in log10 of Msun
    main_sub = f['Group/GroupFirstSub'][()]
    mgas = ( f['Subhalo/SubhaloMassType'][:,0]*1e10 )[main_sub] #Mgas in log10 of Msun/h
    mtot = ( f['Subhalo/SubhaloMass'][()]*1e10 )[main_sub] #total mass in log10 of Msun/h

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
