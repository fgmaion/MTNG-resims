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

        array of floats:
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
                spacing = ( len(vratio_m) - 1 ) // Nhalos
                vratio_bins = np.sort(vratio_m)[np.array([i*spacing for i in range(Nhalos+1)])]

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
                # Filter halos in the current mass bin
                in_mass_bin = (_m200b > 10**mhalo_edges[m, 0]) & (_m200b < 10**mhalo_edges[m, 1])
                sel_temp = np.where(in_mass_bin)[0]

                if Nhalos is not None:
                    fof_choice[m] = {d:[] for d in range(draws)}

                    for d in range(draws):
                        # sample those halos randomly
                        fof_temp = np.random.choice(sel_temp, min(Nhalos, len(sel_temp)), replace=False)
                        fof_choice[m][d].extend(fof_temp)

                        halo_frac[d].append( min(Nhalos, len(sel_temp)) / len(sel_temp))
        
#                        fof_choice[m][d] = np.array(fof_choice[m])

                else:
                    fof_temp=sel_temp
                    fof_choice[m] = fof_temp

                    halo_frac = np.ones(len(fof_temp))


        print(time.time()-t0)


        return {'sel':fof_choice, 'h_frac':halo_frac}

    def total_smf(self, nbins=100):

        bins = np.logspace(8, 13, nbins)
        counts     = np.zeros(nbins-1)
        hist       = np.zeros(nbins-1)
        mstar_mean = np.zeros(nbins-1)

        sel = np.log10(1e10 * self.sim.fof['halo_m200b'][self.sim.sub['parent_halo']['index']]) > 11
        mstar = self.sim.sub['MassType'][sel,4] * 1e10 / self.sim.Cosmology.pars['hubble']
        ids = np.digitize(mstar, bins)

        hist = np.array( [np.sum( np.ones(len(mstar))[np.where(ids==i)]) for i in range(1,len(bins))] )
        mstar_mean = np.array([np.sum(mstar[np.where(ids==i)]) for i in range(1,len(bins))])

        bin_width = np.log10(bins[1:])-np.log10(bins[:-1])
        norm = 1 / ( ( self.sim.header['BoxSize'] / self.sim.Cosmology.pars['hubble'] )**3 * bin_width )

        return  {'smf':norm * hist, 'bins':bins, 'mstar':mstar_mean / hist}

    def halo_smf_draws(self, sel_mask=None, nbins=100, draws=1):
        '''
        Function to get the stellar mass function of a selection, binned as a function of halo masses
        
        Parameters
        ----------
        sel_mask: dict
            dictionary containing the selection masks
        nbins: int
            number of bins in log(mhalo)
        draws: int
            number of selections contained in sel_mask
        
        Returns
        -------
        dict:
            dictionary containing the stellar mass function, bin edges and mean stellar mass per halo
        '''
        
        bins = np.logspace(9.5, 12, nbins)
        counts     = {d:np.zeros(nbins-1) for d in range(draws)}
        hist       = {d:np.zeros(nbins-1) for d in range(draws)}
        mstar_mean = {d:np.zeros(nbins-1) for d in range(draws)}

        first = self.sim.fof['halo_firstsub']#[presel]
        nsubs = self.sim.fof['halo_nsubs']#[presel]

        for m in range(len(sel_mask['sel'])):
            for d in range(draws):
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

    def halo_smf(self, Nhalos, h_frac, tree=None, nbins=100, snap=264):
        '''
        Function to get the stellar mass function of a selection, binned as a function of halo masses
        
        Parameters
        ----------
        Nhalos: int
            number of halos in the selection
        h_frac: array
            array containing the halo fractions
        tree: tree object
            tree object
        nbins: int
            number of bins in log(mhalo)
        snap: int
            snapshot number
        
        Returns
        -------
        dict:
            dictionary containing the stellar mass function, bin edges and mean stellar mass per halo
        '''
        
        bins = np.logspace(8, 13, nbins)
        counts     = np.zeros(nbins-1)
        hist       = np.zeros(nbins-1)
        mstar_mean = np.zeros(nbins-1)

        first = self.sim.fof['halo_firstsub']
        nsubs = self.sim.fof['halo_nsubs']

        for m in range(Nhalos):
            mstar = []
 
            all_parents = np.unique(tree.group_nr[m][tree.all_idx[m][snap]])
        
            for j in range(len(all_parents)):
                mstar.extend(self.sim.sub['MassType'][:,4][first[all_parents[j]]:first[all_parents[j]]+nsubs[all_parents[j]]] / self.sim.Cosmology.pars['hubble'])
            
            mstar = 1e10 * np.array(mstar)

            ids = np.digitize(mstar, bins)
            counts_i = [np.sum( np.ones(len(mstar))[np.where(ids==j)]) for j in range(1,len(bins))]

            counts += counts_i
            hist   += np.array(counts_i) / h_frac[m]
            mstar_mean += [np.sum(mstar[np.where(ids==j)]) for j in range(1,len(bins))]

        bin_width = np.log10(bins[1:])-np.log10(bins[:-1])
        norm = 1 / ( ( self.sim.header['BoxSize'] / self.sim.Cosmology.pars['hubble'] )**3 * bin_width )

        hist *= norm
        mstar_mean = mstar_mean / counts

        return  {'smf':hist, 'bins':bins, 'mstar':mstar_mean}

    def halo_gas_frac(self, sel_mask=None, draws=1, nbins=100):
        '''
            Function to get the fraction of gas in a selection, binned as a function of halo masses
        '''

        bins = np.logspace(12.5, 14.5, nbins)

        _m500c = 1e10 * self.sim.fof['halo_m500c'] / self.sim.Cosmology.pars['hubble']
        _r500c = self.sim.fof['halo_r500c'] / self.sim.Cosmology.pars['hubble']
        _main_sub = self.sim.fof['halo_firstsub']
        _mgas  = 1e10 * self.sim.sub['MassType'][:,0] / self.sim.Cosmology.pars['hubble']
        _mtot =  1e10 * np.sum(self.sim.sub['MassType'], axis=1)[_main_sub] / self.sim.Cosmology.pars['hubble']

        counts     = {d:np.zeros(nbins-1) for d in range(draws)}
        weights    = {d:np.zeros(nbins-1) for d in range(draws)}
        fgas       = {d:np.zeros(nbins-1) for d in range(draws)}
        m500c_mean = {d:np.zeros(nbins-1) for d in range(draws)}

        for d in range(draws):
            for m in range(len(sel_mask['sel'])):
                if sel_mask['h_frac'][d][m]!=0:
                    
                    d_sub = np.sqrt( np.sum( (self.sim.sub['pos'] - self.sim.sub['pos'][_main_sub[sel_mask['sel'][m][d]]])**2, axis=1 ) )
                    rsel = np.where( d_sub < _r500c[sel_mask['sel'][m][d]] )[0]
                    
                    mgas = np.sum(_mgas[rsel])
                    mtot = _mtot[sel_mask['sel'][m][d]]
                    m500c = _m500c[sel_mask['sel'][m][d]]

                    ids = np.digitize(m500c, bins)

                    counts_i = np.array([np.sum( np.ones(len(m500c))[np.where(ids==j)]) for j in range(1,len(bins))])
                    counts[d] += counts_i
                    weights[d] += counts_i / sel_mask['h_frac'][d][m] 

                    fgas[d] += np.array([np.sum((mgas/mtot)[np.where(ids==j)]) for j in range(1,nbins)]) / sel_mask['h_frac'][d][m]
                    m500c_mean[d] += [np.sum(m500c[np.where(ids==j)]) for j in range(1,nbins)]

        for d in range(draws):
            fgas[d] /= weights[d]
            m500c_mean[d] /= counts[d]

        return {'f_gas':fgas, 'm500c':m500c_mean, 'counts':counts}

    def halo_SFR(self, mhalo_edges=None, sel_mask=None, vmax_sel=False, draws=1, nbins=100):
        '''
        '''

        bins = np.logspace(8, 13, nbins)

        presel = np.where(self.m200b>10**mhalo_edges[0,0])[0]
        first = self.sim.fof['halo_firstsub'][presel]
        nsubs = self.sim.fof['halo_nsubs'][presel]
        
        counts     = {d:np.zeros(nbins-1) for d in range(draws)}
        weights    = {d:np.zeros(nbins-1) for d in range(draws)}
        sSFR_mean   = {d:np.zeros(nbins-1) for d in range(draws)}
        mstar_mean = {d:np.zeros(nbins-1) for d in range(draws)}

        if vmax_sel is True:
            for d in range(draws):
                for m in range(len(sel_mask['sel'])):
                    if sel_mask['h_frac'][m][d]!=0:
                        for v in range(len(sel_mask['sel'][m][d])):
                            mstar = self.sim.sub['MassType'][:,4][first[sel_mask['sel'][m][d][v]]:first[sel_mask['sel'][m][d][v]]+nsubs[sel_mask['sel'][m][d][v]]]
                            sSFR = self.sim.sub['SFR'][first[sel_mask['sel'][m][d][v]]:first[sel_mask['sel'][m][d][v]]+nsubs[sel_mask['sel'][m][d][v]]]

                            mstar = 1e10 * np.array(mstar)
                            sSFR  = 1e10 * np.array(sSFR) / mstar

                            ids = np.digitize(mstar, bins)
                            counts_i = np.array([np.sum( np.ones(len(mstar))[np.where(ids==i)]) for i in range(1,len(bins))] )
                            counts[d] += counts_i
                            weights[d] += counts_i / sel_mask['h_frac'][m][d][v]

                            sSFR_mean[d] += np.array([np.sum(sSFR[np.where(ids==j)]) for j in range(1,len(bins))]) / sel_mask['h_frac'][m][d][v]
                            mstar_mean[d] += np.array([np.sum(mstar[np.where(ids==j)]) for j in range(1,len(bins))])

        else:
            for d in range(draws):
                for m in range(len(sel_mask['sel'])):
                    if sel_mask['h_frac'][d][m]!=0:

                        mstar = []
                        sSFR  = []
                        for i in range(len(sel_mask['sel'][m][d])):
                            mstar.extend(self.sim.sub['MassType'][:,4][first[sel_mask['sel'][m][d][i]]:first[sel_mask['sel'][m][d][i]]+nsubs[sel_mask['sel'][m][d][i]]])
                            sSFR.extend(self.sim.sub['SFR'][first[sel_mask['sel'][m][d][i]]:first[sel_mask['sel'][m][d][i]]+nsubs[sel_mask['sel'][m][d][i]]])

                        mstar = 1e10 * np.array(mstar)
                        sSFR  = 1e10 * np.array(sSFR) / mstar

                        ids = np.digitize(mstar, bins)

                        counts_i = np.array([np.sum( np.ones(len(mstar))[np.where(ids==j)]) for j in range(1,len(bins))])
                        weights[d] += counts_i / sel_mask['h_frac'][d][m] 
                        counts[d]  += counts_i

                        sSFR_mean[d] += np.array([np.sum(sSFR[np.where(ids==j)]) for j in range(1,len(bins))]) / sel_mask['h_frac'][d][m] 
                        mstar_mean[d] += np.array([np.sum(mstar[np.where(ids==j)]) for j in range(1,len(bins))])

        for d in range(draws):
            sSFR_mean[d] /= weights[d]
            mstar_mean[d] /= counts[d]

        return {'sSFR':sSFR_mean, 'mstar':mstar_mean, 'counts':counts}

    def total_SFR(self, nbins=100):
        '''
        '''

        bins = np.logspace(8, 13, nbins)

        mstar = np.array(1e10 * self.sim.sub['MassType'][:,4])
        sSFR   = np.array(1e10 * self.sim.sub['SFR']) / mstar

        ids = np.digitize(mstar, bins)

        counts = np.array([np.sum( np.ones(len(mstar))[np.where(ids==j)]) for j in range(1,len(bins))])

        sSFR_mean = np.array([np.sum(sSFR[np.where(ids==j)]) for j in range(1,len(bins))]) / counts 
        mstar_mean = np.array([np.sum(mstar[np.where(ids==j)]) for j in range(1,len(bins))]) / counts

        return {'sSFR':sSFR_mean, 'mstar':mstar_mean, 'counts':counts}


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

def metric(m1,m2,v1,v2,eul_dist,a,b,c):

    d_m = np.abs( (1 - m1[...,np.newaxis]/m2)/0.2 )
    d_v = np.abs( (1 - v1[...,np.newaxis]/v2)/0.2 )

    return a*eul_dist + d_m*b + d_v*c

def cross_match(zoom, snap, name=None):
    '''
    Cross-match two catalogs of halos based on their position and properties.
    Parameters
    ----------
    zoom : bacco.Simulation
        The simulation to match with MTNG
    snap : int
        The snapshot number
    name : str, optional
        The name of the output dictionary, by default None
        
    Returns
    -------
    dict
        A dictionary with the following keys:
        - ind: the index of the matched halo in the zoom simulation
        - d: the distance between the matched halos
    '''

    import numpy.ma as ma
    import scipy

    # Load MTNG
    mtng = bacco.utils.load_MTNG(adr="/cosmos_storage/simulations/TNG_Family/MTNG/", snap=snap)
    mtng.fof['halo_pos'][:,0] = (mtng.fof['halo_pos'][:,0] - 125) % 500
    mtng.sub['pos'][:,0] = (mtng.sub['pos'][:,0] - 125) % 500

    # Load halo selection
    with open("/cosmos_storage/home/fgmaion/MTNG-resims/halo_selection/hydro_halo_sel_1pmbin.txt") as f:
        final_sel = []
        for line in f.readlines():
            final_sel.append(int(line.split()[0]))
    final_sel = np.array(final_sel)

    # position matching
    X1 = zoom.fof['halo_pos']
    X2 = mtng.fof['halo_pos'][final_sel]

    kdt = scipy.spatial.KDTree(X1, boxsize=mtng.header['BoxSize'])
    dist, ind = kdt.query(X2, k=100)

    # load halo properties
    M1 = 1e10 * zoom.fof['halo_m200b'][ind]
    M2 = 1e10 * mtng.fof['halo_m200b'][final_sel]

    pos1 = np.transpose(zoom.fof['halo_pos'][ind], (2,0,1))
    pos2 = mtng.fof['halo_pos'][final_sel].T

    vmax_1 = zoom.sub['vmax'][zoom.fof['halo_firstsub'][ind]]
    vmax_2 = mtng.sub['vmax'][mtng.fof['halo_firstsub'][final_sel]]

    d = metric(M2,M1,vmax_2,vmax_1,dist,1,1,0)

    xmatch = np.zeros(len(M1), dtype=int)
    dmatch = np.zeros(len(M1))
    metr = np.zeros(len(M1))

    for i in range(len(M1)):
        metr[i] = d[i].min()
        xmatch[i] = ind[i,np.where(d[i]==metr[i])[0][0]]
        dmatch[i] = dist[i,np.where(d[i]==metr[i])[0][0]]
    if name is None:
        return {'ind':xmatch, 'd':dmatch}
    else:
        return {name: {'ind':xmatch, 'd':dmatch}}

def cross_match_zooms(zoom1, zoom2):
    '''
    Cross-match two catalogs of halos based on their position and properties.
    Parameters
    ----------
    zoom1 : bacco.Simulation
        First zoom
    zoom2: bacco.Simulation
        Second zoom

    Returns
    -------
    dict
        A dictionary with the following keys:
        - ind: the index of the matched halo in the zoom simulation
        - d: the distance between the matched halos
    '''

    import numpy.ma as ma
    import scipy

    sel = np.where(zoom1.fof['halo_m200b']>0)[0]

    # position matching
    X1 = zoom1.fof['halo_pos'][sel]
    X2 = zoom2.fof['halo_pos']

    kdt = scipy.spatial.KDTree(X1, boxsize=zoom1.header['BoxSize'])
    dist, ind = kdt.query(X2, k=100)

    # load halo properties
    M1 = 1e10 * zoom1.fof['halo_m200b'][sel][ind]
    M2 = 1e10 * zoom2.fof['halo_m200b']

    pos1 = np.transpose(zoom1.fof['halo_pos'][sel][ind], (2,0,1))
    pos2 = np.transpose(zoom2.fof['halo_pos'].T)

    vmax_1 = zoom1.sub['vmax'][zoom1.fof['halo_firstsub'][sel][ind]]
    vmax_2 = zoom2.sub['vmax'][zoom2.fof['halo_firstsub']]

    d = metric(M2,M1,vmax_2,vmax_1,dist,5,1,1)

    xmatch = np.zeros(len(M1), dtype=int)
    dmatch = np.zeros(len(M1))
    metr = np.zeros(len(M1))

    for i in range(len(M1)):
        metr[i] = d[i].min()
        xmatch[i] = ind[i,np.where(d[i]==metr[i])[0][0]]
        dmatch[i] = dist[i,np.where(d[i]==metr[i])[0][0]]
    return {'ind':xmatch, 'd':dmatch}

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

# CAMELS Functions

def camels_stellar_mf(sim_cat, id_name=None, par=None, num=None, nbins=100, baseline=False, boxsize=25, hfac=0.6711):
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
    
    if sim_cat == "1P":

        # catalog name
        if baseline:
            catalog = '/scratch/fgmaion/CAMELS/1P/1P_0/groups_090.hdf5'
        else:
            assert id_name is not None
            assert par is not None

            catalog = '/scratch/fgmaion/CAMELS/1P/1P_p{:d}_'.format(par)+id_name+'/groups_090.hdf5'
    
    elif sim_cat == 'LH':
        assert num is not None

        # catalog name
        catalog = '/scratch/fgmaion/CAMELS/LH/LH_{:d}'.format(num)+'/groups_090.hdf5'

    elif sim_cat == 'CV':
        assert num is not None

        # catalog name
        catalog = '/scratch/fgmaion/CAMELS/CV/CV_{:d}'.format(num)+'/groups_090.hdf5'


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

def camels_sSFR(id_name, par, nbins=20, hfac=0.6711):

    bins = np.logspace(8, 13, nbins)
    
    catalog = '/scratch/fgmaion/CAMELS/1P/1P_p{:d}_'.format(par)+id_name+'/groups_090.hdf5'

    # value of the scale factor
    scale_factor = 1.0

    # open the catalogue
    f = h5py.File(catalog, 'r')

    # read the positions, black hole masses and stellar masses of the subhalos/galaxies
    _mstar = 1e10 * f['Subhalo/SubhaloMassType'][:,4] / hfac #stellar masses in Msun
    _sSFR = 1e10 * f['Subhalo/SubhaloSFR'][:] / _mstar

    # close file
    f.close()

    nbins=30
    bins = np.logspace(8, 13, nbins)

    sSFR = np.zeros(nbins-1)
    mstar_mean = np.zeros(nbins-1)

    for m in range(nbins-1):
        m_sel = np.where(( _mstar>bins[m])&( _mstar<bins[m+1]) )

        sSFR[m] = np.mean( _sSFR[m_sel] )
        mstar_mean[m]= np.mean( _mstar[m_sel] )

    return {'sSFR':sSFR, 'mstar':mstar_mean}

def camels_get_LH_pars(num=None):
    # catalog name
    catalog = '/scratch/fgmaion/CAMELS/LH/LH_{:d}/groups_090.hdf5'.format(num)

    # open the catalogue
    f = h5py.File(catalog, 'r')

    par1 = f['Parameters'].attrs['WindEnergyIn1e51erg']
    par2 = f['Parameters'].attrs['VariableWindVelFactor']
    par3 = f['Parameters'].attrs['RadioFeedbackFactor']
    par4 = f['Parameters'].attrs['RadioFeedbackReiorientationFactor']
    
    # close file
    f.close()

    return np.asarray([par1, par2, par3, par4])

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

def read_central_xmatch(sim_name):
    mtng_id = []
    zoom_id = []

    dx = []
    dy = []
    dz = []

    with open("/lscratch/kwalsen/xmatch/264/"+sim_name+"/selection_xmatch_deltas.csv", 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i==0:
                continue
            line = line.strip('\n').split(',')
            mtng_id.append(int(line[1]))

            try:
                zoom_id.append(int(line[2]))
                dx.append(float(line[3]))
                dy.append(float(line[4]))
                dz.append(float(line[5]))
            except:
                zoom_id.append(-1)
                dx.append(0)
                dy.append(0)
                dz.append(0)
                print("failed for line: {}".format(line))

    return {'mtng_id': np.array(mtng_id), 'zoom_id': np.array(zoom_id), 'dx': np.array(dx), 'dy': np.array(dy), 'dz': np.array(dz)}

def metric(m1,m2,v1,v2,dist,a,b,c):

    #d_m = np.abs(np.log10(m1[...,np.newaxis]/m2))
    #d_v = np.abs(v1[...,np.newaxis]-v2)

    d_m = np.abs( (1 - m1[...,np.newaxis]/m2)/0.2 )
    d_v = np.abs( (1 - v1[...,np.newaxis]/v2)/0.2 )

    return dist*a + d_m*b + d_v*c

def pars(i, mstar):

    arr = np.vstack( (mstar, np.ones(len(mstar)) * wind_en[i],\
                        np.ones(len(mstar)) * wind_vel[i],\
                        np.ones(len(mstar)) * rho_rec[i],\
                        np.ones(len(mstar)) * sf_ts[i],\
                        np.ones(len(mstar)) * ef_kin[i],\
                        np.ones(len(mstar)) * ef_high[i],\
                        np.ones(len(mstar)) * f_re[i])).T

    return arr

def get_zoom_smf(zoom, snap=264, Nbins=50):

    # BE CAREFUL. Right now this works for snap=264, but I am not sure if it would work for other redshifts
    mtng = bacco.utils.load_MTNG(adr="/cosmos_storage/simulations/TNG_Family/MTNG/", snap=264)
    mtng.fof['halo_pos'][:,0] = ( mtng.fof['halo_pos'][:,0] - 125 ) % 500
    
    xmatch = cross_match(zoom, snap=264)

    m200b = np.log10(1e10 * mtng.fof['halo_m200b'])
    
    # load the halo-selection in the fiducial MTNG
    with open("/cosmos_storage/home/fgmaion/MTNG-resims/halo_selection/hydro_halo_sel_1pmbin.txt") as f:
        final_sel = []
        for line in f.readlines():
            final_sel.append(int(line.split()[0]))
    final_sel = np.array(final_sel)

    mbins = np.concatenate(
        (np.arange(11, 11.5, 0.0025),
        np.arange(11.5, 12.5, 0.005),
        np.arange(12.5, 13.5, 0.025),
        np.arange(13.5, 15.01, 0.125))
    )

    h_frac = np.zeros(len(final_sel))
    for m in range(len(mbins)-1):
        h_frac[m] = np.where(( m200b[final_sel]>=mbins[m]) & ( m200b[final_sel]<mbins[m+1]))[0].shape[0] / \
                 np.where(( m200b>=mbins[m]) & ( m200b<mbins[m+1]))[0].shape[0]

    zoom_split = split_halos(zoom)

    zoom_sel = {}
    zoom_sel['sel'] = xmatch['ind'][:,np.newaxis,np.newaxis]
    zoom_sel['h_frac'] = h_frac[np.newaxis, :]

    zoom_smf = zoom_split.halo_smf_draws(sel_mask=zoom_sel, nbins=Nbins, draws=1)

    return zoom_smf



