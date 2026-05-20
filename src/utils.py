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
        # self.r200b = self.sim.fof['halo_r200b']

        G_newton = 4.3009172706e-9 #Mpc/M_sun * (km/s)**2

        # self.halo_v200 = np.sqrt(G_newton*self.m200b/self.r200b)
        # self.vmax_v200 = self.halo_vmax / self.halo_v200


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

    def halo_sel(self, mhalo_edges=None, Nhalos=None, draws=1):
        '''
        Function to sample halos in mass-bins

        array of floats:
        Contains the edges of the mass-bin in which we wish to select our halo

        bool:vmax_sel
        Whether to split the selection not only by mass, but by concentration as well
        '''

        _m200b = self.m200b

        fof_choice = {}
        halo_frac = {d:[] for d in range(draws)}
        
        for m in range(mhalo_edges.shape[0]):
            # Filter halos in the current mass bin
            in_mass_bin = (_m200b >= 10**mhalo_edges[m, 0]) & (_m200b < 10**mhalo_edges[m, 1])
            sel_temp = np.where(in_mass_bin)[0]

            if Nhalos is None:
                # No sampling: return all halos in this bin, identical across draws
                fof_choice[m] = {d: sel_temp for d in range(draws)}
                for d in range(draws):
                    halo_frac[d].append(1.0)
                continue

            fof_choice[m] = {d: np.array([], dtype=int) for d in range(draws)}

            n_draw = min(Nhalos, len(sel_temp))
            for d in range(draws):
                fof_choice[m][d] = np.random.choice(sel_temp, n_draw, replace=False)
                halo_frac[d].append(n_draw / len(sel_temp))

        return {'sel':fof_choice, 'h_frac':halo_frac}

    def total_smf(self, nbins=100):

        bins = np.logspace(9, 12.5, nbins)
        counts     = np.zeros(nbins-1)
        hist       = np.zeros(nbins-1)
        mstar_mean = np.zeros(nbins-1)

        sel = np.log10(1e10 * self.sim.fof['halo_m200b'][self.sim.sub['parent_halo']['index']]) > 9
        mstar = self.sim.sub['MassType'][sel,4] * 1e10 / self.sim.Cosmology.pars['hubble']
        ids = np.digitize(mstar, bins)

        hist = np.array( [np.sum( np.ones(len(mstar))[np.where(ids==i)]) for i in range(1,len(bins))] )
        mstar_mean = np.array([np.sum(mstar[np.where(ids==i)]) for i in range(1,len(bins))])

        bin_width = np.log10(bins[1:])-np.log10(bins[:-1])
        norm = 1 / ( ( self.sim.header['BoxSize'] / self.sim.Cosmology.pars['hubble'] )**3 * bin_width )

        return  {'smf':norm * hist, 'bins':bins, 'mstar':mstar_mean / hist}

    def get_mstar_30kpc(self, isub):
        """
            Calculates stellar mass within 30 kpc for a given subhalo ID.
            Stellar mass is outputed in M_sun units.
        """

        arr_ids = self.sim.get_halo_star_particles(isub, sDM=False, key='pos_ids', mode="subhalo")    
        d_part  = self.sim.get_halo_star_particles(isub, sDM=False, key='pos', mode="subhalo", relative=True) / self.sim.Cosmology.pars['hubble'] # distances are in kpc

        d = np.sqrt(np.sum(d_part**2, axis=1)) # in Mpc

        sel = d < 30/1000 # get particles within 30 kpc
        mstar = np.sum( self.sim.stars['mass'][arr_ids[sel]] ) / self.sim.Cosmology.pars['hubble'] # in Msun

        # Apply Eddington bias correction from Behroozi+19
        z = 0
        sigma_eb = min(0.07 + 0.071*z, 0.3)
        mstar *= 10**( np.random.normal(loc=0, scale=sigma_eb) )

        return mstar

    def get_mstar_30kpc_vec(self, isub_vector):
            """
            Calculates stellar mass within 30 kpc for a vector of subhalo IDs.
            """
            vectorized_func = np.vectorize(self.get_mstar_30kpc, otypes=[np.float64])
            
            # Now call the vectorized function, passing only the vector of inputs
            return vectorized_func(isub_vector)

    def get_mstar_2halfrad(self, isub):
        """
            Calculates stellar mass within the half-mass radius for a given subhalo ID.
            Stellar mass is outputed in M_sun units.
        """

        r_star_half = self.sim.sub['HalfmassRadType'][:,4] / self.sim.header['HubbleParam']

        arr_ids = self.sim.get_halo_star_particles(isub, sDM=False, key='pos_ids', mode="subhalo")    
        d_part  = self.sim.get_halo_star_particles(isub, sDM=False, key='pos', mode="subhalo", relative=True) / self.sim.Cosmology.pars['hubble'] # distances are in kpc

        d = np.sqrt(np.sum(d_part**2, axis=1)) # in Mpc

        sel = d < 2 * r_star_half[isub] # get particles within 2*half-mass radius
        mstar_rhalf = 1e10 * np.sum( self.sim.stars['mass'][arr_ids[sel]] ) / self.sim.Cosmology.pars['hubble'] # in Msun

        return mstar_rhalf

    def get_mstar_2halfrad_vec(self, isub_vector):
        """
        Calculates stellar mass within the half-mass radius for a vector of subhalo IDs.
        """
        vectorized_func = np.vectorize(self.get_mstar_2halfrad, otypes=[np.float64])
        
        # Now call the vectorized function, passing only the vector of inputs
        return vectorized_func(isub_vector)

    def rhalf_m2half(self, sel_mask=None, nbins=100):
        
        bins = np.logspace(8, 13, nbins)
        counts      = np.zeros(nbins-1)
        rhalf_mean  = np.zeros(nbins-1)
        m2half_mean = np.zeros(nbins-1)

        first = self.sim.fof['halo_firstsub']
        nsubs = self.sim.fof['halo_nsubs']
        
        r_star_half = self.sim.sub['HalfmassRadType'][:,4] / self.sim.header['HubbleParam']
        rhalf_total = []
        m2half_total = []
        if sel_mask is not None:
            for m in range(len(sel_mask['sel'])):
                if sel_mask['h_frac'][0][m]!=0:
                    rhalf = []
                    m2half = []
                    for i in range(len(sel_mask['sel'][m][0])):
                        sub_indices = np.arange(first[sel_mask['sel'][m][0][i]],first[sel_mask['sel'][m][0][i]]+nsubs[sel_mask['sel'][m][0][i]])
                        rhalf.extend(r_star_half[sub_indices])
                        rhalf_total.extend(r_star_half[sub_indices])

                        m2half_ = self.get_mstar_2halfrad_vec( sub_indices )
                        m2half.extend( m2half_ )
                        m2half_total.extend(m2half_)
                        
                    m2half = np.array(m2half)
                    rhalf  = np.array(rhalf)

                    ids = np.digitize(m2half, bins)
                    counts_i = [np.sum( np.ones(len(m2half))[np.where(ids==j)]) for j in range(1,len(bins))]
                    counts += counts_i / sel_mask['h_frac'][0][m]

                    rhalf_mean  += np.array([np.sum(rhalf[np.where(ids==j)]) for j in range(1,len(bins))]) / sel_mask['h_frac'][0][m]
                    m2half_mean += np.array([np.sum(m2half[np.where(ids==j)]) for j in range(1,len(bins))]) / sel_mask['h_frac'][0][m]

            rhalf_mean = rhalf_mean / counts
            m2half_mean = m2half_mean / counts

        else:
            rhalf = r_star_half
            m2half_ = self.get_mstar_2halfrad_vec(np.arange(np.sum(self.sim.fof['halo_nsubs'])))

            ids = np.digitize(m2half, bins)
            counts_i = [np.sum( np.ones(len(m2half))[np.where(ids==j)]) for j in range(1,len(bins))]
            counts += counts_i

            rhalf_mean  += [np.sum(rhalf[np.where(ids==j)]) for j in range(1,len(bins))]
            m2half_mean += [np.sum(m2half[np.where(ids==j)]) for j in range(1,len(bins))]

            rhalf_mean = rhalf_mean / counts
            m2half_mean = m2half_mean / counts

        return  {'rhalf_mean':rhalf_mean, 'm2half_mean':m2half_mean, 'rhalf':rhalf_total, 'm2half':m2half_total, 'counts':counts}

    def sSFR_mstar(self, sel_mask=None, nbins=100):
        
        bins = np.logspace(8, 13, nbins)
        counts      = np.zeros(nbins-1)
        sSFR_mean  = np.zeros(nbins-1)
        mstar_mean = np.zeros(nbins-1)

        first = self.sim.fof['halo_firstsub']
        nsubs = self.sim.fof['halo_nsubs']
        
        sSFR_array = self.sim.sub['SFR'] / ( 1e10 * self.sim.sub['MassType'][:,4] / self.sim.header['HubbleParam'] )
        if sel_mask is not None:
            for m in range(len(sel_mask['sel'])):
                if sel_mask['h_frac'][0][m]!=0:
                    sSFR = []
                    mstar = []
                    for i in range(len(sel_mask['sel'][m][0])):
                        sub_indices = np.arange(first[sel_mask['sel'][m][0][i]],first[sel_mask['sel'][m][0][i]]+nsubs[sel_mask['sel'][m][0][i]])
                        sSFR.extend(sSFR_array[sub_indices])

                        mstar_ = 1e10 * self.get_mstar_30kpc_vec( sub_indices )
                        mstar.extend( mstar_ )
                        
                    mstar = np.array(mstar)
                    sSFR  = np.array(sSFR)

                    ids = np.digitize(mstar, bins)
                    counts_i = [np.sum( np.ones(len(mstar))[np.where(ids==j)]) for j in range(1,len(bins))]
                    counts += counts_i / sel_mask['h_frac'][0][m]

                    sSFR_mean  += np.array([np.sum(sSFR[np.where(ids==j)]) for j in range(1,len(bins))]) / sel_mask['h_frac'][0][m]
                    mstar_mean += np.array([np.sum(mstar[np.where(ids==j)]) for j in range(1,len(bins))]) / sel_mask['h_frac'][0][m]

            sSFR_mean = sSFR_mean / counts
            mstar_mean = mstar_mean / counts

        return  {'sSFR_mean':sSFR_mean, 'mstar_mean':mstar_mean, 'counts':counts}

    def smhm_ratio(self, sel_mask=None, nbins=100):
        
        bins = np.logspace(11, 15, nbins)
        counts      = np.zeros(nbins-1)
        smhm_mean  = np.zeros(nbins-1)
        m200c_mean = np.zeros(nbins-1)

        first = self.sim.fof['halo_firstsub']
        nsubs = self.sim.fof['halo_nsubs']
        
        if sel_mask is not None:
            for m in range(len(sel_mask['sel'])):
                if sel_mask['h_frac'][0][m]!=0:
                    smhm = []
                    m200c = []
                    for i in range(len(sel_mask['sel'][m][0])):
                        central_index = first[sel_mask['sel'][m][0][i]]
                        
                        mstar_30 = 1e10 * self.get_mstar_30kpc_vec( central_index )
                        m200c_ = 1e10 * self.sim.fof['halo_m200c'][sel_mask['sel'][m][0][i]] / self.sim.Cosmology.pars['hubble']

                        smhm.append( mstar_30 / m200c_ )
                        m200c.append( m200c_ )
                        
                    m200c = np.array(m200c)
                    smhm  = np.array(smhm)

                    ids = np.digitize(m200c, bins)
                    counts_i = [np.sum( np.ones(len(m200c))[np.where(ids==j)]) for j in range(1,len(bins))]
                    counts += counts_i / sel_mask['h_frac'][0][m]

                    smhm_mean  += np.array([np.sum(smhm[np.where(ids==j)]) for j in range(1,len(bins))]) / sel_mask['h_frac'][0][m]
                    m200c_mean += np.array([np.sum(m200c[np.where(ids==j)]) for j in range(1,len(bins))]) / sel_mask['h_frac'][0][m]

            smhm_mean = smhm_mean / counts
            m200c_mean = m200c_mean / counts

        return  {'smhm_mean':smhm_mean, 'm200c_mean':m200c_mean}

    def bh_mstar(self, sel_mask=None, nbins=100):
        
        bins       = np.logspace(8, 13, nbins)
        counts     = np.zeros(nbins-1)
        mbh_mean   = np.zeros(nbins-1)
        mstar_mean = np.zeros(nbins-1)

        first = self.sim.fof['halo_firstsub']
        nsubs = self.sim.fof['halo_nsubs']
        
        if sel_mask is not None:
            for m in range(len(sel_mask['sel'])):
                if sel_mask['h_frac'][0][m]!=0:
                    mbh = []
                    mstar = []
                    for i in range(len(sel_mask['sel'][m][0])):
                        sub_indices = np.arange(first[sel_mask['sel'][m][0][i]],first[sel_mask['sel'][m][0][i]]+nsubs[sel_mask['sel'][m][0][i]])
                        
                        mbh_ = 1e10 * self.sim.sub['BHMass'][sub_indices] / self.sim.Cosmology.pars['hubble']
                        mstar_ = self.get_mstar_2halfrad_vec(sub_indices)

                        mbh.extend( mbh_ )
                        mstar.extend( mstar_ )
                        
                    mbh = np.array(mbh)
                    mstar = np.array(mstar)

                    ids = np.digitize(mstar, bins)
                    counts_i = [np.sum(np.ones(len(mstar))[np.where(ids==j)]) for j in range(1,len(bins))]
                    counts += counts_i / sel_mask['h_frac'][0][m]

                    mbh_mean  += np.array([np.sum(mbh[np.where(ids==j)]) for j in range(1,len(bins))]) / sel_mask['h_frac'][0][m]
                    mstar_mean += np.array([np.sum(mstar[np.where(ids==j)]) for j in range(1,len(bins))]) / sel_mask['h_frac'][0][m]

            mbh_mean = mbh_mean / counts
            mstar_mean = mstar_mean / counts

        return  {'mbh_mean':mbh_mean, 'mstar_mean':mstar_mean}


    def bh_mf(self, sel_mask=None, nbins=100):
        '''
        Function to get the black-hole mass-function of a simulation
        
        Parameters
        ----------
        sel_mask: dict
            dictionary containing the selection masks
        nbins: int
            number of bins in log(mhalo)
        
        Returns
        -------
        dict:
            dictionary containing the stellar mass function, bin edges and mean stellar mass per halo
        '''
        
        bins = np.logspace(6, 10.5, nbins)
        counts     = np.zeros(nbins-1)
        hist       = np.zeros(nbins-1)
        mbh_mean = np.zeros(nbins-1)

        first = self.sim.fof['halo_firstsub']
        nsubs = self.sim.fof['halo_nsubs']

        for m in range(len(sel_mask['sel'])):
            if sel_mask['h_frac'][0][m]!=0:
                mbh = []
                for i in range(len(sel_mask['sel'][m][0])):
                    mbh.extend(self.sim.sub['BHMass'][first[sel_mask['sel'][m][0][i]]:first[sel_mask['sel'][m][0][i]]+nsubs[sel_mask['sel'][m][0][i]]] / self.sim.Cosmology.pars['hubble'])
                
                mbh = 1e10 * np.array(mbh)

                ids = np.digitize(mbh, bins)
                counts_i = [np.sum( np.ones(len(mbh))[np.where(ids==j)]) for j in range(1,len(bins))]

                counts += counts_i
                hist   += np.array(counts_i) / sel_mask['h_frac'][0][m]
                mbh_mean += [np.sum(mbh[np.where(ids==j)]) for j in range(1,len(bins))]

        bin_width = np.log10(bins[1:])-np.log10(bins[:-1])
        norm = 1 / ( ( self.sim.header['BoxSize'] / self.sim.Cosmology.pars['hubble'] )**3 * bin_width )

        hist *= norm
        mbh_mean = mbh_mean / counts

        return  {'bhmf':hist, 'bins':bins, 'mbh':mbh_mean}

    def halo_smf_draws(self, sel_mask=None, nbins=100, draws=1, m_30kpc=False):
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
        
        bins = np.logspace(9, 12.5, nbins)
        counts     = {d:np.zeros(nbins-1) for d in range(draws)}
        hist       = {d:np.zeros(nbins-1) for d in range(draws)}
        mstar_mean = {d:np.zeros(nbins-1) for d in range(draws)}

        first = self.sim.fof['halo_firstsub']
        nsubs = self.sim.fof['halo_nsubs']

        for m in range(len(sel_mask['sel'])):
            for d in range(draws):
                if sel_mask['h_frac'][d][m]!=0.0:
                    mstar = []
                    for i in range(len(sel_mask['sel'][m][d])):
                        if m_30kpc is True:
                            sub_indices = np.arange(first[sel_mask['sel'][m][d][i]],first[sel_mask['sel'][m][d][i]]+nsubs[sel_mask['sel'][m][d][i]])
                            mstar_ = self.get_mstar_30kpc_vec( sub_indices )
                            mstar.extend( mstar_ )
                        else:
                            start = first[sel_mask['sel'][m][d][i]]
                            stop = start + nsubs[sel_mask['sel'][m][d][i]]
                            mstar.extend(self.sim.sub['MassType'][:,4][start:stop] / self.sim.Cosmology.pars['hubble'])
                    
                    mstar = 1e10 * np.array(mstar)

                    ids = np.digitize(mstar, bins)
                    counts_i = np.array([np.sum( np.ones(len(mstar))[np.where(ids==j)]) for j in range(1,len(bins))])  / sel_mask['h_frac'][d][m]

                    counts[d] += counts_i
                    hist[d]   += np.array(counts_i)
                    mstar_mean[d] += np.array([np.sum(mstar[np.where(ids==j)]) for j in range(1,len(bins))]) / sel_mask['h_frac'][d][m]

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

    def halo_gas_frac_v2(self, sel_mask=None, draws=1, nbins=100):
        '''
            Function to get the fraction of gas in a selection, binned as a function of halo masses
        '''

        bins = np.logspace(12., 15., nbins)

        _m500c = 1e10 * self.sim.fof['halo_m500c'] / self.sim.Cosmology.pars['hubble']
        _r500c = self.sim.fof['halo_r500c'] / self.sim.Cosmology.pars['hubble']

        counts     = {d:np.zeros(nbins-1) for d in range(draws)}
        weights    = {d:np.zeros(nbins-1) for d in range(draws)}
        fgas       = {d:np.zeros(nbins-1) for d in range(draws)}
        m500c_mean = {d:np.zeros(nbins-1) for d in range(draws)}

        for d in range(draws):
            for m in range(len(sel_mask['sel'])):
                if sel_mask['h_frac'][d][m]!=0:
                    # Get particles in halo
                    d_gas = self.sim.get_halo_particles(ihalo=sel_mask['sel'][m][d][0], ptype=0, relative=True) / self.sim.Cosmology.pars['hubble']
                    m_gas = self.sim.get_halo_particles(ihalo=sel_mask['sel'][m][d][0], ptype=0, key='mass')

                    # Restrict to r500c
                    gas_sel = np.sqrt( np.sum( d_gas**2, axis=1 ) ) < _r500c[sel_mask['sel'][m][d]]
                    mgas_500 = 1e10 * np.sum(m_gas[gas_sel]) / self.sim.Cosmology.pars['hubble']

                    m500c = _m500c[sel_mask['sel'][m][d]]

                    ids = np.digitize(m500c, bins)

                    counts_i = np.array([np.sum( np.ones(len(m500c))[np.where(ids==j)]) for j in range(1,len(bins))])
                    counts[d] += counts_i
                    weights[d] += counts_i / sel_mask['h_frac'][d][m] 

                    fgas[d] += np.array([np.sum((mgas_500/m500c)[np.where(ids==j)]) for j in range(1,nbins)]) / sel_mask['h_frac'][d][m]
                    m500c_mean[d] += [np.sum(m500c[np.where(ids==j)]) for j in range(1,nbins)]

        for d in range(draws):
            fgas[d] /= weights[d]
            m500c_mean[d] /= counts[d]

        return {'f_gas':fgas, 'm500c':m500c_mean, 'counts':counts}


    def lite_mtng_gas_frac(self, sel_mask=None, draws=1, nbins=100, red_fac=64):
        """
        Compute the gas fraction as a function of M500c using a spatial tree query
        on a diluted MTNG snapshot.

        Mirrors halo_gas_frac_v2 but does NOT rely on FoF/Subfind particle offsets,
        since those are not consistent with a post-hoc diluted snapshot. Instead,
        for each selected halo we query all diluted gas particles within r500c
        using a single cKDTree built once over the full diluted gas catalogue.

        The gas masses recovered from the diluted snapshot are scaled up by
        `red_fac` to compensate for the dilution. This assumes the dilution kept
        ~1/red_fac of the gas particles on average, which is the case both for
        `np.random.randint` sampling (with replacement, average rate 1/red_fac)
        and for `ID % red_fac == 0` selection (deterministic rate 1/red_fac).

        :param sel_mask: dictionary with at least the keys 'sel' (list of halo
                        index arrays per mass-bin) and 'h_frac' (per-draw
                        selection fractions per mass-bin), as in halo_gas_frac_v2
        :type sel_mask: dict
        :param draws: number of independent draws over the selection
        :type draws: int
        :param nbins: number of M500c bin edges (so nbins-1 bins)
        :type nbins: int
        :param red_fac: dilution factor of the snapshot (e.g. 8 for the diluted
                        MTNG produced by the dilution script)
        :type red_fac: int

        :return: dict with keys 'f_gas', 'm500c', 'counts', each a dict over
                draws, exactly as halo_gas_frac_v2.
        :rtype: dict
        """
        from scipy.spatial import cKDTree

        bins = np.logspace(12.0, 14.5, nbins)
        h = self.sim.Cosmology.pars['hubble']

        # ------------------------------------------------------------------
        # Halo properties (same conventions as halo_gas_frac_v2)
        # ------------------------------------------------------------------
        _m500c = 1e10 * self.sim.fof['halo_m500c'] / h           # Msun
        _r500c = self.sim.fof['halo_r500c'] / h                  # Mpc
        _hpos = self.sim.fof['halo_pos'] / h                     # Mpc

        # ------------------------------------------------------------------
        # Build the spatial index once over the diluted gas particles.
        # cKDTree handles periodic boundaries natively when given boxsize.
        # All inputs to the tree must be in the same units (here: Mpc).
        # ------------------------------------------------------------------
        boxsize = self.sim.header['BoxSize'] / h                 # Mpc
        gas_pos = np.mod(self.sim.gas['pos'], self.sim.header['BoxSize']) / h                        # Mpc
        gas_mass = self.sim.gas['mass']                          # 1e10 Msun/h units

        print("Now Building the cKDTree for gas particles (this may take a while)...")
        tree = cKDTree(gas_pos, boxsize=boxsize)
        print("Finished")

        # ------------------------------------------------------------------
        # Output containers, identical shape to halo_gas_frac_v2
        # ------------------------------------------------------------------
        counts = {d: np.zeros(nbins - 1) for d in range(draws)}
        weights = {d: np.zeros(nbins - 1) for d in range(draws)}
        fgas = {d: np.zeros(nbins - 1) for d in range(draws)}
        m500c_mean = {d: np.zeros(nbins - 1) for d in range(draws)}

        # ------------------------------------------------------------------
        # Main loop over draws and mass-bin selections
        # ------------------------------------------------------------------
        for d in range(draws):
            for m in range(len(sel_mask['sel'])):
                if sel_mask['h_frac'][d][m] == 0:
                    continue

                ihalo = sel_mask['sel'][m][d][0]

                center = _hpos[ihalo]
                r500 = _r500c[ihalo]

                # Particles in a sphere of r500c around the halo centre.
                # Returns a list of indices into gas_pos / gas_mass.
                idx = tree.query_ball_point(center, r500)

                # Diluted gas mass inside r500c, rescaled to recover the
                # full-resolution mass. Same final units as halo_gas_frac_v2:
                # Msun (1e10 factor + /h to convert from 1e10 Msun/h).
                mgas_500 = 1e10 * red_fac * np.sum(gas_mass[idx]) / h

                m500c = _m500c[sel_mask['sel'][m][d]]

                ids = np.digitize(m500c, bins)

                counts_i = np.array(
                    [np.sum(np.ones(len(m500c))[np.where(ids == j)])
                    for j in range(1, len(bins))]
                )
                counts[d] += counts_i
                weights[d] += counts_i / sel_mask['h_frac'][d][m]

                fgas[d] += np.array(
                    [np.sum((mgas_500 / m500c)[np.where(ids == j)])
                    for j in range(1, nbins)]
                ) / sel_mask['h_frac'][d][m]

                m500c_mean[d] += [
                    np.sum(m500c[np.where(ids == j)])
                    for j in range(1, nbins)
                ]

        for d in range(draws):
            # Avoid division-by-zero in empty bins; leave them as 0.
            nz = weights[d] > 0
            fgas[d][nz] /= weights[d][nz]
            nz = counts[d] > 0
            m500c_mean[d][nz] /= counts[d][nz]

        return {'f_gas': fgas, 'm500c': m500c_mean, 'counts': counts}

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

# def metric(m1,m2,v1,v2,eul_dist,fbad,a,b,c,d):

#     d_m = np.abs( (1 - m1[...,np.newaxis]/m2)/0.2 )
#     d_v = np.abs( (1 - v1[...,np.newaxis]/v2)/0.2 )

#     return a*eul_dist + d_m*b + d_v*c + (fbad/0.01)*d

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

    import scipy

    if name is not None:
        try:
            load_match = np.load("/cosmos_storage/home/fgmaion/MTNG-resims/halo_selection/cross_match_{}.npy".format(name), allow_pickle=True)[0]
            return load_match
        except:
            pass

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
    M_zoom = 1e10 * zoom.fof['halo_m200b'][ind]
    M_mtng = 1e10 * mtng.fof['halo_m200b'][final_sel]

    v_zoom = zoom.fof['halo_vel'][ind,:]
    v_mtng = mtng.fof['halo_vel'][final_sel,:]

    cos = np.sum(v_zoom * v_mtng[:,np.newaxis,:], axis=2) / ( np.linalg.norm(v_zoom, axis=2) * np.linalg.norm(v_mtng, axis=1)[:,np.newaxis] )
    v_ratio = np.linalg.norm(v_zoom, axis=2) / np.linalg.norm(v_mtng, axis=1)[:,np.newaxis]

    d = metric(M_mtng, M_zoom, cos, v_ratio, dist, 1, 1, 1, 1)

    xmatch = np.zeros(len(M_zoom), dtype=int)
    dmatch = np.zeros(len(M_zoom))
    metr = np.zeros(len(M_zoom))

    for i in range(len(final_sel)):
        metr[i] = d[i].min()
        xmatch[i] = ind[i,np.where(d[i]==metr[i])[0][0]]
        dmatch[i] = dist[i,np.where(d[i]==metr[i])[0][0]]

    if name is None:
        return {'ind':xmatch, 'd':dmatch}
    else:
        np.save("/cosmos_storage/home/fgmaion/MTNG-resims/cross-match/cross_match_{}.npy".format(name), [{'ind':xmatch, 'd':dmatch}])
        return {'ind':xmatch, 'd':dmatch}

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

def camels_gas_frac(id_name, par, nbins=10, hfac=0.6711):
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

    bins = np.logspace(12.5, 14.5, nbins)
    
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

def metric(m1, m2, cos, v_ratio, dist, a, b, c, d):
    d_m = np.abs((1 - m1[..., np.newaxis]/m2) / 0.2)
    d_v = np.abs((1 - v_ratio) / 0.2)
    d_theta = (1-cos)/0.2

    if np.any(m2 < 1e11):
        penalty = 10
    else:
        penalty=0

    return dist*a + d_m*b + d_v*c + d_theta*d + penalty

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

def get_mtng_bhmf(mtng, nbins=20):
    
    bins = np.logspace(6, 10.5, nbins)

    mbh = mtng.sub['BHMass'] / mtng.Cosmology.pars['hubble']
    mbh = 1e10 * np.array(mbh)

    ids = np.digitize(mbh, bins)
    counts = [np.sum( np.ones(len(mbh))[np.where(ids==j)]) for j in range(1,len(bins))]
    hist   = np.array(counts)

    mbh_mean = np.array([np.sum(mbh[np.where(ids==j)]) for j in range(1,len(bins))])

    bin_width = np.log10(bins[1:])-np.log10(bins[:-1])
    norm = 1 / ( ( mtng.header['BoxSize'] / mtng.Cosmology.pars['hubble'] )**3 * bin_width )

    hist *= norm
    mbh_mean = mbh_mean / counts

    return  {'bhmf':hist, 'bins':bins, 'mbh':mbh_mean}

def lite_mtng_smf_draws(self, sel_mask=None, nbins=100, draws=1, m_30kpc=False,
                        red_fac=64):
    """
    Compute the galaxy stellar mass function from a diluted MTNG snapshot.

    Mirrors halo_smf_draws. For m_30kpc=False the function is identical to
    the original: it reads Subfind subhalo stellar masses from the catalogue,
    which is independent of the snapshot dilution. For m_30kpc=True the
    function replaces the per-subhalo aperture computation by a single
    cKDTree query over the diluted PartType4 catalogue: stars within
    30 physical kpc of each subhalo centre are summed, and the result is
    rescaled by red_fac to recover the full-resolution stellar mass.

    Note on noise: 30 pkpc is a small aperture. At full resolution a galaxy
    typically has tens to hundreds of star particles within it; after
    dilution by red_fac (with replacement, as the dilution script does) this
    drops by the same factor and the per-galaxy mass becomes Poisson-noisy.
    The estimator is unbiased on average but the SMF will show extra scatter
    relative to the full-resolution version, particularly at the faint end.

    :param sel_mask: dictionary with keys 'sel' and 'h_frac', as in
                     halo_smf_draws
    :type sel_mask: dict
    :param nbins: number of stellar-mass bin edges (so nbins-1 bins)
    :type nbins: int
    :param draws: number of independent draws over the selection
    :type draws: int
    :param m_30kpc: if True, compute stellar masses within 30 physical kpc
                    of each subhalo centre using particles; if False, use
                    the Subfind catalogue MassType[:, 4] directly
    :type m_30kpc: bool
    :param red_fac: dilution factor of the snapshot (e.g. 8)
    :type red_fac: int

    :return: dict with keys 'smf', 'bins', 'mstar', matching halo_smf_draws
    :rtype: dict
    """

    bins = np.logspace(9, 12.5, nbins)
    h = self.sim.Cosmology.pars['hubble']

    counts = {d: np.zeros(nbins - 1) for d in range(draws)}
    hist = {d: np.zeros(nbins - 1) for d in range(draws)}
    mstar_mean = {d: np.zeros(nbins - 1) for d in range(draws)}

    first = self.sim.fof['halo_firstsub']
    nsubs = self.sim.fof['halo_nsubs']

    # ------------------------------------------------------------------
    # Tree setup, only needed for the m_30kpc=True branch.
    # Built once, reused across all subhalo queries. Working in Mpc/h
    # (comoving) throughout to match the simulation's native units.
    # ------------------------------------------------------------------
    if m_30kpc:
        boxsize = self.sim.header['BoxSize']                 # Mpc/h, comoving
        star_pos = self.sim.stars['pos']                     # Mpc/h, comoving
        star_mass = 1e10 * self.sim.stars['mass'] / h        # Msun

        # Wrap any particles sitting at exactly boxsize (cKDTree requires
        # 0 <= coord < boxsize strictly).
        star_pos = np.mod(star_pos, boxsize)

        tree = cKDTree(star_pos, boxsize=boxsize)

        # 30 physical kpc -> comoving Mpc/h:
        #   30 pkpc = 30e-3 pMpc = (30e-3 / a) cMpc = (30e-3 / a) * h cMpc/h
        # where a is the scale factor at the snapshot redshift.
        a = self.sim.header['Time']
        aperture = 30e-3 * h / a                             # Mpc/h, comoving

        sub_pos = np.mod(self.sim.sub['pos'], boxsize)       # Mpc/h, comoving

    # ------------------------------------------------------------------
    # Main loop, structure matches halo_smf_draws
    # ------------------------------------------------------------------
    for m in range(len(sel_mask['sel'])):
        for d in range(draws):
            if sel_mask['h_frac'][d][m] == 0.0:
                continue

            mstar = []
            for i in range(len(sel_mask['sel'][m][d])):
                ihalo = sel_mask['sel'][m][d][i]
                start = first[ihalo]
                stop = start + nsubs[ihalo]

                if m_30kpc:
                    # One tree query per subhalo, summed inside the aperture.
                    # red_fac compensates for the snapshot dilution.
                    for isub in range(start, stop):
                        idx = tree.query_ball_point(sub_pos[isub], aperture)
                        m_in = red_fac * np.sum(star_mass[idx])
                        mstar.append(m_in)
                else:
                    mstar.extend(
                        self.sim.sub['MassType'][:, 4][start:stop] / h
                    )

            if m_30kpc:
                mstar = np.array(mstar)
            else:
                mstar = 1e10 * np.array(mstar)

            ids = np.digitize(mstar, bins)
            counts_i = np.array(
                [np.sum(np.ones(len(mstar))[np.where(ids == j)])
                 for j in range(1, len(bins))]
            ) / sel_mask['h_frac'][d][m]

            counts[d] += counts_i
            hist[d] += np.array(counts_i)
            mstar_mean[d] += np.array(
                [np.sum(mstar[np.where(ids == j)])
                 for j in range(1, len(bins))]
            ) / sel_mask['h_frac'][d][m]

    bin_width = np.log10(bins[1:]) - np.log10(bins[:-1])
    norm = 1 / ((self.sim.header['BoxSize'] / h) ** 3 * bin_width)

    for d in range(draws):
        hist[d] *= norm
        nz = counts[d] > 0
        mstar_mean[d][nz] = mstar_mean[d][nz] / counts[d][nz]

    return {'smf': hist, 'bins': bins, 'mstar': mstar_mean}

def get_parameters():
    wind_en_or      = []
    wind_vel_or     = []
    rho_rec_or      = []
    sf_ts_or        = []
    ef_kin_or       = []
    ef_high_or      = []
    f_re_or         = []

    for i in range(31):
        if i<30:
            filename = "/cosmos_storage/simulations/TNG_Family/MN5_resims/param_LH/param_MTNG-hydro_{:d}.txt".format(i)
        else:
            filename = "/cosmos_storage/simulations/TNG_Family/MN5_resims/param_LH/param_MTNG-hydro.txt"

        with open(filename, 'r') as f:
            for line in f.readlines():
                if len(line.split())!=0:
                    if line.split()[0] == 'WindEnergyIn1e51erg':
                        wind_en_or.append(float(line.split()[1]))
                    if line.split()[0] == 'VariableWindVelFactor':
                        wind_vel_or.append(float(line.split()[1]))
                    if line.split()[0] == 'WindFreeTravelDensFac':
                        rho_rec_or.append(float(line.split()[1]))
                    if line.split()[0] == 'MaxSfrTimescale':
                        sf_ts_or.append(float(line.split()[1]))
                    if line.split()[0] == 'RadioFeedbackFactor':
                        ef_kin_or.append(float(line.split()[1]))
                    if line.split()[0] == 'BlackHoleFeedbackFactor':
                        ef_high_or.append(float(line.split()[1]))
                    if line.split()[0] == 'RadioFeedbackReiorientationFactor':
                        f_re_or.append(float(line.split()[1]))

    rho_rec_or = np.log10(rho_rec_or)
    ef_kin_or = np.log10(ef_kin_or)
            
    wind_en   = (np.asarray(wind_en_or) - np.mean(wind_en_or)) / np.std(wind_en_or)
    wind_vel  = (np.asarray(wind_vel_or) - np.mean(wind_vel_or)) / np.std(wind_vel_or)
    rho_rec   = (np.asarray(rho_rec_or) - np.mean(rho_rec_or)) / np.std(rho_rec_or)
    sf_ts     = (np.asarray(sf_ts_or) - np.mean(sf_ts_or)) / np.std(sf_ts_or)
    ef_kin    = (np.asarray(ef_kin_or) - np.mean(ef_kin_or)) / np.std(ef_kin_or)
    ef_high   = (np.asarray(ef_high_or) - np.mean(ef_high_or)) / np.std(ef_high_or)
    f_re      = (np.asarray(f_re_or) - np.mean(f_re_or)) / np.std(f_re_or)

    return wind_en, wind_vel, rho_rec, sf_ts, ef_kin, ef_high, f_re

def pars(i, mass):

    wind_en, wind_vel, rho_rec, sf_ts, ef_kin, ef_high, f_re = get_parameters()

    arr = np.vstack( ( mass, np.ones(len(mass)) * wind_en[i],\
                        np.ones(len(mass)) * wind_vel[i],\
                        np.ones(len(mass)) * rho_rec[i],\
                        np.ones(len(mass)) * sf_ts[i],\
                        np.ones(len(mass)) * ef_kin[i],\
                        np.ones(len(mass)) * ef_high[i],\
                        np.ones(len(mass)) * f_re[i])).T

    return arr