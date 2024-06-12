import numpy as np
import bacco

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

    def sample_halos(self, Nhalos=None, sel_mask=None):
        '''
            int:Nhalos
            Amount of halos we wish to randomly select from total population

            int array:sel_mask
            Array of indices of the pre-selected subhalos. To be used as a mask for subhalo quantities

        '''

        index_mask = self.sim.sub['fof_index'][sel_mask]

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

        parent_mass = self.sim.sub['parent_halo']['mfof'] * 1e10

        # subselection by concentration
        if vmax_sel is True:
            halo_vmax = self.sim.sub['vmax'][self.sim.fof['halo_firstsub']]
            parent_vmax = halo_vmax[self.sim.sub['fof_index']]

            avg_vmax = np.mean( parent_vmax[np.where( (parent_mass>10**mhalo_edges[0]) & (parent_mass<10**mhalo_edges[1]) & (self.sim.sub['LenType'][:,4]>200) & (np.sum(self.sim.sub['MassType'], axis=1)>1))] )
            if vmax_sel == 'high':
                sel_mask = np.where( (parent_mass>10**mhalo_edges[0]) & (parent_mass<10**mhalo_edges[1]) & (self.sim.sub['LenType'][:,4]>200) & (np.sum(self.sim.sub['MassType'], axis=1)>1) & (parent_vmax > avg_vmax ) )
            elif vmax_sel == 'low':
                sel_mask = np.where( (parent_mass>10**mhalo_edges[0]) & (parent_mass<10**mhalo_edges[1]) & (self.sim.sub['LenType'][:,4]>200) & (np.sum(self.sim.sub['MassType'], axis=1)>1) & (parent_vmax <= avg_vmax) )
            else:
                raise ValueError('The option given for vmax_sel is not supported.')
        else:
            sel_mask = {}
            fof_choice = []
            halo_frac = np.zeros(mhalo_edges.shape[0])
            for m in range(mhalo_edges.shape[0]):
                # select galaxies/subhalos in halos of a given mass
                if DM_only is False:
                    sel_temp = np.where( (parent_mass>10**mhalo_edges[m,0]) & (parent_mass<10**mhalo_edges[m,1]) & (self.sim.sub['LenType'][:,4]>200) & (np.sum(self.sim.sub['MassType'], axis=1)>1) )[0]
                else:
                    sel_temp = np.where( (parent_mass>10**mhalo_edges[m,0]) & (parent_mass<10**mhalo_edges[m,1]) & (self.sim.sub['len']>200) )[0]

                # sample those halos randomly
                mask, fof_temp, halo_frac[m] = self.sample_halos(Nhalos=Nhalos, sel_mask=sel_temp)
                fof_choice.extend(fof_temp)

                # get the galaxies in the selected halos
                sel_mask[m] = sel_temp[np.where(mask)[0]]

        fof_choice = np.array(fof_choice)

        mhalos_tot = np.sum(self.sim.fof['halo_mfof'][fof_choice]*1e10)
        nhalos = len(fof_choice)

        return {'sel':sel_mask, 'h_idx':fof_choice, 'mh_tot':mhalos_tot, 'Nh':nhalos, 'h_frac':halo_frac}

    def stellar_mf(self, sel_mask=None, mhalo_edges=None, nbins=100, Nhalos=None, vmax_sel=None):
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
        for m in range(len(sel_mask['sel'])):
            mstar = ( (self.sim.sub['MassType'][:,4])[sel_mask['sel'][m]] * 1e10 )
            hist += np.histogram(mstar, bins=bins)[0] / sel_mask['h_frac'][m]

        return  {'smf':hist, 'bins':bins}

    def get_bpo(
            self, recompute=False, IA_terms=("J2=2", "J2=22", "J22=2", "J222=", "J2-2-2-")
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

            pbm = pb.ProbabilisticBiasManager(dm_mtng, variables=variables, damping_scale=0.2, ngrid=384, cachedir='/lscratch/fgmaion/Intrinsic_Alignments/MTNG/Notebooks/cachedir')
            # Note if you pass the parameter  cachedir="path/to/some/empty/directory"
            # you may save some time, at the cost of storing some extra files

            IA_model = pbm.setup_bias_model(pb.IA_TensorBiasND, terms=IA_terms, spatial_order=2)

            q = self.q_pos(self.sim)
            I = bacco.utils.S_to_I(self.sim.sub['subhalo_stellar_MOI'])

            bias = pbm.fit_bias(model=IA_model, tracer_q=q, error='qjack4', tracer_properties={'I':I})
            bpo = np.float32(IA_model.bpo)
            np.save("/lscratch/fgmaion/prob-bias/MTNG/biases/IA_bias_so0_mtng_z{:.2f}".format(z), [{'bias':bias, 'bpo':bpo}])
        
        else:
            load_bias = np.load("/lscratch/fgmaion/prob-bias/MTNG/biases/IA_bias_so0_mtng_z{:.2f}.npy".format(z), allow_pickle=True)[0]
            bpo = load_bias['cs_bpo']

        return bpo

    def bias_sm(self, sel_mask=None, bpo=None, mhalo_edges=None, Nhalos=None, vmax_sel=None, recompute=False):
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
        mstar = (self.sim.sub['MassType'][:,4])[sel_comb] * 1e10

        # bias per object of final selected subhalos
        bpo_sel = bpo[sel_comb]

        mstar = np.array(mstar, dtype=np.float)
        bpo_sel = np.array(bpo_sel)            

        # selection of galaxies per stellar-mass
        D = 0.5
        ms_edges = np.arange(9.5,12.5,D)
        idx = np.digitize(np.log10(mstar), bins=ms_edges)
        
        bias = np.zeros((len(ms_edges)-1, 6))
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
