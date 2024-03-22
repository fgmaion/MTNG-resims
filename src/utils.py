import numpy as np

class split_halos():

    def __init__(self, sim):
        self.sim = sim

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

        return mask, fof_choice

    def subhalo_sel(self, mass_edges=None, vmax_sel=False):
        '''
        array of floats:mass_edges
        Contains the edges of the mass-bin in which we wish to select our halo

        bool:vmax_sel
        Whether to split the selection not only by mass, but by concentration as well
        '''

        parent_mass = self.sim.sub['parent_halo']['mfof'] * 1e10

        if vmax_sel is not None:
            halo_vmax = self.sim.sub['vmax'][self.sim.fof['halo_firstsub']]
            parent_vmax = halo_vmax[self.sim.sub['fof_index']]

            avg_vmax = np.mean( parent_vmax[np.where( (parent_mass>10**mass_edges[0]) & (parent_mass<10**mass_edges[1]) & (self.sim.sub['LenType'][:,4]>200) & (np.sum(self.sim.sub['MassType'], axis=1)>1))] )
            if vmax_sel == 'high':
                sel_mask = np.where( (parent_mass>10**mass_edges[0]) & (parent_mass<10**mass_edges[1]) & (self.sim.sub['LenType'][:,4]>200) & (np.sum(self.sim.sub['MassType'], axis=1)>1) & (parent_vmax > avg_vmax ) )
            elif vmax_sel == 'low':
                sel_mask = np.where( (parent_mass>10**mass_edges[0]) & (parent_mass<10**mass_edges[1]) & (self.sim.sub['LenType'][:,4]>200) & (np.sum(self.sim.sub['MassType'], axis=1)>1) & (parent_vmax <= avg_vmax) )
            else:
                raise ValueError('The option given for vmax_sel is not supported.')
        else:
            sel_mask = np.where( (parent_mass>10**mass_edges[0]) & (parent_mass<10**mass_edges[1]) & (self.sim.sub['LenType'][:,4]>200) & (np.sum(self.sim.sub['MassType'], axis=1)>1) )

        return sel_mask

    def stellar_mf(self, mass_edges=[12.5,13], nbins=100, Nhalos=None, vmax_sel=None):
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
        
        sel_mask         = self.subhalo_sel(mass_edges=mass_edges, vmax_sel=vmax_sel)
        mask, fof_choice = self.sample_halos(Nhalos=Nhalos, sel_mask=sel_mask)

        mstar = ( (self.sim.sub['MassType'][:,4])[sel_mask] * 1e10 )[mask]
        mhalos_tot  = np.sum(self.sim.fof['halo_mfof'][fof_choice]*1e10)
        nhalos = len(fof_choice)

        bins = np.logspace(8, 13, nbins)
        hist = np.histogram(mstar, bins=bins)

        return  {'smf':hist, 'mh_tot':mhalos_tot, 'Nh':nhalos, 'h_idx':fof_choice}