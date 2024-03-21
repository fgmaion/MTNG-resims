def stellar_mf(sim=None, mass_edges=[12.5,13], nbins=100, Nhalos=None, vmax_sel=None):
    parent_mass = sim.sub['parent_halo']['mfof'] * 1e10

    if vmax_sel is not None:
        halo_vmax = sim.sub['vmax'][sim.fof['halo_firstsub']]
        parent_vmax = halo_vmax[sim.sub['fof_index']]

        avg_vmax = np.mean( parent_vmax[np.where( (parent_mass>10**mass_edges[0]) & (parent_mass<10**mass_edges[1]) & (sim.sub['LenType'][:,4]>200) & (np.sum(sim.sub['MassType'], axis=1)>1))] )
        if vmax_sel == 'high':
            sel_mask = np.where( (parent_mass>10**mass_edges[0]) & (parent_mass<10**mass_edges[1]) & (sim.sub['LenType'][:,4]>200) & (np.sum(sim.sub['MassType'], axis=1)>1) & (parent_vmax > avg_vmax ) )
        elif vmax_sel == 'low':
            sel_mask = np.where( (parent_mass>10**mass_edges[0]) & (parent_mass<10**mass_edges[1]) & (sim.sub['LenType'][:,4]>200) & (np.sum(sim.sub['MassType'], axis=1)>1) & (parent_vmax <= avg_vmax) )
        else:
            raise ValueError('The option given for vmax_sel is not supported.')
    else:
        sel_mask = np.where( (parent_mass>10**mass_edges[0]) & (parent_mass<10**mass_edges[1]) & (sim.sub['LenType'][:,4]>200) & (np.sum(sim.sub['MassType'], axis=1)>1) )

    index_mask = sim.sub['fof_index'][sel_mask]
    mstar_mask = (sim.sub['MassType'][:,4])[sel_mask] * 1e10
    parent_mass  = parent_mass[sel_mask]

    unique_indices = np.unique(index_mask)
    if Nhalos is not None:
        fof_choice = np.random.choice(unique_indices, min(Nhalos, len(unique_indices)), replace=False)
        mask = np.isin(index_mask, fof_choice)
        mstar = mstar_mask[mask]
        mhalos_tot  = np.sum(sim.fof['halo_mfof'][fof_choice]*1e10)
        nhalos = len(fof_choice)

    else: 
        fof_choice = unique_indices
        mstar = mstar_mask
        mhalos_tot = np.sum(np.unique(parent_mass))
        nhalos = len(np.unique(parent_mass))

    bins = np.logspace(8, 13, nbins)
    hist = np.histogram(mstar, bins=bins)

    return  {'smf':hist, 'mh_tot':mhalos_tot, 'Nh':nhalos, 'h_idx':fof_choice}