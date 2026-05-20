import numpy as np
import bacco

import halotools.mock_observables as ht

import sys
sys.path.append("/cosmos_storage/home/fgmaion/MTNG-resims/src")
import utils

object_type = 'groups' # 'clusters' or 'groups'

def get_profiles(zoom, r_bins, ih_list):

    hpos  = zoom.fof['halo_pos']
    r500  = zoom.fof['halo_r500c']
    m200c = 1e10 * zoom.fof['halo_m200c']

    if object_type == 'clusters':
        msel = ( m200c[ih_list] > 10**(14.5) )
    elif object_type == 'groups':
        msel = ( m200c[ih_list] > 1e13 ) & ( m200c[ih_list] < 1e14 )
    sel_ids = ih_list[msel]

    r_dic = {}
    rho_dic = {}

    for i in range(3):
        if i==0:
            # DM
            _pos = zoom.dm['pos']
            _mass = 1e10 * np.ones_like(_pos[:,0]) * zoom.header['ParticleMass']
            r_dic['dm'] = np.zeros((len(sel_ids), len(r_bins)-1))
            rho_dic['dm'] = np.zeros((len(sel_ids), len(r_bins)-1))
        if i==1:
            # Gas
            _pos = zoom.gas['pos']
            _mass = 1e10 * zoom.gas['mass']
            r_dic['gas'] = np.zeros((len(sel_ids), len(r_bins)-1))
            rho_dic['gas'] = np.zeros((len(sel_ids), len(r_bins)-1))
        elif i==2:
            # Stars
            _pos = zoom.stars['pos']
            _mass = 1e10 * zoom.stars['mass']
            r_dic['stars'] = np.zeros((len(sel_ids), len(r_bins)-1))
            rho_dic['stars'] = np.zeros((len(sel_ids), len(r_bins)-1))

        for j in range(len(sel_ids)):
            print("Measuring the profile for halo {:d} of {:d}".format(j+1, len(sel_ids)))
            _hpos = np.array([hpos[sel_ids[j]]])
            _r500 = r500[sel_ids[j]]

            y, c = ht.radial_profile_3d(_hpos ,_pos, _mass, return_counts=True, rbins_normalized=r_bins, normalize_rbins_by=_r500)
            rr = 10**((np.log10(r_bins[1:])+np.log10(r_bins[:-1]))*0.5)
            #rr = (rbins[1:]+rbins[:-1])*0.5

            phys_r500_kpc = _r500 * 1e3 / zoom.Cosmology.pars['hubble']
            volume = (4 * np.pi / 3) * (r_bins[1:]**3 - r_bins[:-1]**3) * phys_r500_kpc**3

            dens1 = y * c / volume
            x1 = rr

            if i==0:
               r_dic['dm'][j,:] = x1
               rho_dic['dm'][j,:] = dens1
            if i==1:
               r_dic['gas'][j,:] = x1
               rho_dic['gas'][j,:] = dens1
            elif i==2:
               r_dic['stars'][j,:] = x1
               rho_dic['stars'][j,:] = dens1

    return r_dic, rho_dic

### Get the IDs for the chosen halos ###

name_list = ['LH_{:d}'.format(i) for i in range(30)] + ['fiducial']

sigma8 = 0.8159 #CHECK ME
ns     = 0.9667 #CHECK ME
tau    = 0.0965 #CHECK ME

_snap = 264
zoom = {}

for i in range(len(name_list)):
    base = "/cosmos_storage/simulations/TNG_Family/MN5_resims/"+name_list[i]+"/hydro_output/"
    zoom[name_list[i]] = bacco.Simulation(basedir=base, halo_file="groups_{:03d}/fof_subhalo_tab_{:03d}".format(_snap,_snap), dm_file="snapdir_{:03d}/snapshot_{:03d}".format(_snap,_snap),\
			    sim_format='TNG500', fixedPk=True, use_orphans=False,\
                            tau=tau, ns=ns, sigma8=sigma8, use_ids=False, tree_file="groups_{:03d}/subhalo_prog_{:03d}".format(_snap,_snap), numpart=4320**3)

### Load the Halo Selection ###
xmatch = {}
for i in range(len(name_list)):
    xmatch[name_list[i]] = utils.cross_match(zoom[name_list[i]], snap=264, name=name_list[i])

### Get those profiles
for i in range(len(name_list)):
    r_dic, rho_dic = get_profiles(zoom[name_list[i]], r_bins=np.logspace(-2, np.log10(3), 15), ih_list=xmatch[name_list[i]]['ind'])

    if object_type == 'clusters':
        np.save("/cosmos_storage/home/fgmaion/MTNG-resims/results/profiles/prof_clusters_{}.npy".format(name_list[i]), {'r':r_dic, 'rho':rho_dic})
    elif object_type == 'groups':
        np.save("/cosmos_storage/home/fgmaion/MTNG-resims/results/profiles/prof_groups_{}.npy".format(name_list[i]), {'r':r_dic, 'rho':rho_dic})
