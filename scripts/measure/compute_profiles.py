import numpy as np

import halotools.mock_observables as ht

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import utils
import paths
import loading

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

_snap = 264

zoom = loading.load_all_zooms(
    name_list, snap=_snap, use_ids=False, numpart=4320**3,
    tree_file="groups_{0:03d}/subhalo_prog_{0:03d}".format(_snap))

### Load the Halo Selection ###
xmatch = {}
for name in name_list:
    xmatch[name] = utils.cross_match(zoom[name], snap=264, name=name)

### Get those profiles
outdir = os.path.join(paths.RESULTS_DIR, "profiles")
os.makedirs(outdir, exist_ok=True)
for name in name_list:
    r_dic, rho_dic = get_profiles(zoom[name], r_bins=np.logspace(-2, np.log10(3), 15), ih_list=xmatch[name]['ind'])

    if object_type == 'clusters':
        np.save(os.path.join(outdir, "prof_clusters_{}.npy".format(name)), {'r':r_dic, 'rho':rho_dic})
    elif object_type == 'groups':
        np.save(os.path.join(outdir, "prof_groups_{}.npy".format(name)), {'r':r_dic, 'rho':rho_dic})
