import numpy as np
import bacco

import halotools.mock_observables as ht

import sys
sys.path.append("/cosmos_storage/home/fgmaion/MTNG-resims/src")
import utils

def get_profiles(zoom, r_bins, ih_list):

    hpos = zoom.fof['halo_pos']
    r200 = zoom.fof['halo_r200c']

    r_dic = {}
    rho_dic = {}

    for i in range(1):
        if i==0:
            # DM
            _pos = zoom.dm['pos']
            _mass = np.ones_like(_pos[:,0])  * zoom.header['ParticleMass'] * 1e10
            r_dic['dm'] = np.zeros((len(ih_list), len(r_bins)-1))
            rho_dic['dm'] = np.zeros((len(ih_list), len(r_bins)-1))
        elif i==1:
            # Gas
            _pos = zoom.gas['pos']
            _mass = zoom.gas['mass'] * 1e10
            r_dic['gas'] = np.zeros((len(ih_list), len(r_bins)-1))
            rho_dic['gas'] = np.zeros((len(ih_list), len(r_bins)-1))
        elif i==2:
            # Stars
            _pos = zoom.stars['pos']
            _mass = zoom.stars['mass'] * 1e10
            r_dic['stars'] = np.zeros((len(ih_list), len(r_bins)-1))
            rho_dic['stars'] = np.zeros((len(ih_list), len(r_bins)-1))

        elif i==3:
            # BH
            _pos = zoom.bh['pos']
            _mass = zoom.bh['mass'] * 1e10
            r_dic['bh'] = np.zeros((len(ih_list), len(r_bins)-1))
            rho_dic['bh'] = np.zeros((len(ih_list), len(r_bins)-1))

        for j in range(1):
            _hpos = np.array([hpos[ih_list[j]]])

            y, c = ht.radial_profile_3d(_hpos, _pos, _mass, return_counts=True, rbins_absolute=r_bins)
            rr = 10**((np.log10(r_bins[1:])+np.log10(r_bins[:-1]))*0.5)
            #rr = (rbins[1:]+rbins[:-1])*0.5
            volume = 4 * np.pi / 3 *(r_bins[1:]**3 - r_bins[:-1]**3)

            dens1 = y * c / volume
            x1 = rr

            if i==0:
                r_dic['dm'][j,:] = x1
                rho_dic['dm'][j,:] = dens1
            elif i==1:
                r_dic['gas'][j,:] = x1
                rho_dic['gas'][j,:] = dens1
            elif i==2:
                r_dic['stars'][j,:] = x1
                rho_dic['stars'][j,:] = dens1
            elif i==3:
                r_dic['bh'][j,:] = x1
                rho_dic['bh'][j,:] = dens1

    return r_dic, rho_dic

### Get the IDs for the chosen halos ###

name_list = ['LH_{:d}'.format(i) for i in range(1)]

sigma8 = 0.8159 #CHECK ME
ns     = 0.9667 #CHECK ME
tau    = 0.0965 #CHECK ME

_snap = 264
zoom = {}

for i in range(len(name_list)):
    base = "/cosmos_storage/simulations/TNG_Family/MN5_resims/"+name_list[i]+"/hydro_output/"
    zoom[name_list[i]] = bacco.Simulation(basedir=base, halo_file="groups_{:03d}/fof_subhalo_tab_{:03d}".format(_snap,_snap), dm_file="snapdir_{:03d}/snapshot_{:03d}".format(_snap,_snap), sim_format='TNG500', fixedPk=True, use_orphans=False,\
                            tau=tau, ns=ns, sigma8=sigma8, use_ids=False, tree_file="groups_{:03d}/subhalo_prog_{:03d}".format(_snap,_snap), numpart=4320**3)

### Load the Halo Selection ###
xmatch = {}
for i in range(len(name_list)):
    xmatch[name_list[i]] = utils.cross_match(zoom[name_list[i]], snap=264, name=name_list[i])

### Get those profiles

r_dic, rho_dic = get_profiles(zoom[name_list[0]], r_bins=np.logspace(-2, 1.5, 15), ih_list=xmatch[name_list[0]]['ind'])

np.save("/cosmos_storage/home/fgmaion/MTNG-resims/results/profiles/r_and_rho_test.npy", r_dic)
