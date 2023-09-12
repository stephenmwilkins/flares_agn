import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt
import astropy.constants as const
import astropy.units as u

from uncertainties import unumpy

import flares

import FLARE.photom as photom
import FLARE.plt as fplt

mass_cut = 5.

def l_agn(m_dot, etta=0.1, etta_r=0.15):
    m_dot = 10**m_dot*1.98847*10**30/(365*24*60*60) # accretion rate in SI
    c = 299792458 #speed of light
    etta = etta*(1-etta_r)
    return np.log10(etta*m_dot*c**2*10**7) # output in log10(erg/s)

def M_edd(m_bh):
    m_bh = m_bh * u.solMass
    m_edd = (4*np.pi*const.G*m_bh*const.m_p)/(0.1*const.c*const.sigma_T)
    return (m_edd.to(u.solMass / u.yr)).value

def L_edd(m_bh):
    return 3*10**4*(const.L_sun.to(u.erg*u.s**(-1))).value*10**m_bh



# Observed quasars (MOVE SOMEWHERE LATER)

# Onoue
onoue = {'label': 'Onoue+2019'}
onoue['lum'] = np.array([8.96*10**45, 1.03*10**45, 6.77*10**45, 4.44*10**45, 4.38*10**45, 2.66*10**45])
onoue['mass'] = np.array([22.*10**8, 0.38*10**8, 6.3*10**8, 11*10**8, 7.1*10**8, 7.0*10**8])
onoue['edd_rate'] = np.array([0.16, 1.1, 0.43, 0.17, 0.24, 0.24])
onoue['err_edd_u'] = np.array([0.04, 0.5, 0.08, 0.04, 0.18, 0.06])
onoue['err_edd_l'] = np.array([0.02, 0.3, 0.05, 0.05, 0.08, 0.01])
onoue['err_mass_u'] = np.array([2*10**8, 0.1*10**8, 0.8*10**8, 3*10**8, 2.4*10**8, 1.4*10**8])
onoue['err_mass_l'] = np.array([6*10**8, 0.18*10**8, 1.2*10**8, 2*10**8, 5.2*10**8, 2.3*10**8])




cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=5., vmax=10.)

flares_dir = '../../../data/simulations'

fl = flares.flares(f'{flares_dir}/flares_no_particlesed.hdf5', sim_type='FLARES')
df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
halo = fl.halos
print(fl.tags) # print list of available snapshots
tags = fl.tags #This would be z=5

Mdot = fl.load_dataset('BH_Mdot', arr_type='Galaxy') # Black hole mass of galaxy
MBH = fl.load_dataset('BH_Mass', arr_type='Galaxy') # Black hole mass of galaxy
Mdm = fl.load_dataset('Mdm', arr_type='Galaxy') # Black hole mass of galaxy
Mgas = fl.load_dataset('Mgas', arr_type='Galaxy') # Black hole mass of galaxy
Mstar = fl.load_dataset('Mstar', arr_type='Galaxy') # Black hole mass of galaxy


X = Mdot



# --- FLARES is composed of 40 individual simulations. Each simulation has a statistical weight (basically the likelhood a similar region would appear in a volume of the real Universe). To get true distribution functions we need to weight each galaxy by the weight of the simulation.

# --- This bit of codes reads in the properties for all of the galaxies in every simulation and also produces a corresponding weight array. These can be combined later to get a true relationship.

df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
weights = np.array(df['weights'])

import h5py
eagle = h5py.File(f'{flares_dir}/EAGLE_REF_sp_info.hdf5')

fig, axes = plt.subplots(2, 3, figsize = (6, 4), sharex = True, sharey=True)
fig.subplots_adjust(left=0.07, bottom=0.15, top=1.0, right=0.85, wspace=0.0, hspace=0.0)

m_bh = np.linspace(5., 10, 100)
accr = np.linspace(-10, 1, 100)

eagletags = np.flip(np.array(['002_z009p993', '003_z008p988', '004_z008p075', '005_z007p050', '006_z005p971', '008_z005p037']))

for i, tag in enumerate(np.flip(fl.tags)):
    z = np.flip(fl.zeds)[i]
    ws, x, y, mdm, mgas, mstar = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    eagletag = eagletags[i]

    for ii in range(len(halo)):
        s = (np.log10(MBH[halo[ii]][tag])+10 > 5.5) & (np.log10(Mstar[halo[ii]][tag])+10 > 8)
        ws = np.append(ws, np.ones(np.shape(X[halo[ii]][tag][s]))*weights[ii])
        x = np.append(x, X[halo[ii]][tag][s])
        y = np.append(y, np.log10(MBH[halo[ii]][tag][s])+10)
        mdm = np.append(mdm, Mdm[halo[ii]][tag][s])
        mgas = np.append(mgas, Mgas[halo[ii]][tag][s])
        mstar = np.append(mstar, Mstar[halo[ii]][tag][s])


    #print(len(mdm), len(mstar), len(mgas), len(y))
    mhalo = np.log10(mdm + mgas + mstar) + 10
    axes.flatten()[i].scatter(mhalo, y, c='k', alpha=0.5, s=4, zorder=5)
    axes.flatten()[i].scatter(mhalo, y, c=cmap(norm(z)), alpha=0.7, s=4, zorder=10)


    axes.flatten()[i].text(0.2, 0.8, r'$\rm z={0:.0f}$'.format(z), fontsize=8, transform=axes.flatten()[i].transAxes, color=cmap(norm(z)))

    axes.flatten()[i].set_xticks([8,9,10,11, 12, 13])
    axes.flatten()[i].set_yticks([6, 7, 8, 9])
    axes.flatten()[i].set_xlim(9.,13.5)
    axes.flatten()[i].set_ylim(5.9, 9.5)


fig.text(0.45, 0.05, r'$\rm log_{10}[M_{*}\;/\;M_{\odot}]$', ha = 'center', va = 'bottom', fontsize=10)
fig.text(0.01,0.55, r'$\rm log_{10}[M_{BH}\;/\;M_{\odot}]$', ha = 'left', va = 'center', rotation = 'vertical', fontsize=10)

fig.savefig(f'figures/M_BH_M_star.pdf', bbox_inches='tight')
fig.clf()
