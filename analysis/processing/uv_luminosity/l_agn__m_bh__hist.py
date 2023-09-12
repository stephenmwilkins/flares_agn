import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt

import flares

import astropy.constants as constants
import astropy.units as units

import FLARE.photom as photom
import FLARE.plt as fplt

from scipy.integrate import simps, trapz

import astropy.constants as const
import astropy.units as u

from scipy.interpolate import interp1d

import _pickle as pickle

bol_correction = pickle.load(open('bolometric_correction.p', 'rb'))
ratio_from_t = interp1d(bol_correction['AGN_T'], bol_correction['ratio']['FUV'])


mass_cut = 5.

def t_bb(m, m_dot):
    return 2.24*10**9*(m_dot)**(1/4)*(m)**(-1/2) #2.24*10**9*m_dot**(4)*m**(-8) #

def l_agn(m_dot, etta=0.1):
    m_dot = (m_dot*u.M_sun/u.yr).to(u.kg / u.s) # accretion rate in SI
    c = const.c #speed of light
    etta = etta
    l = (etta*m_dot*c**2).to(u.erg/u.s)
    return np.log10(l.value) # output in log10(erg/s)

def l_edd(m_bh):
    return np.log10(1.26*10**38*m_bh)

cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=5., vmax=10.)

flares_dir = '../../../../data/simulations'

fl = flares.flares(f'{flares_dir}/flares_no_particlesed.hdf5', sim_type='FLARES')
df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
halo = fl.halos
print(fl.tags) # print list of available snapshots
tags = fl.tags #This would be z=5

MBH = fl.load_dataset('BH_Mass', arr_type='Galaxy') # Black hole mass of galaxy
MDOT = fl.load_dataset('BH_Mdot', arr_type='Galaxy') # Black hole accretion rate
MS = fl.load_dataset('Mstar', arr_type='Galaxy') # Black hole accretion rate
LFUV = fl.load_dataset('FUV', arr_type=f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic/')

#LUM = fl.load_dataset('DustModelI', arr_type=f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity')

Y = MBH
X = MDOT

df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
weights = np.array(df['weights'])


for i, tag in enumerate(np.flip(fl.tags)):

    fig = plt.figure(figsize=(3, 3))
    left = 0.2
    bottom = 0.2
    width = 0.75
    height = 0.75
    ax = fig.add_axes((left, bottom, width, height))


    z = np.flip(fl.zeds)[i]
    ws, x, y, mstar, lstar = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    for ii in range(len(halo)):
        s = (np.log10(MS[halo[ii]][tag])+10 > 8)
        ws = np.append(ws, np.ones(np.shape(X[halo[ii]][tag][s]))*weights[ii])
        x = np.append(x, X[halo[ii]][tag][s])
        y = np.append(y, Y[halo[ii]][tag][s])
        mstar = np.append(mstar, np.log10(MS[halo[ii]][tag][s])+10)
        lstar = np.append(lstar, np.log10(LFUV[halo[ii]][tag][s]))

    h = 0.6777  # Hubble parameter

    print(f'N_obj = {len(x)}')

    # converting MBHacc units to M_sol/yr
    x *= h


    y *= 10**10
    yy = 10**np.linspace(5.5, 10, 100)
    xx = l_edd(yy)
    yy = np.log10(yy)
    y = np.log10(y)
    b = np.array([l_agn(q) for q in x])

    x=b

    cmap2d = plt.cm.Blues
    norm2d = mpl.colors.LogNorm(vmax=100)

    cmapper = ax.hexbin(x, y, cmap=cmap2d, gridsize=(32,12), extent=[39, 47, 5.5, 9.5], edgecolor='none', linewidths=0.2, norm=norm2d) # weights=ws,
    ax.plot(xx, yy, c='k', linewidth=1, ls='--', alpha=0.8)

    ax.text(0.7, 0.9, r'$\rm z={0:.0f}$'.format(z), fontsize=8, transform=ax.transAxes,
                           color=cmap(norm(z)))

    cax = fig.add_axes([width + left, bottom, 0.05, height])
    bar = fig.colorbar(cmapper, cax=cax, orientation='vertical')
    cax.set_yticklabels([0, 1, 2])
    cax.set_ylabel(r'$\rm log_{10}[N]$')

    ax.set_ylim(5.5, 9)
    ax.set_xlim(42.7, 47)

    ax.set_xlabel(r'$\rm log_{10}[L_{AGN, bol}\;/\;erg\,s^{-1}]$')
    ax.set_ylabel(r'$\rm log_{10}[M_{BH}\;/\;M_{\odot}]$')

    #ax.legend(loc='lower left', prop={'size': 6})

    fig.savefig(f'figures/mbh_lagn/individual/mbh_lagn_2d_{int(z)}_logmap.pdf', bbox_inches='tight')
    fig.clf()

