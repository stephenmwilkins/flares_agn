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


cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=5., vmax=10.)

flares_dir = '../../../../data/simulations'

fl = flares.flares(f'{flares_dir}/flares_no_particlesed.hdf5', sim_type='FLARES') #_no_particlesed
df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
halo = fl.halos
print(fl.tags) # print list of available snapshots
tags = fl.tags #This would be z=5

MBH = fl.load_dataset('BH_Mass', arr_type='Galaxy') # Black hole mass of galaxy
MDOT = fl.load_dataset('BH_Mdot', arr_type='Galaxy') # Black hole accretion rate
MS = fl.load_dataset('Mstar', arr_type='Galaxy') # Black hole accretion rate
LFUV = fl.load_dataset('FUV', arr_type=f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic/')
LBOL = fl.load_dataset('Intrinsic', arr_type=f'Galaxy/BPASS_2.2.1/Chabrier300/Indices/Lbol/')

#LUM = fl.load_dataset('DustModelI', arr_type=f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity')

Y = MBH
X = MDOT

df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
weights = np.array(df['weights'])


fig, axes = plt.subplots(2, 3, figsize = (6, 4), sharex = True, sharey=True)
fig.subplots_adjust(left=0.07, bottom=0.15, top=1.0, right=0.85, wspace=0.0, hspace=0.0)

for i, tag in enumerate(np.flip(fl.tags)):


    z = np.flip(fl.zeds)[i]
    ws, x, y, mstar, lstar, lbol = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    for ii in range(len(halo)):
        s = (np.log10(MS[halo[ii]][tag])+10 > 8)
        ws = np.append(ws, np.ones(np.shape(X[halo[ii]][tag][s]))*weights[ii])
        x = np.append(x, X[halo[ii]][tag][s])
        y = np.append(y, Y[halo[ii]][tag][s])
        mstar = np.append(mstar, np.log10(MS[halo[ii]][tag][s])+10)
        lstar = np.append(lstar, np.log10(LFUV[halo[ii]][tag][s]))
        lbol = np.append(lbol, np.log10(LBOL[halo[ii]][tag][s]))

    h = 0.6777  # Hubble parameter


    # converting MBHacc units to M_sol/yr
    x *= h

    y *= 10**10
    b = t_bb(y, x)

    s_t = np.array(b) > 10**4

    q = np.array([l_agn(g, etta=0.1) for g in x[s_t]])

    ws = ws[s_t]

    fuv_agn = (ratio_from_t(b[s_t])) * 10 ** q / ((const.c/(1500*u.AA).to(u.m)).to(u.Hz)).value

    x = mstar[s_t]

    y = np.log10(fuv_agn / 10**lstar[s_t])

    s_outshine = (y > 0)

    print(z, np.sum(s_outshine))

    # -- this will calculate the weighted quantiles of the distribution
    quantiles = [0.84, 0.50, 0.16]  # quantiles for range
    bins = np.arange(8, 12, 0.25)  #  x-coordinate bins
    bincen = (bins[:-1] + bins[1:]) / 2.
    out = flares.binned_weighted_quantile(x, y, ws, bins, quantiles)

    # --- plot the median and quantiles for bins with >10 galaxies

    N, bin_edges = np.histogram(x, bins=bins)
    Ns = N > 10

    axes.flatten()[i].plot(bincen, out[:, 1], c=cmap(norm(z)), ls=':')
    axes.flatten()[i].plot(bincen[Ns], out[:, 1][Ns], c=cmap(norm(z)), label=rf'$\rm z={int(z)}$')
    axes.flatten()[i].fill_between(bincen[Ns], out[:, 0][Ns], out[:, 2][Ns], color=cmap(norm(z)),
                                   alpha=0.4)

    axes.flatten()[i].axhline(0, alpha=0.8, c='k', ls='--', linewidth=1)


    axes.flatten()[i].scatter(mstar[s_t][s_outshine], y[s_outshine], s=5, color=cmap(norm(z)))

    axes.flatten()[i].set_xlim(9, 11)
    axes.flatten()[i].set_ylim(-8, 1.5)

    axes.flatten()[i].set_xticks([9, 9.5, 10, 10.5])

    axes.flatten()[i].text(0.97, 0.92, r'$\rm z={0:.0f}$'.format(z), fontsize=8, transform=axes.flatten()[i].transAxes,
                           color=cmap(norm(z)), ha='right')


    #ax.set_xlabel(r'$\rm log_{10}[L_{FUV}\;/\;erg\,s^{-1}\,Hz^{-1}]$')
    #ax.set_ylabel(r'$\rm log_{10}[\phi\;/\;Mpc^{-3}\, dex^{-1}]$')

fig.text(0.01, 0.55, r'$\rm log_{10}[L_{AGN, FUV} \; / \; L_{stellar, FUV}]$', ha = 'left', va = 'center', rotation = 'vertical', fontsize=10)
fig.text(0.45,0.05, r'$\rm log_{10}[M_{*}\;/\;M_{\odot}]$', ha = 'center', va = 'bottom', fontsize=10)

fig.savefig(f'figures/agn_fuv_frac_grid_mstar.pdf', bbox_inches='tight')
fig.clf()

