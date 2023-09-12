import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import flares

import sys

sys.path.insert(1, '../../../../data/literature')

from agn_obs import literature

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

def tau_dust(bh_los, dtm_ratio, lam=1500, kappa=0.0795, gamma=-1):
    # Defaults are for 1500Å (FUV), kappa_ISM from Vijayan et al. 2021, gamma from Wilkins et al. 2017
    tau = kappa * dtm_ratio * bh_los * (lam/5500)**gamma
    return tau

def attn(lum, bh_los, dtm_ratio, lam=1500, kappa=0.0795, gamma=-1):
    # lum = log10(L)
    return np.log10(10**lum*np.exp(-1*tau_dust(bh_los, dtm_ratio, lam, kappa, gamma)))


cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=5., vmax=10.)

flares_dir = '../../../../data/simulations'

#fl = flares.flares(f'{flares_dir}/flares_old.hdf5', sim_type='FLARES') #_no_particlesed
fl = flares.flares(f'{flares_dir}/flares_no_particlesed.hdf5', sim_type='FLARES') #_no_particlesed
df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
halo = fl.halos
print(fl.tags) # print list of available snapshots
tags = fl.tags #This would be z=5

MBH = fl.load_dataset('BH_Mass', arr_type='Galaxy') # Black hole mass of galaxy
MDOT = fl.load_dataset('BH_Mdot', arr_type='Galaxy') # Black hole accretion rate
MS = fl.load_dataset('Mstar', arr_type='Galaxy') # Black hole accretion rate
LFUV_dust = fl.load_dataset('FUV', arr_type=f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI/')
LFUV_int = fl.load_dataset('FUV', arr_type=f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic/')

LBOL = fl.load_dataset('DustModelI', arr_type=f'Galaxy/BPASS_2.2.1/Chabrier300/Indices/Lbol/')
BHLOS = pickle.load(open(f'{flares_dir}/bh_los.p', 'rb')) #fl.load_dataset('BH_los', arr_type='Galaxy')
DTM = fl.load_dataset('DTM', arr_type='Galaxy')

#LUM = fl.load_dataset('DustModelI', arr_type=f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity')

Y = MBH
X = MDOT

df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
weights = np.array(df['weights'])

attn_out = {}

fig, axes = plt.subplots(2, 3, figsize = (6, 4), sharex = True, sharey=True)
fig.subplots_adjust(left=0.07, bottom=0.15, top=1.0, right=0.85, wspace=0.0, hspace=0.0)
for i, tag in enumerate(np.flip(fl.tags)):

    z = np.flip(fl.zeds)[i]

    ws, x, y, mstar, lstar_dust, lstar_int, los, dtm = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    for ii in range(len(halo)):
        s = (np.log10(MS[halo[ii]][tag])+10 > 8)
        ws = np.append(ws, np.ones(np.shape(X[halo[ii]][tag][s]))*weights[ii])
        x = np.append(x, X[halo[ii]][tag][s])
        y = np.append(y, Y[halo[ii]][tag][s])
        mstar = np.append(mstar, np.log10(MS[halo[ii]][tag][s])+10)
        lstar_dust = np.append(lstar_dust, np.log10(LFUV_dust[halo[ii]][tag][s]))
        lstar_int = np.append(lstar_int, np.log10(LFUV_int[halo[ii]][tag][s]))
        los = np.append(los, BHLOS[halo[ii]][tag][s])
        dtm = np.append(dtm, DTM[halo[ii]][tag][s])

    h = 0.6777  # Hubble parameter

    x *= h

    # converting MBHacc units to M_sol/yr
    #x *= h * 6.445909132449984E23  # g/s
    #x = x/constants.M_sun.to('g').value  # convert to M_sol/s
    #x *= units.yr.to('s')  # convert to M_sol/yr


    y *= 10**10


    b = t_bb(y, x)

    s_t = np.array(b) > 10**4

    ws = ws[s_t]

    q = np.array([l_agn(g, etta=0.1) for g in x[s_t]])

    mstar = mstar[s_t]
    lstar_int = lstar_int[s_t]
    lstar_dust = lstar_dust[s_t]

    y = (ratio_from_t(b[s_t]))*10**q
    #y = (1/4.4) * 10 ** q

    agn_intrinsic = np.log10(y /  ((const.c/(1500*u.AA).to(u.m)).to(u.Hz)).value)
    agn_dust = attn(agn_intrinsic, los[s_t], dtm[s_t])

    print(min(agn_dust), max(agn_dust))

    agn = agn_dust - agn_intrinsic
    stellar = lstar_dust - lstar_int

    # -- this will calculate the weighted quantiles of the distribution
    quantiles = [0.84, 0.50, 0.16]  # quantiles for range
    bins = np.arange(7.5, 12, 0.5)  #  x-coordinate bins
    bincen = (bins[:-1] + bins[1:]) / 2.
    out = flares.binned_weighted_quantile(mstar, agn, ws, bins, quantiles)

    out2 = flares.binned_weighted_quantile(mstar, stellar, ws, bins, quantiles)

    # --- plot the median and quantiles for bins with >10 galaxies

    N, bin_edges = np.histogram(mstar, bins=bins)
    Ns = N > 10

    axes.flatten()[i].plot(bincen, out[:, 1], c=cmap(norm(z)), ls=':')
    axes.flatten()[i].plot(bincen[Ns], out[:, 1][Ns], c=cmap(norm(z)), ls='-')
    axes.flatten()[i].fill_between(bincen[Ns], out[:, 0][Ns], out[:, 2][Ns], color=cmap(norm(z)),
                                   alpha=0.4)

    axes.flatten()[i].plot(bincen, out2[:, 1], c=cmap(norm(z)), ls=':')
    axes.flatten()[i].plot(bincen[Ns], out2[:, 1][Ns], c=cmap(norm(z)), ls='--')
    axes.flatten()[i].fill_between(bincen[Ns], out2[:, 0][Ns], out2[:, 2][Ns], color=cmap(norm(z)),
                                   alpha=0.4)

    axes.flatten()[i].set_xlim(8.5, 11.5)
    axes.flatten()[i].set_ylim(-4.5, 0.5)
    #axes.flatten()[i].set_ylim(0., 0.35)

    axes.flatten()[i].set_xticks([9, 10, 11])
    #axes.flatten()[i].set_yticks([0, 0.1, 0.2, 0.3])
    axes.flatten()[i].set_yticks([-4, -3, -2, -1, 0])

    axes.flatten()[i].text(0.8, 0.9, r'$\rm z={0:.0f}$'.format(z), fontsize=8, transform=axes.flatten()[i].transAxes,
                           color=cmap(norm(z)), ha='left')

axes.flatten()[i].plot(-99, -99, color='k', ls='-', label='AGN', alpha=0.8)
axes.flatten()[i].plot(-99, -99, color='k', ls='--', label='Stellar', alpha=0.8)
axes.flatten()[i].legend(loc='center right')

fig.text(0.01, 0.55, r'$\rm log_{10}[L_{FUV, dust} \; / \; L_{FUV, intrinsic}]$', ha='left', va='center', rotation='vertical',
         fontsize=10)
fig.text(0.45, 0.05, r'$\rm log_{10}[M_{*}\;/\;M_{\odot}]$', ha='center', va='bottom', fontsize=10)

fig.savefig(f'figures/fuv_dust_to_intrinsic__mstar__grid.pdf', bbox_inches='tight')
fig.clf()