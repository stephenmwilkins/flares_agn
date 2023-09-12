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

fl = flares.flares(f'{flares_dir}/flares_no_particlesed.hdf5', sim_type='FLARES')
df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
halo = fl.halos
print(fl.tags) # print list of available snapshots
tags = fl.tags #This would be z=5

MBH = fl.load_dataset('BH_Mass', arr_type='Galaxy') # Black hole mass of galaxy
MDOT = fl.load_dataset('BH_Mdot', arr_type='Galaxy') # Black hole accretion rate
MS = fl.load_dataset('Mstar_30', arr_type='Galaxy') # Black hole accretion rate
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
        s = (np.log10(MS[halo[ii]][tag])+10 > 7)
        ws = np.append(ws, np.ones(np.shape(X[halo[ii]][tag][s]))*weights[ii])
        x = np.append(x, X[halo[ii]][tag][s])
        y = np.append(y, Y[halo[ii]][tag][s])
        mstar = np.append(mstar, np.log10(MS[halo[ii]][tag][s])+10)
        lstar = np.append(lstar, np.log10(LFUV[halo[ii]][tag][s]))

    h = 0.6777  # Hubble parameter


    # converting MBHacc units to M_sol/yr
    x *= h * 6.445909132449984E23  # g/s
    x = x/constants.M_sun.to('g').value  # convert to M_sol/s
    x *= units.yr.to('s')  # convert to M_sol/yr


    y *= 10**10


    b = t_bb(y, x)

    s_t = np.array(b) > 10**4

    ws = ws[s_t]

    q = np.array([l_agn(g, etta=0.1) for g in x[s_t]])

    x = np.array(mstar)[s_t]


    y = (ratio_from_t(b[s_t]))*10**q

    y = np.log10(y /  ((const.c/(1500*u.AA).to(u.m)).to(u.Hz)).value)

    yy = np.log10(10**lstar[s_t] + 10**y)

    # --- simply print the ranges of the quantities

    print(f'z={z}')
    print(f'm_star,min = {np.min(x)}, m_star,med = {np.median(x)}, m_star,max = {np.max(x)}')
    print(f'L_agn,bol,min = {np.min(q)}, L_agn,bol,med = {np.median(q)}, L_agn,bol,max = {np.max(q)}')
    print(f'L_agn,uv,min = {np.min(y)}, L_agn,uv,med = {np.median(y)}, L_agn,uv,max = {np.max(y)}')


    binw = 0.5
    bins = np.arange(28,32,binw)
    b_c = bins[:-1]+binw/2

    N_weighted_total, edges_total = np.histogram(yy, bins=bins, weights=ws)

    N_weighted_gal, edges_gal = np.histogram(lstar[s_t], bins = bins, weights = ws)

    N_weighted, edges = np.histogram(y, bins = bins, weights = ws)

    h = 0.6777
    vol = (4/3)*np.pi*(14/h)**3

    phi_total = N_weighted_total/(binw*vol)
    phi_gal = N_weighted_gal/(binw*vol)
    phi = N_weighted/(binw*vol)

    ax.plot(bins[:-1] + binw / 2, np.log10(phi_gal), ls='dotted', c=cmap(norm(z)), label=rf'Stellar')
    ax.plot(bins[:-1] + binw / 2, np.log10(phi), ls='dashed', c=cmap(norm(z)), label = rf'AGN')
    ax.plot(bins[:-1] + binw / 2, np.log10(phi_total), ls='-', c=cmap(norm(z)), label=rf'Total')

    ax.text(0.7, 0.9, r'$\rm z={0:.0f}$'.format(z), fontsize=8, transform=ax.transAxes,
                           color=cmap(norm(z)))

    ax.set_ylim(-8, -2)

    ax.set_xlabel(r'$\rm log_{10}[L_{FUV}\;/\;erg\,s^{-1}\,Hz^{-1}]$')
    ax.set_ylabel(r'$\rm log_{10}[\phi\;/\;Mpc^{-3}\, dex^{-1}]$')

    ax.legend(loc='lower left', prop={'size': 6})

    fig.savefig(f'figures/individual_lfs/uvlf_{int(z)}.pdf', bbox_inches='tight')
    fig.clf()

