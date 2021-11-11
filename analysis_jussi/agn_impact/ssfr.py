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

import h5py

def t_bb(m, m_dot):
    return 2.24*10**9*(m_dot)**(1/4)*(m)**(-1/2) #2.24*10**9*m_dot**(4)*m**(-8) #

def l_agn(m_dot, etta=0.1):
    m_dot = (m_dot*u.M_sun/u.yr).to(u.kg / u.s) # accretion rate in SI
    c = const.c #speed of light
    etta = etta
    l = (etta*m_dot*c**2).to(u.erg/u.s)
    return np.log10(l.value) # output in log10(erg/s)


cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=4., vmax=15.)

flares_dir = '../../../data/simulations'

#fl = flares.flares(f'{flares_dir}/flares_no_particlesed.hdf5', sim_type='FLARES') #_no_particlesed

fl_03_AGN = h5py.File(f'{flares_dir}/FLARES_03_sp_info.hdf5')
fl_03_noAGN = h5py.File(f'{flares_dir}/FLARES_03_noAGN_sp_info.hdf5')

fl_24_AGN = h5py.File(f'{flares_dir}/FLARES_24_sp_info.hdf5')
fl_24_noAGN = h5py.File(f'{flares_dir}/FLARES_24_noAGN_sp_info.hdf5')

runs = [fl_03_AGN, fl_03_noAGN, fl_24_AGN, fl_24_noAGN]
run_labels = ['03_AGN', '03_noAGN', '24_AGN', '24_noAGN']

tags = np.flip(['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000', '005_z010p000', '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000', '011_z004p770'])
zeds = np.flip([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4])

for i, run in enumerate(runs):

    fig, axes = plt.subplots(3, 4, figsize=(8, 6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, bottom=0.15, top=1.0, right=0.85, wspace=0.0, hspace=0.0)

    for ii, tag in enumerate(tags):

        zed = zeds[ii]

        x = np.log10(run[tag]['Galaxy']['Mstar_30'])+10
        y = np.log10(run[tag]['Galaxy']['SFR_inst_30'])
        y = y - x + 9
        #axes.flatten()[ii].scatter(np.log10(run[tag]['Galaxy']['Mstar_30'])+10,
        #                        np.log10(run[tag]['Galaxy']['SFR_inst_30']), c='k', alpha=0.6)

        # -- this will calculate the weighted quantiles of the distribution
        quantiles = [0.84, 0.50, 0.16]  # quantiles for range
        bins = np.arange(7, 11, 0.2)  #  x-coordinate bins
        bincen = (bins[:-1] + bins[1:]) / 2.
        out = flares.binned_weighted_quantile(x, y, weights=np.ones(len(x)), bins=bins, quantiles=quantiles)

        # --- plot the median and quantiles for bins with >10 galaxies

        N, bin_edges = np.histogram(x, bins=bins)
        Ns = (N >= 10)

        axes.flatten()[ii].plot(bincen, out[:, 1], c=cmap(norm(zed)), ls=':')
        axes.flatten()[ii].plot(bincen[Ns], out[:, 1][Ns], c=cmap(norm(zed)))
        axes.flatten()[ii].fill_between(bincen[Ns], out[:, 0][Ns], out[:, 2][Ns], color=cmap(norm(zed)), alpha=0.2)

        axes.flatten()[ii].set_ylim(0., 1.7)
        axes.flatten()[ii].set_xlim(8., 10.9)

        axes.flatten()[ii].text(0.8, 0.9, r'$\rm z={0:.0f}$'.format(zed), fontsize=8,
                               transform=axes.flatten()[ii].transAxes,
                               color=cmap(norm(zed)))

    fig.text(0.01, 0.55, r'$\rm log_{10}[sSFR_{inst} \; / \; Gyr^{-1}]$', ha='left', va='center',
             rotation='vertical', fontsize=10)
    fig.text(0.45, 0.05, r'$\rm log_{10}[M_{*}\;/\;M_{\odot}]$', ha='center', va='bottom', fontsize=10)

    fig.savefig(f'figures/sSFR_Mstar_{run_labels[i]}.pdf', bbox_inches='tight')
    fig.clf()
