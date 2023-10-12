
import numpy as np
import matplotlib.cm as cm
import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib.colors import Normalize

import scipy.stats as stats
from scipy.stats import binned_statistic
import cmasher as cmr

import h5py

import flare.plt as fplt
import flare.photom as phot
from flare.photom import M_to_lum
import flares_utility.limits
import flares_utility.plt
import flares_utility.analyse as analyse
import flares_utility.stats

from unyt import c, Msun, yr, Lsun

flares_dir = '/Users/sw376/Dropbox/Research/data/simulations/flares'

tags = ['005_z010p000', '006_z009p000', '007_z008p000',
        '008_z007p000', '009_z006p000', '010_z005p000']
redshifts = [10, 9, 8, 7, 6, 5]



AGN = {}
noAGN = {}

AGN['03'] = h5py.File(f'{flares_dir}/FLARES_03_sp_info_noparticles.hdf5')
noAGN['03'] = h5py.File(f'{flares_dir}/FLARES_03_noAGN_sp_info_noparticles.hdf5')

AGN['24'] = h5py.File(f'{flares_dir}/FLARES_24_sp_info_noparticles.hdf5')
noAGN['24'] = h5py.File(f'{flares_dir}/FLARES_24_noAGN_sp_info_noparticles.hdf5')

AGN['03'].visit(print)


xlimits = [8., 11.] # Msun
ylimits = [-0.5, 1.0]

bins = np.arange(7., 11., 0.25)
bin_centres = 0.5*(bins[:-1]+bins[1:])

fig = plt.figure(figsize = (3.5, 3.5))

left  = 0.15
height = 0.8
bottom = 0.15
width = 0.8

ax = fig.add_axes((left, bottom, width, height))



for simulation in ['03']:

    for z, tag, c in zip(redshifts, tags, cmr.take_cmap_colors('cmr.bubblegum_r', len(redshifts))):

        X = np.log10(AGN[simulation][f'{tag}/Galaxy/Mstar_30'][()]) + 10.
        Y = np.log10(AGN[simulation][f'{tag}/Galaxy/SFR_inst_30'][()]) - X

        mean_AGN, bin_edges, N, = binned_statistic(X, Y, bins = bins, statistic = 'mean')

        X = np.log10(noAGN[simulation][f'{tag}/Galaxy/Mstar_30'][()]) + 10.
        Y = np.log10(noAGN[simulation][f'{tag}/Galaxy/SFR_inst_30'][()]) - X

        mean_noAGN, bin_edges, N, = binned_statistic(X, Y, bins = bins, statistic = 'mean')

        print(mean_AGN - mean_noAGN)

        ax.plot(bin_centres, mean_noAGN - mean_AGN, c=c, lw=1, label = rf'$\rm z={z}$')


ax.axhline(0.0, lw=2, c='k', alpha=0.1)

ax.legend(fontsize=8, labelspacing=0.1)

ax.set_xlim(xlimits)
ax.set_ylim(ylimits)


ax.set_xlabel(r'$\rm log_{10}(M_{\star}/M_{\odot})$')
ax.set_ylabel(r'$\rm log_{10}(<sSFR_{no-AGN}>/<sSFR_{AGN}>)$')



filename = f'figs/impact_ssfr.pdf'
print(filename)

fig.savefig(filename)

