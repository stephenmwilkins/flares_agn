
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.lines as mlines

import cmasher as cmr

import h5py

import flare.plt as fplt
import flare.photom as phot
from flare.photom import M_to_lum
import flares_utility.analyse as analyse


tags = ['005_z010p000', '006_z009p000', '007_z008p000',
        '008_z007p000', '009_z006p000', '010_z005p000']
redshifts = [10, 9, 8, 7, 6, 5]


binw = 0.25
X_limits = [6.51, 9.49]
Y_limits = [-8.01, -3.49]


filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'
flares = analyse.analyse(filename)



fig = plt.figure(figsize = (3.5, 3.5))

left  = 0.15
height = 0.8
bottom = 0.15
width = 0.8

ax = fig.add_axes((left, bottom, width, height))

# --- FLARES


# flares.list_datasets()

V = (4./3) * np.pi * (flares.radius)**3 # Mpc^3


bin_edges = np.arange(*[6.5, 10.], binw)
bin_centres = bin_edges[:-1]+binw/2

quantity = ['Galaxy', 'BH_Mass']


for z, c in zip(redshifts, cmr.take_cmap_colors('cmr.bubblegum_r', len(redshifts))):

    tag = flares.tag_from_zed[z]

    ## ---- get data
    phi = np.zeros(len(bin_centres))
    N = np.zeros(len(bin_centres))

    X = flares.load_dataset(tag, *quantity)

    
    for i, (sim, w) in enumerate(zip(flares.sims, flares.weights)):

        x = np.log10(np.array(X[sim])) + 10.
        x = x[x>0.0]

        N_temp, _ = np.histogram(x, bins = bin_edges)

        N += N_temp

        phi_temp = (N_temp / V) / binw

        phi += phi_temp * w
   
    ax.plot(bin_centres, np.log10(phi), ls = '--', c=c, lw=1.5)
    ax.plot(bin_centres[N>4], np.log10(phi[N>4]), ls = '-', c=c, lw=1.5, label = rf'$\rm z={z}$')

    # Matthee
    if z==5:

        x = [7.5, 8.1]
        y = [-4.29, -5.05]
        xerr = [0.2, 0.4]
        yerr = [[0.15, 0.30], [0.11, 0.18]]

        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", c=c, markersize=5, label=r'$\rm Matthee+23$')




ax.legend(fontsize=7, labelspacing=0.0)
ax.set_xlim(X_limits)
ax.set_ylim(Y_limits)

ax.fill_between([6,7.], [-9.,-9.], [0,0], color='k', alpha=0.05)


ax.set_ylabel(r'$\rm\log_{10}(\phi/Mpc^{-3}\ dex^{-1})$')
ax.set_xlabel(r'$\rm \log_{10}(M_{\bullet}/M_{\odot})$')

fig.savefig(f'figs/Mbh_DF_allz.pdf')
fig.clf()
