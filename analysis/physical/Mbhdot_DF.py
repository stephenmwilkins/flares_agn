
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


# set style
plt.style.use('../matplotlibrc.txt')

tags = ['005_z010p000', '006_z009p000', '007_z008p000',
        '008_z007p000', '009_z006p000', '010_z005p000']
redshifts = [10, 9, 8, 7, 6, 5]


binw = 0.25
X_limits = [-10., 3.]
Y_limits = [-8.5, -3.49]


filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'
flares = analyse.analyse(filename)



N = len(redshifts)
left = 0.1
top = 0.95
bottom = 0.1
right = 0.975
panel_width = (right-left)/N
panel_height = top-bottom
fig, axes = plt.subplots(2, 3, figsize = (7,5), sharey = True, sharex = True)
plt.subplots_adjust(left=left, top=top, bottom=bottom, right=right, wspace=0.0, hspace=0.1)




# --- FLARES


# flares.list_datasets()

V = (4./3) * np.pi * (flares.radius)**3 # Mpc^3


bin_edges = np.arange(*X_limits, binw)
bin_centres = bin_edges[:-1]+binw/2


quantity = ['Galaxy', 'BH_Mdot']


for z, c, ax in zip(redshifts, cmr.take_cmap_colors('cmr.gem_r', len(redshifts)), axes.flatten()):

    tag = flares.tag_from_zed[z]

    ## ---- get data
    phi = np.zeros(len(bin_centres))
    N = np.zeros(len(bin_centres))

    X = flares.load_dataset(tag, *quantity)
    Mstar = flares.load_dataset(tag, 'Galaxy', 'Mstar_30')
    MBH = flares.load_dataset(tag, 'Galaxy', 'BH_Mass')
    
    for i, (sim, w) in enumerate(zip(flares.sims, flares.weights)):

        MBH_ = np.log10(MBH[sim])+10.

        x = np.log10(np.array(X[sim])) + 10.
        x = x[MBH_>5.]

        print(np.median(x))

        N_temp, _ = np.histogram(x, bins = bin_edges)

        N += N_temp

        phi_temp = (N_temp / V) / binw

        phi += phi_temp * w


   
    ax.plot(bin_centres, np.log10(phi), ls = '-', c=c, lw=1)


    ax.text(0.5, 1.02, rf'$\rm z={z}$', horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes, fontsize = 7)

    ax.set_xlim(X_limits)
    ax.set_ylim(Y_limits)


fig.text(0.04, bottom+(top-bottom)/2, r'$\rm\log_{10}[\phi/Mpc^{-3}\ dex^{-1}]$', rotation = 90, va='center', fontsize = 9)
fig.text(left+(right-left)/2, 0.04, r'$\rm \log_{10}(\dot{M}_{\bullet}/M_{\odot}\ yr^{-1})$', ha='center', fontsize = 9)



fig.savefig(f'figs/Mbhdot_DF.pdf')


fig.clf()
