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
X_limits = [6.51, 9.99]
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

quantity = ['Galaxy', 'BH_Mass']


for z, c, ax in zip(redshifts, cmr.take_cmap_colors('cmr.gem_r', len(redshifts)), axes.flatten()):

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


   
    # Matthe
    if z==5:

        x = [7.5, 8.1]
        y = [-4.29, -5.05]
        xerr = [0.2, 0.4]
        yerr = [[0.15, 0.30], [0.11, 0.18]]

        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", c=c, markersize=5, label=r'$\rm Matthee+23$')

    # He
    if z==5:
        x = np.arange(7.5, 10.7, 0.3)

        y = np.array([12.14, 9.92, 11.30, 13.93, 8.07, 4.46, 0.49, 1.61, 0.19, 0.04, 0.01])
        yerr = np.array([8.7, 6.24, 4.34, 3.68, 2.87, 1.89, 0.28, 1.08, 0.01, 0.01, 0.01])

        # ax.scatter(x, np.log10(y)-7, marker="s", c=c, s=5, label=r'$\rm He+23\ (z=4)$', alpha=0.5)
        yerr_u = np.log10(y+yerr)-np.log10(y)
        yerr_l = np.log10(y)-np.log10(y-yerr)
        print(yerr_l)
        yerr = np.array([yerr_l, yerr_u])
        ax.errorbar(x, np.log10(y)-7., yerr=yerr, fmt="s", c=c, markersize=3, label=r'$\rm He+23\ (z=4)$', alpha=0.5)


    ax.plot(bin_centres, np.log10(phi), ls = '-', c=c, lw=1)


    ax.text(0.5, 1.02, rf'$\rm z={z}$', horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes, fontsize = 7)

    ax.set_xlim(X_limits)
    ax.set_ylim(Y_limits)
    ax.legend(fontsize=8)


fig.text(0.04, bottom+(top-bottom)/2, r'$\rm\log_{10}[\phi/Mpc^{-3}\ dex^{-1}]$', rotation = 90, va='center', fontsize = 9)
fig.text(left+(right-left)/2, 0.04, r'$\rm \log_{10}(M_{\bullet}/M_{\odot})$', ha='center', fontsize = 9)



fig.savefig(f'figs/Mbh_DF.pdf')


fig.clf()
