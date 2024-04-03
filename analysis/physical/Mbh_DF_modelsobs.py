
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib.lines import Line2D

import cmasher as cmr

import h5py

import flare.plt as fplt
import flare.photom as phot
from flare.photom import M_to_lum
import flares_utility.analyse as analyse
from flares_utility.stats import poisson_confidence_interval
from unyt import c, Msun, yr, g, s, erg

import illustris_python as il

# set style
plt.style.use('../matplotlibrc.txt')


tags = ['005_z010p000', '006_z009p000', '007_z008p000',
        '008_z007p000', '009_z006p000', '010_z005p000']
redshifts = [10, 9, 8, 7, 6, 5]


binw = 0.25
X_limits = [6.625, 9.99]
Y_limits = [-8.5, -3.49]


filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'
flares = analyse.analyse(filename)



N = len(redshifts)
left = 0.1
top = 0.95
bottom = 0.15
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


for z, c, ax in zip(redshifts, cmr.take_cmap_colors('cmr.bubblegum_r', len(redshifts)), axes.flatten()):

    tag = flares.tag_from_zed[z]    

    X = flares.load_dataset(tag, *quantity)
    Mstar = flares.load_dataset(tag, 'Galaxy', 'Mstar_30')
    
    phi = np.zeros(len(bin_centres))
    N = np.zeros(len(bin_centres))

    for i, (sim, w) in enumerate(zip(flares.sims, flares.weights)):

        x = np.log10(np.array(X[sim])) + 10.

        N_temp, _ = np.histogram(x, bins=bin_edges)

        N += N_temp

        phi_temp = (N_temp / V) / binw

        phi += phi_temp * w

    upper = np.array([poisson_confidence_interval(n)[1] for n in N])
    lower = np.array([poisson_confidence_interval(n)[0] for n in N])

    phi_upper = phi * upper / N
    phi_lower = phi * lower / N
    ax.fill_between(bin_centres, np.log10(phi_upper), np.log10(phi_lower), alpha=0.1, color=c)

    ax.plot(bin_centres, np.log10(phi), ls='-', c=c, lw=2, alpha=1.0, zorder=2)
    # ax.plot(bin_centres[N>4], np.log10(phi[N>4]), ls = ls_, c=c, lw=lw_)


    # ------------------------------------------------------------------------------------
    # Add observations

    co = '0.7'

    # Matthee+2023
    if z==5:
        x = [7.5, 8.1]
        y = [-4.29, -5.05]
        xerr = [0.2, 0.4]
        yerr = [[0.15, 0.30], [0.11, 0.18]]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", c=co, markersize=5, label=r'$\rm Matthee+23$', zorder=0)

    # He+
    if z==5:
        x = np.arange(7.5, 10.7, 0.3)
        y = np.array([12.14, 9.92, 11.30, 13.93, 8.07, 4.46, 0.49, 1.61, 0.19, 0.04, 0.01])
        yerr = np.array([8.7, 6.24, 4.34, 3.68, 2.87, 1.89, 0.28, 1.08, 0.01, 0.01, 0.01])
        # ax.scatter(x, np.log10(y)-7, marker="s", c=c, s=5, label=r'$\rm He+23\ (z=4)$', alpha=0.5)
        yerr_u = np.log10(y+yerr)-np.log10(y)
        yerr_l = np.log10(y)-np.log10(y-yerr)
        print(yerr_l)
        yerr = np.array([yerr_l, yerr_u])
        ax.errorbar(x, np.log10(y)-7., yerr=yerr, fmt="s", c=co, markersize=3, label=r'$\rm He+23\ (z=4)$', zorder=0)


    # ------------------------------------------------------------------------------------
    # Add other models

    # Add Bluetides

    # Add Astrid

    # Add Simba

    



    # Add EAGLE
    

    # Add Illustris

    # Add illustris TNG100
    z_to_snaphots = {5: 17, 6: 13, 7: 11}
    if z in [5,6,7]:

        data_dir = '/Users/sw376/Dropbox/Research/data/simulations/illustris'
        simulation = 'TNG100-1'
        fields = ['SubhaloBHMass']
        X = il.groupcat.loadSubhalos(f'{data_dir}/{simulation}/outputs',z_to_snaphots[z],fields=fields)
        X = X[X>0.0]
        X = X/0.7
        log10X = np.log10(X) + 10. 

        N, bin_edges = np.histogram(log10X, bins=bin_edges)
        print(N)
        phi = N/(100**3)/binw
        ax.plot(bin_centres, np.log10(phi), c='0.7', lw=1, ls='--', label=r'$\rm TNG100$', zorder=0)

    # Add illustris TNG300
    # z_to_snaphots = {5: 17, 6: 13, 7: 11}
    # if z in [5,6,7]:

    #     data_dir = '/Users/sw376/Dropbox/Research/data/simulations/illustris'
    #     simulation = 'TNG100-1'
    #     fields = ['SubhaloBHMass']
    #     X = il.groupcat.loadSubhalos(f'{data_dir}/{simulation}/outputs',z_to_snaphots[z],fields=fields)
    #     X = X[X>0.0]
    #     X = X/0.7
    #     log10X = np.log10(X) + 10. 

    #     N, bin_edges = np.histogram(log10X, bins=bin_edges)
    #     print(N)
    #     phi = N/(100**3)/binw
    #     ax.plot(bin_centres, np.log10(phi), c='0.7', lw=1, ls='--', label=r'$\rm TNG100$', zorder=0)





    ax.text(0.5, 1.02, rf'$\rm z={z}$', horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes, fontsize = 7)

    ax.set_xlim(X_limits)
    ax.set_ylim(Y_limits)

    ax.legend(fontsize=8)


fig.text(0.04, bottom+(top-bottom)/2, r'$\rm\log_{10}[\phi/Mpc^{-3}\ dex^{-1}]$', rotation = 90, va='center', fontsize = 9)
fig.text(left+(right-left)/2, 0.08, r'$\rm \log_{10}(M_{\bullet}/M_{\odot})$', ha='center', fontsize = 9)

fig.savefig(f'figs/Mbh_DF_modelsobs.pdf')


fig.clf()
