import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt

import flares

import FLARE.photom as photom
import FLARE.plt as fplt

cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=-0.3, vmax=0.3)

flares_dir = '../../../data/simulations'

fl = flares.flares(f'{flares_dir}/flares_no_particlesed.hdf5', sim_type='FLARES')
df = pd.read_csv(f'{flares_dir}/weights_grid.txt')

halo = fl.halos
print(fl.tags) # print list of available snapshots
tags = fl.tags  #This would be z=5


MBH = fl.load_dataset('BH_Mass', arr_type='Galaxy') # Black hole mass of galaxy
MS = fl.load_dataset('Mstar', arr_type='Galaxy') # Black hole accretion rate

X = MBH

deltas = np.log10(1+np.array(df['delta']))

print(min(np.array(df['weights'])), max(np.array(df['weights'])))

# --- FLARES is composed of 40 individual simulations. Each simulation has a statistical weight (basically the likelhood a similar region would appear in a volume of the real Universe). To get true distribution functions we need to weight each galaxy by the weight of the simulation.

# --- This bit of codes reads in the properties for all of the galaxies in every simulation and also produces a corresponding weight array. These can be combined later to get a true relationship.

for i, tag in enumerate(fl.tags):
    z = fl.zeds[i]

    fig = plt.figure(figsize=(3, 3))
    left = 0.2
    bottom = 0.2
    width = 0.75
    height = 0.75
    ax = fig.add_axes((left, bottom, width, height))

    weights = np.array(df['weights'])

    binw = 0.5
    bins = np.arange(5,10,binw)
    b_c = bins[:-1]+binw/2

    vol = h = 0.6777
    vol = (4 / 3) * np.pi * (14 / h) ** 3

    x = {}
    ws = {}

    for ii in range(len(halo)):
        x[ii] = np.array([])
        ws[ii] = np.array([])
        s = (np.log10(MS[halo[ii]][tag]) + 10 > 8)&(np.log10(X[halo[ii]][tag])+10 > 5.5)
        ws[ii] = np.append(ws[ii], np.ones(np.shape(X[halo[ii]][tag][s]))*weights[ii])
        x[ii] = np.append(x[ii], np.log10(X[halo[ii]][tag][s]))


        x[ii] += 10 # units are 1E10 M_sol

        N_weighted, edges = np.histogram(x[ii], bins = bins, weights = ws[ii])

        phi = N_weighted/(binw*vol)

        ax.plot(bins[:-1] + binw / 2, np.log10(phi), c=cmap(norm(deltas[ii])))

    ax.text(0.8, 0.9, r'$\rm z={0:.0f}$'.format(z), fontsize=8, transform=ax.transAxes,
                           color='k', ha='left')

    ax.set_xlim(5.8, 9.2)
    ax.set_ylim(-8.,-4.5)

    cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    cmapper.set_array([])

    cax = fig.add_axes([width + left, bottom, 0.05, height])
    bar = fig.colorbar(cmapper, cax=cax, orientation='vertical')#, format='%d')
    #bar.set_ticks()
    cax.set_ylabel(r'$\rm log_{10}[1+\delta]$')

    ax.set_xlabel(r'$\rm log_{10}[M_{BH}\;/\;M_{\odot}]$')
    ax.set_ylabel(r'$\rm log_{10}[\phi\;/\;Mpc^{-3}\, dex^{-1}]$')
    fig.savefig(f'figures/regions_per_z_weighted/MBH_DF_{z}.pdf', bbox_inches='tight')
    fig.clf()
