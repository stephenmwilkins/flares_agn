import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm

mpl.use('Agg')
import matplotlib.pyplot as plt

import flares

import FLARE.plt as fplt



import FLARE.photom as photom
import FLARE.plt as fplt

cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=5., vmax=10.)

flares_dir = '../../../data/simulations'

fl = flares.flares(f'{flares_dir}/flares_no_particlesed.hdf5', sim_type='FLARES')
df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
halo = fl.halos
print(fl.tags) # print list of available snapshots
tags = fl.tags #This would be z=5




MS = fl.load_dataset('Mstar_30', arr_type='Galaxy') # stellar mass of galaxy
MBH = fl.load_dataset('BH_Mass', arr_type='Galaxy') # Black hole mass of galaxy

X = MS

# --- FLARES is composed of 40 individual simulations. Each simulation has a statistical weight (basically the likelhood a similar region would appear in a volume of the real Universe). To get true distribution functions we need to weight each galaxy by the weight of the simulation.

# --- This bit of codes reads in the properties for all of the galaxies in every simulation and also produces a corresponding weight array. These can be combined later to get a true relationship.


fig = plt.figure(figsize=(3,3))
left  = 0.2
bottom = 0.2
width = 0.75
height = 0.75
ax = fig.add_axes((left, bottom, width, height))

tags = np.flip(np.array([fl.tags[1], fl.tags[3], fl.tags[5]]))
zeds = np.flip(np.array([fl.zeds[1], fl.zeds[3], fl.zeds[5]]))

for i, tag in enumerate(tags):
    z = zeds[i]
    weights = np.array(df['weights'])

    ws, ws_y, x, y = np.array([]), np.array([]), np.array([]), np.array([])
    for ii in range(len(halo)):
        s = (np.log10(MS[halo[ii]][tag]) + 10 > 8)
        s_bh = (np.log10(MBH[halo[ii]][tag][s])+10 > 5.5)
        ws = np.append(ws, np.ones(np.shape(X[halo[ii]][tag][s]))*weights[ii])
        x = np.append(x, np.log10(X[halo[ii]][tag][s]))

        ws_y = np.append(ws_y, np.ones(np.shape(X[halo[ii]][tag][s][s_bh])) * weights[ii])
        y = np.append(y, np.log10(X[halo[ii]][tag][s][s_bh]))


    x += 10 # units are 1E10 M_sol
    y += 10  # units are 1E10 M_sol

    binw = 0.25
    bins = np.arange(8,12,binw)
    b_c = bins[:-1]+binw/2

    N_allgals, edges = np.histogram(x, bins = bins, weights = ws)
    N_weighted, edges = np.histogram(y, bins = bins, weights = ws_y)

    h = 0.6777
    vol = (4/3)*np.pi*(14/h)**3

    phi_allgals = N_allgals/(binw*vol)
    phi = N_weighted/(binw*vol)

    ax.plot(bins[:-1] + binw / 2, np.log10(phi_allgals), '--', c=cmap(norm(z)))
    ax.plot(bins[:-1] + binw / 2, np.log10(phi), c=cmap(norm(z)), label = rf'$\rm z={int(z)}$')

ax.set_ylim(-7.5, -2.5)
ax.set_xlim(8)
ax.legend(prop={'size': 6})



ax.set_xlabel(r'$\rm log_{10}[M_{\star}\;/\;M_{\odot}]$')
ax.set_ylabel(r'$\rm log_{10}[\phi\;/\;Mpc^{-3}\, dex^{-1}]$')
fig.savefig(f'figures/GSMF_smbh_hosts.pdf')
fig.clf()
