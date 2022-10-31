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
bounds = [0, 1, 2, 3, 4]#, 5, 6]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


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

#bin_edges = [(-0.3, -0.15), (-0.15, -0.04), (-0.04, 0.04), (0.04, 0.12), (0.12, 0.22), (0.22, 0.3)]
bin_edges = [(-0.04, 0.04), (0.04, 0.12), (0.12, 0.22), (0.22, 0.3)]

left, bottom, top, right = 0.07, 0.15, 1.0, 0.85

fig, axes = plt.subplots(2, 2, figsize = (5, 4), sharex = True)
fig.subplots_adjust(left=left, bottom=bottom, top=top, right=right, wspace=0.0, hspace=0.0)

for i, tag in enumerate(fl.tags[::-1][:2]):
    z = fl.zeds[::-1][i]

    weights = np.array(df['weights'])

    binw = 0.5
    bins = np.arange(5,10,binw)
    b_c = bins[:-1]+binw/2
    quantiles = [0.84, 0.50, 0.16]

    vol = h = 0.6777
    vol = (4 / 3) * np.pi * (14 / h) ** 3

    cmap_ticks = []
    cmap_tickpos = []

    for bin_number in bounds[:-1]:
        ms = np.array([])
        x = np.array([])
        ws = np.array([])

        count = 0

        for ii in range(len(halo)):

            if deltas[ii] >= bin_edges[bin_number][0] and deltas[ii] < bin_edges[bin_number][1]:
                s = (np.log10(MS[halo[ii]][tag]) + 10 > 8)&(np.log10(X[halo[ii]][tag])+10 > 5.5)
                ws = np.append(ws, np.ones(np.shape(X[halo[ii]][tag][s]))*weights[ii])
                x = np.append(x, np.log10(X[halo[ii]][tag][s]))
                ms = np.append(ms, np.log10(MS[halo[ii]][tag][s]))
                count +=1

        x += 10 # units are 1E10 M_sol
        ms += 10

        N_weighted, edges = np.histogram(x, bins = bins)#, weights = ws)
        Ns = N_weighted > 5

        out = flares.binned_weighted_quantile(x, np.log10(10**x/10**ms), ws, bins, quantiles)

        err = np.sqrt(N_weighted)/(binw*vol*count)

        phi = N_weighted/(binw*vol*count)

        if z==5.:
            print(bin_edges[bin_number])
            print(phi)
            print(err)

        axes.flatten()[i].plot(bins[:-1] + binw / 2, np.log10(phi), c=cmap(norm(bin_number+0.5)), alpha=0.8)
        for j, bin in enumerate(N_weighted):
            axes.flatten()[i].plot([bins[:-1][j] + binw / 2, bins[:-1][j] + binw / 2], [np.log10(phi[j]-err[j]), np.log10(phi[j]+err[j])], c=cmap(norm(bin_number+0.5)), alpha=0.8)

        axes.flatten()[i+2].plot(b_c, out[:, 1], c=cmap(norm(bin_number+0.5)), ls='-')
        #axes.flatten()[i+2].plot(b_c[Ns], out[:, 1][Ns], c=cmap(norm(bin_number+0.5)), ls='-')
        axes.flatten()[i+2].fill_between(b_c[Ns], out[:, 0][Ns], out[:, 2][Ns], color=cmap(norm(bin_number+0.5)),
                                       alpha=0.4)


        cmap_ticks.append(rf"$\rm [{bin_edges[bin_number][0]}, {bin_edges[bin_number][1]}] \; ({count}) $")
        cmap_tickpos.append(bin_number+0.5)

    axes.flatten()[i].text(0.8, 0.9, r'$\rm z={0:.0f}$'.format(z), fontsize=8, transform=axes.flatten()[i].transAxes,
                           color='k', ha='left')
    axes.flatten()[i+2].text(0.8, 0.1, r'$\rm z={0:.0f}$'.format(z), fontsize=8, transform=axes.flatten()[i+2].transAxes,
                           color='k', ha='left')

    axes.flatten()[i].set_xlim(5.7, 9.3)
    axes.flatten()[i].set_ylim(-5.6,-2.85)

    #axes.flatten()[i+2].set_xlim(5.8, 9.2)
    axes.flatten()[i+2].set_ylim(-4.6,-1.8)

axes.flatten()[1].tick_params(labelleft=False)
axes.flatten()[3].tick_params(labelleft=False)


axes.flatten()[0].set_ylabel(r'$\rm log_{10}[\phi\;/\;Mpc^{-3}\, dex^{-1}]$')
axes.flatten()[2].set_ylabel(r'$\rm log_{10}[M_{BH}\;/\;M_{*}]$')
axes.flatten()[2].set_xlabel(r'$\rm log_{10}[M_{BH}\;/\;M_{\odot}]$')
axes.flatten()[3].set_xlabel(r'$\rm log_{10}[M_{BH}\;/\;M_{\odot}]$')


cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
cmapper.set_array([])

cax = fig.add_axes([right, bottom, 0.03, top-bottom])
bar = fig.colorbar(cmapper, cax=cax, ticks=cmap_tickpos, orientation='vertical')#, format='%d')
bar.ax.set_yticklabels(cmap_ticks)
bar.ax.tick_params(labelsize=8)
cax.set_ylabel(r'$\rm [log_{10}[1+\delta]] \; (N_{regions})$')

fig.savefig(f'figures/regions_per_z_binned/BHMF_MBH_to_Mstar_z5z6_as_Mstar.pdf', bbox_inches='tight')
fig.clf()
