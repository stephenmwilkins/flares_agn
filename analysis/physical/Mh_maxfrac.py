
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
X_limits = [8., 14.]
Y_limits = [0, 1.1]
# Y_limits = [8., 14.]

filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_bh.hdf5'
flares = analyse.analyse(filename)

filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_noparticlesed_v4.hdf5'
master = analyse.analyse(filename)

flares.list_datasets()
# print('-'*80)
# master.list_datasets()

print(flares.tags)


fig = plt.figure(figsize = (3.5, 3.5))

left  = 0.15
height = 0.8
bottom = 0.15
width = 0.8

ax = fig.add_axes((left, bottom, width, height))

ax.axhline(1.0, c='k', lw=2, alpha=0.1)

# bin_edges = np.arange(*X_limits, binw)
# bin_centres = bin_edges[:-1]+binw/2

tag = '010_z005p000'


Mstar = flares.load_dataset(tag, 'Galaxy', 'Mstar')
# Mstar_ = master.load_dataset(tag, 'Galaxy', 'Mstar')
Mdm = flares.load_dataset(tag, 'Galaxy', 'Mdm')
Nbh = flares.load_dataset(tag, 'Galaxy', 'BH_Length')


log10Ms = []
frac = []

for i, (sim, w) in enumerate(zip(flares.sims, flares.weights)):

    Mbhs = flares.load_particles(sim, tag, 'BH_Mass', return_dict = False)

    log10Mstar = np.log10(Mstar[sim]) + 10.
    log10Mdm = np.log10(Mdm[sim]) + 10.

    for j, log10M in enumerate(log10Mstar):

        if len(Mbhs[j])>0:
            
            maxMbh = np.max(Mbhs[j])
            sumMbh = np.sum(Mbhs[j])

        if np.log10(maxMbh)+10>7.:

            log10Ms.append(log10M)
            # log10Ms.append(np.log10(maxMbh)+10.)
            frac.append(maxMbh/sumMbh)


frac = np.array(frac)

print(np.sum(frac>0.9)/len(frac))


ax.scatter(log10Ms, frac, s=1, c='k', alpha=0.2)




ax.set_xlim(X_limits)
ax.set_ylim(Y_limits)

ax.set_ylabel(r'$\rm\ max(M_{\bullet})/\sum M_{\bullet}$')
ax.set_xlabel(r'$\rm \log_{10}(M_{\star}/M_{\odot})$')

fig.savefig(f'figs/Mh_maxfrac.pdf')
fig.clf()