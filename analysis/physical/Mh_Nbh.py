
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib.colors import Normalize

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
X_limits = [8., 14.]
Y_limits = [0, 29.]
# Y_limits = [8., 14.]

filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_bh.hdf5'
flares = analyse.analyse(filename)

filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_noparticlesed_v4.hdf5'
master = analyse.analyse(filename)

flares.list_datasets()
print('-'*80)
master.list_datasets()

print(flares.tags)


fig = plt.figure(figsize = (3.5, 3.5))

left  = 0.15
height = 0.8
bottom = 0.15
width = 0.8

ax = fig.add_axes((left, bottom, width, height))



# bin_edges = np.arange(*X_limits, binw)
# bin_centres = bin_edges[:-1]+binw/2

tag = '010_z005p000'


Mstar = flares.load_dataset(tag, 'Galaxy', 'Mstar')
# Mstar_ = master.load_dataset(tag, 'Galaxy', 'Mstar')
Mdm = flares.load_dataset(tag, 'Galaxy', 'Mdm')
Nbh = flares.load_dataset(tag, 'Galaxy', 'BH_Length')


norm = Normalize(vmin=5.0, vmax=9.0)
cmap = cmr.ember

Nbh7 = np.array([])
log10Mstar = np.array([])
log10Mbh = np.array([])

for i, (sim, w) in enumerate(zip(flares.sims, flares.weights)):

    Mbhs = flares.load_particles(sim, tag, 'BH_Mass', return_dict = False)

    log10Mbh = np.hstack([log10Mbh, np.log10(np.array([np.sum(x) for x in Mbhs]))+10.])
    log10Mstar = np.hstack([log10Mstar, np.log10(Mstar[sim]) + 10.])
    Nbh7 = np.hstack([Nbh7, np.array([np.sum(np.log10(x)+10>7.) for x in Mbhs])])


print(len(log10Mstar))
print(len(Nbh7))
print(len(log10Mbh))

ax.scatter(log10Mstar, Nbh7, s=2, alpha=0.5, label = 'N(M_{\bullet}>10^{7}\ M_{\odot})', c=cmap(norm(log10Mbh)))
   
print(np.sum(Nbh7>2)/np.sum(Nbh7>0))

# ax.legend(fontsize=8)
ax.set_xlim(X_limits)
ax.set_ylim(Y_limits)

ax.set_ylabel(r'$\rm\ N_{BH}$')
ax.set_xlabel(r'$\rm \log_{10}(M_{\star}/M_{\odot})$')

fig.savefig(f'figs/Mh_Nbh.png')
fig.clf()