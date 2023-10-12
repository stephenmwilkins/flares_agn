
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import cmasher as cmr
import flares_utility.analyse as analyse
import flare.plt as fplt

binw = 0.25
X_limits = [6.51, 9.99]
Y_limits = [-6.49, -2.51]


filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'
flares = analyse.analyse(filename)

fig = plt.figure(figsize = (3.5, 3.5))

left  = 0.15
height = 0.7
bottom = 0.15
width = 0.8

ax = fig.add_axes((left, bottom, width, height))
# ax2 = ax.twinx()
cax = fig.add_axes([left, bottom+height, width, 0.03])



# --- FLARES


# flares.list_datasets()

V = (4./3) * np.pi * (flares.radius)**3 # Mpc^3


bin_edges = np.arange(*X_limits, binw)
bin_centres = bin_edges[:-1]+binw/2

quantity = ['Galaxy', 'BH_Mass']


norm = Normalize(vmin=-0.35, vmax=0.35)
cmap = cmr.get_sub_cmap('cmr.copper', 0.2, 0.8)


z = 5.0
tag = '010_z005p000'

tag = flares.tag_from_zed[z]
X = flares.load_dataset(tag, *quantity)

for i, (sim, w, delta) in enumerate(zip(flares.sims, flares.weights, flares.deltas)):

    x = np.log10(np.array(X[sim])) + 10.
    x = x[x>0.0]

    N, _ = np.histogram(x, bins = bin_edges)

    phi = (N / V) / binw

    ax.plot(bin_centres, np.log10(phi), ls = '-', c=cmap(norm(np.log10(1+delta))), lw=1)


ax.axhline(np.log10(1/V/binw), c='k', alpha=0.1)

# total weighted M_BH mass function
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

ax.plot(bin_centres, np.log10(phi), ls = '--', c='k', lw=1.5)
ax.plot(bin_centres[N>4], np.log10(phi[N>4]), ls = '-', c='k', lw=1.5, label = rf'$\rm z={z}$')





ax.set_xlim(X_limits)
ax.set_ylim(Y_limits)

ax.set_xlabel(r'$\rm \log_{10}(M_{\bullet}/M_{\odot})$')
ax.set_ylabel(r'$\rm\log_{10}[\phi/Mpc^{-3}\ dex^{-1}]$')

# add colourbar
cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(cmapper, cax=cax, orientation='horizontal')
cax.xaxis.tick_top()
cax.xaxis.set_label_position('top')
cax.set_xlabel(r'$\rm log_{10}(1+\delta_{14})$', fontsize=7)
cax.tick_params(axis='x', labelsize=6)


fig.savefig(f'figs/Mbh_DF_environment.pdf')


fig.clf()
