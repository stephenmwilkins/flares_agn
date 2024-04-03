
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

binw = 0.25
X_limits = [6.51, 9.49]
Y_limits = [-8.01, -3.49]




tag = '010_z005p000'


fig = plt.figure(figsize = (3.5, 3.5))

left  = 0.15
height = 0.8
bottom = 0.15
width = 0.8

ax = fig.add_axes((left, bottom, width, height))

bin_edges = np.arange(*[6.5, 10.], binw)
bin_centres = bin_edges[:-1]+binw/2


## original method

filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'
flares = analyse.analyse(filename)
V = (4./3) * np.pi * (flares.radius)**3 # Mpc^3
phi = np.zeros(len(bin_centres))
N = np.zeros(len(bin_centres))
X = flares.load_dataset(tag, 'Galaxy', 'BH_Mass')
for i, (sim, w) in enumerate(zip(flares.sims, flares.weights)):
    x = np.log10(np.array(X[sim])) + 10.
    x = x[x>0.0]
    N_temp, _ = np.histogram(x, bins = bin_edges)
    N += N_temp
    phi_temp = (N_temp / V) / binw
    phi += phi_temp * w

ax.plot(bin_centres, np.log10(phi), ls = '-', c='k', lw=3, alpha=0.5, label = 'original')

## sum

filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_bh.hdf5'
flares = analyse.analyse(filename)
V = (4./3) * np.pi * (flares.radius)**3 # Mpc^3
phi = np.zeros(len(bin_centres))
N = np.zeros(len(bin_centres))

for i, (sim, w) in enumerate(zip(flares.sims, flares.weights)):

    X = flares.load_particles(sim, tag, 'BH_Mass', return_dict=False)
    log10X =  np.log10(np.array([np.sum(x) for x in X]))+10.

    N_temp, _ = np.histogram(log10X, bins = bin_edges)
    N += N_temp
    phi_temp = (N_temp / V) / binw
    phi += phi_temp * w

ax.plot(bin_centres, np.log10(phi), ls = '--', c='r', lw=1.5, label = 'sum')


## max

filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_bh.hdf5'
flares = analyse.analyse(filename)
V = (4./3) * np.pi * (flares.radius)**3 # Mpc^3
phi = np.zeros(len(bin_centres))
N = np.zeros(len(bin_centres))

for i, (sim, w) in enumerate(zip(flares.sims, flares.weights)):

    X = flares.load_particles(sim, tag, 'BH_Mass', return_dict=False)
    
    log10X = []
    for x in X:
        if len(x)>0:
            log10X.append(np.log10(np.max(x))+10.)

    N_temp, _ = np.histogram(log10X, bins = bin_edges)
    N += N_temp
    phi_temp = (N_temp / V) / binw
    phi += phi_temp * w

ax.plot(bin_centres, np.log10(phi), ls = '--', c='g', lw=1.5, label = 'max')



ax.legend(fontsize=7, labelspacing=0.0)
ax.set_xlim(X_limits)
ax.set_ylim(Y_limits)

ax.set_ylabel(r'$\rm\log_{10}(\phi/Mpc^{-3}\ dex^{-1})$')
ax.set_xlabel(r'$\rm \log_{10}(M_{\bullet}/M_{\odot})$')

fig.savefig(f'figs/Mbh_DF_methodcomparison.pdf')
fig.clf()
