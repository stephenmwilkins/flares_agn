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



X_limits = [5, 10.]
Y_limits = [-8.01, -3.49]


filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'
flares = analyse.analyse(filename)



fig = plt.figure(figsize = (3.5, 3.5))

left  = 0.15
height = 0.8
bottom = 0.15
width = 0.8

ax = fig.add_axes((left, bottom, width, height))

# --- FLARES


# flares.list_datasets()

V = (4./3) * np.pi * (flares.radius)**3 # Mpc^3


quantity = ['Galaxy', 'BH_Mass']


n7 = np.zeros(len(redshifts))
n8 = np.zeros(len(redshifts))

for j, z in enumerate(redshifts):

    print(z,'-'*30)
    tag = flares.tag_from_zed[z]
    X = flares.load_dataset(tag, *quantity)

    N7 = 0.0
    N8 = 0.0
    
    for i, (sim, w) in enumerate(zip(flares.sims, flares.weights)):

        x = np.log10(np.array(X[sim])) + 10.

        N_ = np.sum(x>7)
        N7 += N_*w/V

        N_ = np.sum(x>8)
        N8 += N_*w/V

    n7[j] = N7
    n8[j] = N8

ax.plot(redshifts, np.log10(n7), ls = '-', c='k', lw=1, label = r'$\rm FLARES\ M_{\bullet}>10^{7}\ M_{\odot}$')
ax.plot(redshifts, np.log10(n8), ls = '-', c='k', lw=2, label = r'$\rm FLARES\ M_{\bullet}>10^{8}\ M_{\odot}$')



# eagle

filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/EAGLE_REF_sp_info.hdf5'
eagle = analyse.analyse(filename)

hf = h5py.File(filename, 'r')
# hf.visit(print)

N = []

tags = [
    '002_z009p993',
    '003_z008p988',
    '004_z008p075',
    '005_z007p050',
    '006_z005p971',
    '008_z005p037',    
]
redshifts = [
    9.993,
    8.988,
    8.075,
    7.050,
    5.971,
    5.037,
]

for limit, lw in zip([7, 8], [1,2]):
    N = []
    for tag, redshift in zip(tags, redshifts):

        X = np.log10(hf[f'{tag}/Galaxy/BH_Mass'][()]) + 10.
        n = np.sum(X>limit)
        N.append(n)

    ax.plot(redshifts, np.log10(np.array(N)/(100**3)), c='0.5', lw=lw, ls='--', label = rf'$\rm EAGLE\ M_{{\bullet}}>10^{limit}\ M_{{\odot}}$')





ax.legend(fontsize=8)
ax.set_xlim(X_limits)
ax.set_ylim(Y_limits)


ax.set_ylabel(r'$\rm\log_{10}(n/Mpc^{-3})$')
ax.set_xlabel(r'$\rm z$')

fig.savefig(f'figs/Mbh_n.pdf')
fig.clf()
