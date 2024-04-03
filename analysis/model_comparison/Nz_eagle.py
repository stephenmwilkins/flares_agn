
import numpy as np
import matplotlib.pyplot as plt
import h5py
import flares_utility.analyse as analyse

# set style
plt.style.use('../matplotlibrc.txt')


tags = ['005_z010p000', '006_z009p000', '007_z008p000',
        '008_z007p000', '009_z006p000', '010_z005p000']
redshifts = [10, 9, 8, 7, 6, 5]


binw = 0.25
X_limits = [6.51, 9.99]
Y_limits = [-8.5, -3.49]



fig = plt.figure(figsize = (3.5, 3.5))

left  = 0.15
height = 0.8
bottom = 0.15
width = 0.8

ax = fig.add_axes((left, bottom, width, height))

xlimits = np.array([5., 10.])
ylimits = np.array([1., 2000])

quantities = []
quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mass',
                  'name': 'BH_Mass', 'log10': True})



# flares

filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'
flares = analyse.analyse(filename)


for limit, lw in zip([7, 8], [1,2]):

    N = []
    for z, tag in zip(redshifts, tags):

        D = flares.get_datasets(tag, quantities)
        
        N.append(np.sum(D['log10BH_Mass']>limit))

    ax.plot(redshifts, N, c='k', lw=lw, ls='-', label = rf'$\rm FLARES\ M_{{\bullet}}>10^{limit}\ M_{{\odot}}$')


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

    ax.plot(redshifts, N, c='0.5', lw=lw, ls='--', label = rf'$\rm EAGLE\ M_{{\bullet}}>10^{limit}\ M_{{\odot}}$')


ax.legend(fontsize=8)
ax.set_ylim(ylimits)
ax.set_xlim(xlimits)
ax.set_yscale('log')
ax.set_xlabel(r'$\rm z$')
ax.set_ylabel(r'$\rm N(M_{\bullet}>10^7\ M_{\odot})$')

fig.savefig(f'figs/Nz.pdf')
