
import numpy as np
import matplotlib.cm as cm
import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
import scipy.stats as stats
import cmasher as cmr
import h5py
import flare.photom as phot
from flare.photom import M_to_lum
import flares_utility.limits
import flares_utility.plt
import flares_utility.analyse as analyse
import flares_utility.stats
import utils as u
from unyt import Msun

# set style
plt.style.use('../matplotlibrc.txt')

x_limits = [5., 10.]


# Set up plots
N = len(redshifts)
left = 0.1
top = 0.95
bottom = 0.1
right = 0.975
panel_width = (right-left)/N
panel_height = top-bottom
fig, axes = plt.subplots(2, 3, figsize = (7,5), sharey = True, sharex = True)
plt.subplots_adjust(left=left, top=top, bottom=bottom, right=right, wspace=0.0, hspace=0.1)


filename='/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'
flares = analyse.analyse(filename, default_tags=False)

tags = ['005_z010p000', '006_z009p000', '007_z008p000',
        '008_z007p000', '009_z006p000', '010_z005p000']
redshifts = [10, 9, 8, 7, 6, 5]


quantities = []

quantities.append({'path': 'Galaxy', 'dataset': f'Mstar_30',
                'name': 'Mstar', 'log10': True})
quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mass',
                'name': 'BH_Mass', 'log10': True})
quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mdot',
                'name': 'BH_Mdot', 'log10': True})






for tag, redshift, ax in zip(tags, redshifts, axes.flatten()):

    D = flares.get_datasets(tag, quantities)
    s = D['log10Mstar']>9.

    stellar_mass =  D['Mstar'][s] * Msun
    blackhole_accretion_rate = D['BH_Mdot'][s] * u.accretion_rate_units
    blackhole_mass = D['BH_Mass'][s] * u.blackhole_mass_units 

    eddington_accretion_rate = u.calculate_eddington_accretion_rate(blackhole_mass)
    bolometric_luminosity = u.calcualte_bolometric_luminosity(blackhole_accretion_rate)

    eddington_ratio = blackhole_accretion_rate/eddington_accretion_rate


    ax.scatter(np.log10(stellar_mass), 
               np.log10(blackhole_mass), 
               s=1, 
               c=cmap(norm(np.log10(blackhole_accretion_rate))))

    ax.set_ylim(ylimits)
    ax.set_xlim(xlimits)



# limits[y] = [-10, 0.5]
# limits[x] = x_limits

# cmap = cmr.bubblegum

# fig, axes = flares_utility.plt.linear_redshift(
#     D, zeds, x, y, s, limits=limits, scatter=False, bins=15, rows=2, lowz=True, add_weighted_range=True)


# flaxes = axes.flatten()


# colors = cmr.take_cmap_colors('cmr.infinity', 5, cmap_range=(0.15, 0.85))


fig.savefig(f'figs/Mstar_Mbh_Mbhdot.pdf')

