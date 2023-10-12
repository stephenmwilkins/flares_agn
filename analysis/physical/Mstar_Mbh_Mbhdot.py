
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

x_limits = [5., 10.]
# x_limits = [28.01, 30.49]

filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'

flares = analyse.analyse(filename, default_tags=False)

flares.list_datasets()

tags = ['005_z010p000', '006_z009p000', '007_z008p000',
        '008_z007p000', '009_z006p000', '010_z005p000']
redshifts = [10, 9, 8, 7, 6, 5]


N = len(redshifts)
left = 0.1
top = 0.95
bottom = 0.1
right = 0.975
panel_width = (right-left)/N
panel_height = top-bottom
fig, axes = plt.subplots(2, 3, figsize = (7,5), sharey = True, sharex = True)
plt.subplots_adjust(left=left, top=top, bottom=bottom, right=right, wspace=0.0, hspace=0.1)





x = 'Mstar'
y = 'BH_Mass'
z = 'BH_Mdot'

xlimits = [8., 12.]
ylimits = [5., 9.]

norm = Normalize(vmin=-10., vmax=0.5)
cmap = cm.plasma

# ----------------------------------------------------------------------
# --- define quantities to read in [not those for the corner plot, that's done later]

quantities = []
quantities.append({'path': 'Galaxy', 'dataset': f'Mstar_30',
                  'name': 'Mstar', 'log10': True})
quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mass',
                  'name': 'BH_Mass', 'log10': True})
quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mdot',
                  'name': 'BH_Mdot', 'log10': True})

D = {}
s = {}



for tag, redshift, ax in zip(tags, redshifts, axes.flatten()):

    # --- get quantities (and weights and deltas)
    D = flares.get_datasets(tag, quantities)

    print(np.median(D['log10Mstar']))

    s = D['log10Mstar']>9.


    print(np.median(D['log10BH_Mass'][s]))
    print(np.median(D['log10BH_Mdot'][s]))

    ax.scatter(D['log10Mstar'][s], D['log10BH_Mass'][s], s=1, c=cmap(norm(D['log10BH_Mdot'][s])))

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

