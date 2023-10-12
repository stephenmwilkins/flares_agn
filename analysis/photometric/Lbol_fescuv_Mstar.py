
import numpy as np
import matplotlib.cm as cm
import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
import scipy.stats as stats
from scipy.stats import pearsonr

import cmasher as cmr

import h5py

from unyt import c, Msun, yr
import flare.photom as phot
from flare.photom import M_to_lum
import flares_utility.limits
import flares_utility.plt
import flares_utility.analyse as analyse
import flares_utility.stats

filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'

flares = analyse.analyse(filename, default_tags=False)

# flares.list_datasets()



tag = '010_z005p000'


# ----------------------------------------------------------------------
# --- define quantities to read in [not those for the corner plot, that's done later]

quantities = []
quantities.append({'path': 'Galaxy', 'dataset': f'Mstar_30',
                  'name': 'Mstar', 'log10': True})
quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mass',
                  'name': 'BH_Mass', 'log10': True})
quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mdot',
                  'name': 'BH_Mdot', 'log10': True})
quantities.append({'path': 'Galaxy', 'dataset': f'BH_los',
                  'name': 'BH_los', 'log10': True})



fig = plt.figure(figsize = (3.5, 3.5))

left  = 0.15
height = 0.8
bottom = 0.15
width = 0.8

ax = fig.add_axes((left, bottom, width, height))


x = 'Mstar'
y = 'BH_Mass'
z = 'BH_Mdot'


xlimits = np.array([43., 47.])
ylimits = np.array([0., 1.])



norm = Normalize(vmin=8., vmax=11.5)
cmap = cm.plasma

# ----------------------------------------------------------------------
# --- define quantities to read in [not those for the corner plot, that's done later]

quantities = []
quantities.append({'path': 'Galaxy', 'dataset': f'Mstar_30',
                  'name': 'Mstar', 'log10': True})
quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mass',
                  'name': 'BH_Mass', 'log10': True})
quantities.append({'path': 'Galaxy', 'dataset': f'BH_los',
                  'name': 'BH_los', 'log10': True})
quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mdot',
                  'name': 'BH_Mdot', 'log10': True})
quantities.append({'path': 'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic/', 'dataset': f'FUV',
                  'name': 'FUV_Intrinsic', 'log10': False})
quantities.append({'path': 'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI/', 'dataset': f'FUV',
                  'name': 'FUV_DustModelI', 'log10': False})
quantities.append({'path': 'Galaxy', 'dataset': f'DTM',
                  'name': 'DTM', 'log10': False})

D = flares.get_datasets(tag, quantities)


conversion = 0.1 * Msun * c**2 / yr
log10conversion = np.log10(conversion.to('erg/s').value)

Lbol = D['log10BH_Mdot'] + log10conversion


Astar = 2.5*np.log10(D['FUV_Intrinsic']/D['FUV_DustModelI'])

kappa = 0.0795
gamma = -1.0
tau_uv = kappa * D['DTM'] * D['BH_los'] * (1500./5500.)**gamma

fesc_uv = np.exp(-tau_uv)




ax.scatter(Lbol, fesc_uv, s=1, c=cmap(norm(D['log10Mstar'])), alpha = 0.2)


ax.set_ylim(ylimits)
ax.set_xlim(xlimits)

ax.set_xlabel(r'$\rm log_{10}(L_{bol}/erg\ s^{-1})$')
ax.set_ylabel(r'$\rm f_{esc, FUV}$')

ax.legend()

fig.savefig(f'figs/Lbol_fescuv_Mstar.pdf')
