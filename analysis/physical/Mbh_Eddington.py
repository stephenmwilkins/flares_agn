
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

import flare.plt as fplt
import flare.photom as phot
from flare.photom import M_to_lum
import flares_utility.limits
import flares_utility.plt
import flares_utility.analyse as analyse
import flares_utility.stats

from unyt import c, Msun, yr, Lsun

filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'

flares = analyse.analyse(filename, default_tags=False)

# flares.list_datasets()


tag = '010_z005p000' # z=5


quantities = []

quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mass',
                  'name': 'BH_Mass', 'log10': True})
quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mdot',
                  'name': 'BH_Mdot', 'log10': True})

D = {}
s = {}

# --- get quantities (and weights and deltas)
D = flares.get_datasets(tag, quantities)


xlimits = Mbh_limit = [6.51, 9.49] # Msun
ylimits = [0.00001, 1.2]

conversion = 0.1 * Msun * c**2 / yr
log10conversion = np.log10(conversion.to('erg/s').value)




fig = plt.figure(figsize = (3.5, 3.5))

left  = 0.15
height = 0.7
bottom = 0.15
width = 0.7

ax = fig.add_axes((left, bottom, width, height))
ax2 = ax.twinx()
cax = fig.add_axes([left, bottom+height, width, 0.03])



s = D['log10BH_Mass']>6.5


#eddington
log10Lbol_ = np.log10(3.2) + 4 + np.log10((1*Lsun).to('erg/s').value)

log10Lbol = log10Lbol_ + Mbh_limit

print(log10Lbol)


log10T = np.log10(2.24E9 * D['BH_Mdot'][s] ** (1 / 4) * D['BH_Mass'][s]**-0.5)

print(np.min(log10T), np.max(log10T))

norm = Normalize(vmin=4.01, vmax=5.99)
cmap = cmr.ember

n = norm


ax.fill_between([0,7],[-10,-10],[7,7], color='k',alpha=0.05)


Lbol = D['log10BH_Mdot'][s] + log10conversion
Ledd = D['log10BH_Mass'][s] + log10Lbol_

Eddington = Lbol - Ledd

print(np.median(Eddington))

ax.scatter(D['log10BH_Mass'][s], 10**Eddington, s=1, zorder=1, c=cmap(norm(log10T)))


ax.set_xlim(Mbh_limit)
ax.set_ylim(ylimits)
# ax2.set_ylim(Lbol_limit)
ax.set_yscale('log')

ax.set_xlabel(r'$\rm log_{10}(M_{\bullet}/M_{\odot})$')
ax.set_ylabel(r'$\rm \lambda$')


# add colourbar
cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(cmapper, cax=cax, orientation='horizontal')
cax.xaxis.tick_top()
cax.xaxis.set_label_position('top')
cax.set_xlabel(r'$\rm log_{10}(T_{BB}/K)$', fontsize=7)
cax.tick_params(axis='x', labelsize=6)


filename = f'figs/Mbh_Eddington.pdf'
print(filename)

fig.savefig(filename)

