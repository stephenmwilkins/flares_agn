
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

from unyt import c, Msun, yr, Lsun

filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'

flares = analyse.analyse(filename, default_tags=False)

flares.list_datasets()

tag = '010_z005p000'
redshifts = [10, 9, 8, 7, 6, 5]





base_size = 3.5
left = 0.15
height = 0.7
bottom = 0.15
width = 0.8

fig = plt.figure(figsize=(base_size, base_size*width/height))

ax = fig.add_axes((left, bottom, width, height))
cax = fig.add_axes([left, bottom+height, width, 0.03])


ax.fill_between([0,20],[0,0],[7,7], color='k',alpha=0.05)


x = 'Mstar'
y = 'BH_Mass'
z = 'BH_Mdot'

xlimits = np.array([9.1, 11.9])
ylimits = np.array([5.1, 9.9])


for ratio, lw in zip([-3., -2., -1.], [1,2,3,4]):
    
    y = xlimits+ratio
    ax.plot(xlimits, y, c='k', alpha=0.1, lw=lw)

    m = (y[1]-y[0])/(xlimits[1]-xlimits[0])
    aspect = (ylimits[1]-ylimits[0])/(xlimits[1]-xlimits[0])
    angle = np.arctan(m/aspect)
    x = 9.2
    ax.text(x, x + ratio + 0.15, rf'$\rm M_{{\bullet}}/M_{{\star}}={10**ratio}$', rotation=180*angle/np.pi, fontsize = 8, c='0.5')

norm = Normalize(vmin=-10., vmax=0.5)
cmap = cmr.voltage

# ----------------------------------------------------------------------
# --- define quantities to read in [not those for the corner plot, that's done later]

quantities = []
quantities.append({'path': 'Galaxy', 'dataset': f'Mstar_30',
                  'name': 'Mstar', 'log10': True})
quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mass',
                  'name': 'BH_Mass', 'log10': True})
quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mdot',
                  'name': 'BH_Mdot', 'log10': True})



D = flares.get_datasets(tag, quantities)

print(np.median(D['log10Mstar']))

s = D['log10Mstar']>9.

conversion = 0.1 * Msun * c**2 / yr
log10conversion = np.log10(conversion.to('erg/s').value)


log10Lbol = D['log10BH_Mdot'] + log10conversion
log10Ledd = np.log10(3.2) + 4 + np.log10((1*Lsun).to('erg/s').value) + D['log10BH_Mass']

log10LbolLedd = log10Lbol - log10Ledd

print(np.min(log10LbolLedd), np.max(log10LbolLedd))


ax.scatter(D['log10Mstar'][s], D['log10BH_Mass'][s], s=1, c=cmap(norm(log10LbolLedd[s])))

ax.set_ylim(ylimits)
ax.set_xlim(xlimits)

ax.set_xlabel(r'$\rm log_{10}(M_{\star}/M_{\odot})$')
ax.set_ylabel(r'$\rm log_{10}(M_{\bullet}/M_{\odot})$')


# add colourbar
cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(cmapper, cax=cax, orientation='horizontal')
cax.xaxis.tick_top()
cax.xaxis.set_label_position('top')
cax.set_xlabel(r'$\rm log_{10}(L_{bol}/L_{Edd})$', fontsize=8)
cax.tick_params(axis='x', labelsize=6)



filename = f'figs/Mstar_Mbh_Mbhdot_5.pdf'
fig.savefig(filename)
print(filename)

