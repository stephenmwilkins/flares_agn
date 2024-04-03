
import numpy as np
import matplotlib.cm as cm
import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
import scipy.stats as stats
from astropy.table import Table
import cmasher as cmr
import h5py
import flare.photom as phot
from flare.photom import M_to_lum
import flares_utility.limits
import flares_utility.plt
import flares_utility.analyse as analyse
import flares_utility.stats
from unyt import c, Msun, yr, Lsun
import utils as u

# set style
plt.style.use('../matplotlibrc.txt')



filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'

flares = analyse.analyse(filename, default_tags=False)

tag = '010_z005p000'

quantities = []

quantities.append({'path': 'Galaxy', 'dataset': f'Mstar_30',
                'name': 'Mstar', 'log10': True})
quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mass',
                'name': 'BH_Mass', 'log10': True})
quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mdot',
                'name': 'BH_Mdot', 'log10': True})

D = flares.get_datasets(tag, quantities)
s = D['log10Mstar']>9.

stellar_mass =  D['Mstar'][s] * Msun
blackhole_accretion_rate = D['BH_Mdot'][s] * u.accretion_rate_units
blackhole_mass = D['BH_Mass'][s] * u.blackhole_mass_units 

eddington_accretion_rate = u.calculate_eddington_accretion_rate(blackhole_mass)
bolometric_luminosity = u.calcualte_bolometric_luminosity(blackhole_accretion_rate)

eddington_ratio = blackhole_accretion_rate/eddington_accretion_rate





base_size = 3.5
left = 0.15
height = 0.7
bottom = 0.15
width = 0.8

fig = plt.figure(figsize=(base_size, base_size*width/height))

ax = fig.add_axes((left, bottom, width, height))
cax = fig.add_axes([left, bottom+height, width, 0.03])


ax.fill_between([0,20],[0,0],[7,7], color='k',alpha=0.05)


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

# plot flares
ax.scatter(np.log10(stellar_mass.to('Msun')), 
           np.log10(blackhole_mass), 
           s=1, 
           c=cmap(norm(np.log10(eddington_ratio))))


# Kormendy and Ho (read off Hazboulit)
x = np.array([9., 12.])
y = np.array([6.3, 9.8])
ax.plot(x,y,c='k',lw=1,ls='--',label=r'$\rm Kormendy\ and\ Ho\ (2013)\ [z=0]$')

# Reines and Volunte (2015)
x = np.array([9., 12.])
alpha = 7.45 
beta = 1.05
y = alpha + beta * (x-11.)
ax.plot(x,y,c='k',lw=1,ls='-.',label=r'$\rm Reines\ and\ Volonteri\ (2015)\ [z=0]$')




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

ax.legend(fontsize=7)

filename = f'figs/Mstar_Mbh_Mbhdot_5.pdf'
fig.savefig(filename)
print(filename)

