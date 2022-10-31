import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt

import flares

import astropy.constants as constants
import astropy.units as units

import FLARE.photom as photom
import FLARE.plt as fplt

from scipy.integrate import simps, trapz

import astropy.constants as const
import astropy.units as u

from scipy.interpolate import interp1d

import _pickle as pickle

cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=5., vmax=10.)

def radial(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)

flares_dir = '../../../data/simulations'

fl = flares.flares(f'{flares_dir}/flares_no_particlesed.hdf5', sim_type='FLARES') #_no_particlesed
df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
weights = np.array(df['weights'])

halo = fl.halos

tags = fl.tags

BH_Coordinates = fl.load_dataset('BH_Coordinates', arr_type='Galaxy')
COP = fl.load_dataset('COP', arr_type='Galaxy')
Mstar = fl.load_dataset('Mstar', arr_type='Galaxy')

fig, axes = plt.subplots(2, 3, figsize = (6, 4), sharex = True, sharey=True)
fig.subplots_adjust(left=0.07, bottom=0.15, top=1.0, right=0.85, wspace=0.0, hspace=0.0)

for i, tag in enumerate(np.flip(fl.tags)):


    z = np.flip(fl.zeds)[i]
    ws, x, y, mstar = np.array([]), np.array([]), np.array([]), np.array([])
    for ii in range(len(halo)):
        s = (np.log10(Mstar[halo[ii]][tag])+10 > 8)#&(radial(*BH_Coordinates[halo[ii]][tag]) > 2000)
        ws = np.append(ws, np.ones(np.shape(Mstar[halo[ii]][tag][s]))*weights[ii])
        x = np.append(x, radial(*COP[halo[ii]][tag])[s])
        y = np.append(y, radial(*BH_Coordinates[halo[ii]][tag])[s])
        mstar = np.append(mstar, np.log10(Mstar[halo[ii]][tag][s])+10)

    ss = abs(x-y) < 100.

    ss_ = abs(x-y) > 100.
    print(sum(ss_), sum(ss_)/sum(ss))

    print(y[ss_])

    axes.flatten()[i].scatter(mstar[ss_], x[ss_] - y[ss_], color='k', s=2, alpha=0.7)
    #axes.flatten()[i].scatter(x[ss], x[ss]-y[ss], color='k', s=2, alpha=0.7)
    #axes.flatten()[i].axhline(0, c='k', ls='--')

    axes.flatten()[i].text(0.8, 0.9, r'$\rm z={0:.0f}$'.format(z), fontsize=8,
                               transform=axes.flatten()[i].transAxes,
                               color=cmap(norm(z)), ha='left')

fig.text(0.01, 0.55, r'$\rm \Delta loc$', ha = 'left', va = 'center', rotation = 'vertical', fontsize=10)
#fig.text(0.45,0.05, r'$\rm COP $', ha = 'center', va = 'bottom', fontsize=10)
fig.text(0.45,0.05, r'$\rm log_{10}[M_{*} \, / \, M_{\odot}] $', ha = 'center', va = 'bottom', fontsize=10)

fig.savefig(f'figures/coordinates_mstar.pdf', bbox_inches='tight')
fig.clf()