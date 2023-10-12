
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

import cmasher as cmr

import h5py

import flare.plt as fplt
import flare.photom as phot
from flare.photom import M_to_lum
import flares_utility.analyse as analyse
from flares_utility.plt import measure_weighted_median, weighted_median

from unyt import c, Msun, yr, g, s

tags = ['005_z010p000', '006_z009p000', '007_z008p000',
        '008_z007p000', '009_z006p000', '010_z005p000']
redshifts = [10, 9, 8, 7, 6, 5]
xlimits = [44.51, 46.49]
ylimits = [0.001, 1.1]

binw =  0.25
bins = np.arange(44., 47., binw)
bin_centres = 0.5*(bins[:-1]+bins[1:])


filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'
flares = analyse.analyse(filename)



fig = plt.figure(figsize = (3.5, 3.5))

left  = 0.15
height = 0.8
bottom = 0.15
width = 0.8

ax = fig.add_axes((left, bottom, width, height))


# converting MBHacc units to M_sol/yr

conversion = 0.1 * Msun * c**2 / yr
log10conversion = np.log10(conversion.to('erg/s').value)



quantities = []
quantities.append({'path': 'Galaxy', 'dataset': f'Mstar_30',
                  'name': 'Mstar', 'log10': True})
quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mass',
                  'name': 'BH_Mass', 'log10': True})
quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mdot',
                  'name': 'BH_Mdot', 'log10': True})
# quantities.append({'path': 'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic/', 'dataset': f'FUV',
#                   'name': 'FUV_Intrinsic', 'log10': False})
quantities.append({'path': 'Galaxy/BPASS_2.2.1/Chabrier300/Indices/Lbol', 'dataset': 'DustModelI',
                  'name': 'Lstars', 'log10': True})



for z, c in zip(redshifts, cmr.take_cmap_colors('cmr.bubblegum_r', len(redshifts))):

    
    tag = flares.tag_from_zed[z]

    D = flares.get_datasets(tag, quantities)

    log10Mstar = D['log10Mstar']
    log10Lbhs = D['log10BH_Mdot'] + log10conversion
    log10Lstars = D['log10Lstars']

    log10Ltotal = np.log10(10**log10Lbhs + 10**log10Lstars)


    Ntot, _ = np.histogram(log10Ltotal, bins=bins)

    s = log10Lbhs > log10Lstars

    Nagn, _ = np.histogram(log10Ltotal[s], bins=bins)

    f = Nagn/Ntot

    ax.plot(bin_centres, f, lw=2, c=c)

    s = log10Lbhs > log10Lstars-1.
    Nagn, _ = np.histogram(log10Ltotal[s], bins=bins)

    f10 = Nagn/Ntot

    ax.plot(bin_centres, f10, lw=1, c=c, ls='--')


ax.set_xticks([45., 45.5, 46.])
ax.set_yscale('log')
ax.set_xlim(xlimits)
ax.set_ylim(ylimits)


ax.set_ylabel(r'$\rm f$')
ax.set_xlabel(r'$\rm\log_{10}(L_{\bullet}+L_{\star}/erg\ s^{-1})$')


handles = []
handles.append(Line2D([0], [0], label=r'$\rm L_{\bullet}>L_{\star}$', lw=2, ls='-', c='k'))
handles.append(Line2D([0], [0], label=r'$\rm L_{\bullet}>0.1\times L_{\star}$', lw=1, ls='--', c='k'))

for z, c in zip(redshifts, cmr.take_cmap_colors('cmr.bubblegum_r', len(redshifts))):
    handles.append(Line2D([0], [0], label=rf'$\rm z={z}$', color=c, lw=1))


ax.legend(handles=handles, fontsize=8, labelspacing=0.1)



fig.savefig(f'figs/fraction_Lbol.pdf')


fig.clf()
