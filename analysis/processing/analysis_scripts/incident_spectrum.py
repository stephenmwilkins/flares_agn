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


cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin = np.log10(1e4), vmax = np.log10(1.5e6))

output_dir = '../cloudy/output/linear'

AGN_T = np.linspace(10000, 1.5e6, 100)  # range of AGN temperatures to model


fig = plt.figure(figsize=(3,3))
left  = 0.1
bottom = 0.1
width = 0.75
height = 0.75
ax = fig.add_axes((left, bottom, width, height))



for i in range(len(AGN_T)):
    lam, incident, transmitted, nebular, total, linecont = np.loadtxt(f'{output_dir}/{i+1}.cont', delimiter='\t',
                                                                      usecols=(0, 1, 2, 3, 4, 8)).T

    ax.plot(np.log10(lam), np.log10(incident),c = cmap(norm(np.log10(AGN_T[i]))))

cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
cmapper.set_array([])

cax = fig.add_axes([width+left, bottom, 0.05, height])
bar = fig.colorbar(cmapper, cax=cax, orientation='vertical', format='%d')
bar.set_ticks([4, 5, 6])
cax.set_ylabel(r'$\rm log_{10}[T_{AGN} \; /\; K]$')


ax.set_ylim(37, 45)
ax.set_xlim(-4, 6.2)

ax.set_ylabel(r"$\rm log_{10}[\lambda F_{\lambda} \; / \; erg \;s^{-1}]$")
ax.set_xlabel(r"$\rm log_{10}[\lambda \; / \; \AA]$")

fig.savefig(f'figures/spectra/incident.pdf', bbox_inches='tight')