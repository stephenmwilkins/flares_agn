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

from scipy.integrate import simps

uvlims = [1e3, 1e4]

import astropy.units as u
import astropy.constants as c
lower_lim = ((c.c/(((1 * u.Ry).to(u.J))/c.h)).to(u.AA)).value
upper_lim = ((c.c/((((7.354e6) * u.Ry).to(u.J))/c.h)).to(u.AA)).value

uvlims = [upper_lim, lower_lim] # limits Ross used (trying to see what he actually did)

cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin = np.log10(1e4), vmax = np.log10(1.5e6))

output_dir_bol = '../../cloudy/output/linear/total'
output_dir_uv = '../../cloudy/output/linear/total'

AGN_T = np.linspace(10000, 1.5e6, 100)  # range of AGN temperatures to model


fig = plt.figure(figsize=(3,3))
left  = 0.1
bottom = 0.1
width = 0.75
height = 0.75
ax = fig.add_axes((left, bottom, width, height))

y = np.array([])

for i in range(len(AGN_T)):
    lam_bol, incident_bol, transmitted_bol, nebular_bol, total_bol, linecont_bol = np.loadtxt(f'{output_dir_bol}/{i+1}.cont', delimiter='\t',
                                                                      usecols=(0, 1, 2, 3, 4, 8)).T

    s = (lam_bol > uvlims[0])&(lam_bol < uvlims[1])

    y = np.append(y, simps(np.flip(total_bol[s]/lam_bol[s]), np.flip(lam_bol[s]))/simps(np.flip(total_bol/lam_bol), np.flip(lam_bol)))

print(y)

ax.plot(AGN_T/10**5, y)

'''
cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
cmapper.set_array([])

cax = fig.add_axes([width+left, bottom, 0.05, height])
bar = fig.colorbar(cmapper, cax=cax, orientation='vertical', format='%d')
bar.set_ticks([4, 5, 6])
cax.set_ylabel(r'$\rm log_{10}[T_{AGN} \; /\; K]$')
'''

#ax.set_ylim(25, 43)
#ax.set_xlim(-4, 6.2)

ax.set_ylabel(r"$\rm F_{\lambda, uv} \; / \; F_{\lambda, bol}$")
ax.set_xlabel(r"$\rm T_{AGN} \; /\; 10^{5}\;K$")

fig.savefig(f'figures/spectra/l_bol__l_uv__ratio_Ross_total.pdf', bbox_inches='tight')