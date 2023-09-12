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

spectrum_index = 0


# This part sets the integration range for the selected part of the spectrum
spectrum_parts = {'uv':[1400, 1600], 'optical':[4000, 8000]}
spectrum_part = list(spectrum_parts.keys())[spectrum_index]
lims = spectrum_parts[spectrum_part]

'''
# Ross' method for testing
import astropy.units as u
import astropy.constants as c
lower_lim = ((c.c/(((1 * u.Ry).to(u.J))/c.h)).to(u.AA)).value
upper_lim = ((c.c/((((7.354e6) * u.Ry).to(u.J))/c.h)).to(u.AA)).value

#uvlims = [upper_lim, lower_lim] # limits Ross used (trying to see what he actually did)
'''


cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin = np.log10(1e4), vmax = np.log10(1.5e6))

# directory of cloudy output
output_dir = '../../cloudy/output/linear/total'

AGN_T = np.linspace(10000, 1.5e6, 100)  # range of AGN temperatures to model


fig = plt.figure(figsize=(3,3))
left  = 0.1
bottom = 0.1
width = 0.75
height = 0.75
ax = fig.add_axes((left, bottom, width, height))

y = np.array([])
lbols = np.array([])
luvs = np.array([])

for i in range(len(AGN_T)):
    # read in cloudy
    lam_bol, incident_bol, transmitted_bol, nebular_bol, total_bol, linecont_bol = np.loadtxt(f'{output_dir}/{i+1}.cont', delimiter='\t',
                                                                      usecols=(0, 1, 2, 3, 4, 8)).T

    # getting the selected part of the spectrum
    s = (lam_bol > lims[0])&(lam_bol<lims[1])

    # storing bolometric luminosities and uv luminosities (this doesn't factor into anything, I just used it as sanity check)
    lbols = np.append(lbols, trapz(np.flip(total_bol/lam_bol), np.flip(lam_bol)))
    luvs = np.append(luvs, trapz(np.flip(total_bol[s]/lam_bol[s]), np.flip(lam_bol[s])))

    # calculating the ratio of L_uv/L_bol
    y = np.append(y, simps(total_bol[s]/lam_bol[s], lam_bol[s])/simps(total_bol/lam_bol, lam_bol))

print(y)

# plotting
ax.plot(np.log10(AGN_T), y, c='k', alpha=0.8)

ax.set_ylabel(r"$\rm F_{\lambda, uv} \; / \; F_{\lambda, bol}$")
ax.set_xlabel(r"$\rm log_{10}[T_{AGN} \; /\; K]$")


fig.savefig(f'figures/spectra/l_bol__l_uv__ratio_{spectrum_part}_test.pdf', bbox_inches='tight')

print('L_bol, AGN')
print(lbols)
print('L_uv, AGN')
print(luvs)

print(min(lbols), np.median(lbols), max(lbols))