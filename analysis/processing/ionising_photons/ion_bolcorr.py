import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt


from scipy.integrate import simps, trapz

from scipy.interpolate import interp1d

import flares

import astropy.constants as constants
import astropy.units as units

import FLARE.photom as photom
import FLARE.plt as fplt

spectrum_index = 2

FUV = 1500
spectrum_parts = {'uv':[1400, 1600], 'optical':[4000, 8000], 'ionising':[0, 912]}
spectrum_part = list(spectrum_parts.keys())[spectrum_index]
lims = spectrum_parts[spectrum_part]

output_dir_bol = '../cloudy/output/linear/total_1000'

AGN_T = np.linspace(10000, 1.5e6, 1000)  # range of AGN temperatures to model



fig = plt.figure(figsize=(3,3))
left  = 0.1
bottom = 0.1
width = 0.75
height = 0.75
ax = fig.add_axes((left, bottom, width, height))

bol_corr_ion, bol_corr_optical, bol_corr_uv, bol_corr_uv_int = np.array([]), np.array([]), np.array([]), np.array([])


for i in range(len(AGN_T)):
    lam_bol, incident_bol, transmitted_bol, nebular_bol, total_bol, linecont_bol = np.loadtxt(f'{output_dir_bol}/{i+1}.cont', delimiter='\t',
                                                                      usecols=(0, 1, 2, 3, 4, 8)).T

    nu = lambda l: 3E8 / (l * 1E-10)
    lam_nu = lam_bol[::-1]
    tot_nu = total_bol[::-1]
    int_nu = incident_bol[::-1]

    s = (lam_bol[::-1] > lims[0])&(lam_bol[::-1]<lims[1])

    total_bol = total_bol/lam_bol
    incident_bol = incident_bol/lam_bol

    bol_corr_ion = np.append(bol_corr_ion, simps(int_nu[s]/lam_nu[s], lam_nu[s])/10**43)
    #print(simps(int_nu/lam_nu, lam_nu)/10**43)


ax.plot(np.log10(AGN_T), bol_corr_ion, c='k', ls='-', label='ion')

ax.set_ylabel(r"$\rm L_{\lambda, ion} \; / \; L_{bol}$")
ax.set_xlabel(r"$\rm log_{10}[T_{AGN} \; /\; K]$")
#ax.set_xlabel(r"$\rm T_{AGN} \; /\; 10^{5}\;K$")

#ax.legend(loc='best', prop={'size': 6})

fig.savefig(f'figures/bolometric_correction_ion_test.pdf', bbox_inches='tight')