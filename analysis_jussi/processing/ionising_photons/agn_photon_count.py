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

import _pickle as pickle

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
    lam_nu = lam_bol #[::-1]
    tot_nu = total_bol[::-1]
    int_nu = incident_bol*lam_nu/(3E18) #[::-1]

    #s = (lam_bol[::-1] > lims[0])&(lam_bol[::-1]<lims[1])
    s = (lam_bol > lims[0]) & (lam_bol < lims[1])

    total_bol = total_bol/lam_bol
    incident_bol = incident_bol/lam_bol

    #print(simps(np.flip(incident_bol), np.flip(lam_bol)), simps(int_nu, nu(lam_nu)))

    conv = ((constants.h*constants.c/((lam_nu[s]*units.AA).to(units.m))).to(units.erg)).value
    #bol_corr_ion = np.append(bol_corr_ion, simps(int_nu[s]/(lam_nu[s]*conv), lam_nu[s])/np.interp(1500., lam_nu, int_nu))

    bol_corr_ion = np.append(bol_corr_ion,
                             (simps(int_nu[s] / (conv), nu(lam_nu[s])) / np.interp(nu(1500.), nu(lam_nu), int_nu))/1E43)

    #bol_corr_ion = np.append(bol_corr_ion,
    #                         np.sum(int_nu[s] / (lam_nu[s] * conv)) / np.interp(1500., lam_nu, int_nu))
    #print(simps(int_nu/lam_nu, lam_nu)/10**43)


ax.plot(np.log10(AGN_T), np.log10(bol_corr_ion), c='k', ls='-', label='ion')

ax.set_ylabel(r"$\rm log_{10}[\xi_{ion} \; / \; erg^{-1} \; Hz]$")
ax.set_xlabel(r"$\rm log_{10}[T_{AGN} \; /\; K]$")
#ax.set_xlabel(r"$\rm T_{AGN} \; /\; 10^{5}\;K$")

#ax.legend(loc='best', prop={'size': 6})

fig.savefig(f'figures/numphot_ion_xi_bolcorr.pdf', bbox_inches='tight')

output = {'xi': bol_corr_ion, 'T_AGN': AGN_T}

pickle.dump(output, open('xi_corr_test_1.p', 'wb'))