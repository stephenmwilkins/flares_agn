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

from scipy.interpolate import interp1d

import _pickle as pickle


def xi_ion(lam, sed, lims=[0, 912]):
    """
    :param lam: wavelength array in Å
    :param sed: SED as F_nu/(erg/s/Hz)
    :param lims: limits of integration in Å
    :return: ionising photon production efficiency
    """
    s = ((lam >= 0) & (lam < 912)).nonzero()[0]
    conv = 1.98644586e-08/(lam[s]**2*1E10)
    #conv = ((constants.h * constants.c / ((lam[s] * units.AA).to(units.m))).to(units.erg)).value  # alternative using astropy.units and astropy.constants

    xi = simps(sed[s]/conv, lam[s])/np.interp(1500, lam, sed)

    return xi

def lum_ion(lam, sed, lims=[0, 912]):
    """
    :param lam: wavelength array in Å
    :param sed: SED as F_nu/(erg/s/Hz)
    :param lims: limits of integration in Å
    :return: ionising photon production efficiency
    """
    s = ((lam >= lims[0]) & (lam < lims[1])).nonzero()[0]
    conv = (lam[s])
    #conv = ((constants.h * constants.c / ((lam[s] * units.AA).to(units.m))).to(units.erg)).value  # alternative using astropy.units and astropy.constants

    xi = simps(sed[s]/conv, lam[s])

    return xi


spectrum_index = 2

FUV = 1500
spectrum_parts = {'uv':[1400, 1600], 'optical':[4000, 8000], 'ion':[0, 912]}
spectrum_part = list(spectrum_parts.keys())[spectrum_index]
lims = spectrum_parts[spectrum_part]

import astropy.units as u
import astropy.constants as c
lower_lim = ((c.c/(((1 * u.Ry).to(u.J))/c.h)).to(u.AA)).value
upper_lim = ((c.c/((((7.354e6) * u.Ry).to(u.J))/c.h)).to(u.AA)).value

#uvlims = [upper_lim, lower_lim] # limits Ross used (trying to see what he actually did)

cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin = 0, vmax = 5)

output_dir_bol = '../cloudy/output/linear/total_1000'

AGN_T = np.linspace(10000, 1.5e6, 1000)  # range of AGN temperatures to model

y1, y2, y3, y4, y5, y6 = np.array([]),np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
lbols = np.array([])
luvs = np.array([])
linc = np.array([])

luv_integrate = np.array([])

fig = plt.figure(figsize=(3,3))
left  = 0.1
bottom = 0.1
width = 0.75
height = 0.75
ax = fig.add_axes((left, bottom, width, height))


bol_corr_optical = np.array([])
bol_corr_uv = np.array([])
bol_corr_uv_int = np.array([])
bol_corr_xi = np.array([])
bol_corr_ion = np.array([])
bol_corr_xi_nobolscale = np.array([])

for i in range(len(AGN_T)):
    lam_bol, incident_bol, transmitted_bol, nebular_bol, total_bol, linecont_bol = np.loadtxt(f'{output_dir_bol}/{i+1}.cont', delimiter='\t',
                                                                      usecols=(0, 1, 2, 3, 4, 8)).T

    nu = lambda l: 3E8 / (l * 1E-10)
    lam_nu = lam_bol[::-1]
    tot_nu = total_bol[::-1]
    int_nu = incident_bol[::-1]


    lam_fuv_idx = (np.abs(lam_bol - FUV)).argmin()

    s = (lam_bol > lims[0])&(lam_bol<lims[1])

    sss = (lam_nu > 0)&(lam_nu<1E10)#(lam_nu > lims[0])&(lam_nu<lims[1])

    total_bol = total_bol/lam_bol
    incident_bol = incident_bol/lam_bol

    #print(min(int_nu))
    #print(simps(np.flip(incident_bol), np.flip(lam_bol)), simps(int_nu[sss]/lam_nu[sss], lam_nu[sss]), lum_ion(lam_nu, int_nu), np.sum((incident_bol[1:]+incident_bol[:-1])/(lam_bol[1:]+lam_bol[:-1])*lam_bol[:-1]-lam_bol[1:]))

    lbols = np.append(lbols, simps(np.flip(total_bol), np.flip(lam_bol)))
    luvs = np.append(luvs, simps(np.flip(total_bol[s]), np.flip(lam_bol[s])))
    linc = np.append(lbols, simps(np.flip(incident_bol), np.flip(lam_bol)))

    total_bol_norm = total_bol / simps(np.flip(total_bol), np.flip(lam_bol))
    incident_bol_norm = incident_bol / simps(np.flip(incident_bol), np.flip(lam_bol))

    # Two measures using exactly 1500\AA (division by integrating over the total SED and just the input/incident SED)
    y1 = np.append(y1, total_bol_norm[lam_fuv_idx]/simps(np.flip(total_bol_norm), np.flip(lam_bol)))
    y2 = np.append(y2, total_bol_norm[lam_fuv_idx]/simps(np.flip(incident_bol_norm), np.flip(lam_bol)))

    # Two measures using average over the range 1400-1600\AA (division by integrating over the total SED and just the input/incident SED)
    y3 = np.append(y3, total_bol_norm[lam_fuv_idx] / simps(np.flip(total_bol_norm), np.flip(lam_bol)))
    y4 = np.append(y4, np.mean(total_bol_norm[s]) / simps(np.flip(incident_bol_norm), np.flip(lam_bol)))

    # Two measures using average over the range 1400-1600\AA (division by 10^43)
    y5 = np.append(y5, total_bol[lam_fuv_idx] / 10**43)
    y6 = np.append(y6, np.mean(total_bol[s]) / 10**43)

    bol_corr_uv = np.append(bol_corr_uv, (np.interp(1500., lam_nu, tot_nu) / 1E43))
    bol_corr_uv_int = np.append(bol_corr_uv_int, (np.interp(1500., lam_nu, int_nu) / 1E43))
    bol_corr_optical = np.append(bol_corr_optical, (np.interp(5500., lam_nu, tot_nu) / 1E43))
    bol_corr_xi = np.append(bol_corr_xi, xi_ion(lam_nu, int_nu) / 1E43)
    bol_corr_ion = np.append(bol_corr_ion, lum_ion(lam_nu, int_nu) / 1E43)
    bol_corr_xi_nobolscale = np.append(bol_corr_xi_nobolscale, xi_ion(lam_nu, int_nu))


    luv_integrate = np.append(luv_integrate, np.sum(tot_nu[s]/(lam_nu[s]))/1E43)
    #luv_integrate = np.append(luv_integrate, np.sum(tot_nu[s] * 3E8 / (lam_nu[s] * 1E-10) ** 2) * 1E-10 / 1E43)

    #print(f'Exact: lamda={lam_bol[lam_fuv_idx]}, L_FUV={total_bol[lam_fuv_idx]}')
    #print(f'Mean: lamda=({lam_bol[(np.abs(lam_bol - lims[0])).argmin()]},{lam_bol[(np.abs(lam_bol - lims[1])).argmin()]}], L_FUV={np.mean(total_bol[s])}')

    #print(np.sum(int_nu/(lam_nu)))

ratio_from_t1 = interp1d(AGN_T, y1)
ratio_from_t2 = interp1d(AGN_T, y2)
ratio_from_t3 = interp1d(AGN_T, y3)
ratio_from_t4 = interp1d(AGN_T, y4)
ratio_from_t5 = interp1d(AGN_T, y5)
ratio_from_t6 = interp1d(AGN_T, y6)

outputs = [y1, y2, y3, y4, y5, y6]
interpols = [ratio_from_t1, ratio_from_t2, ratio_from_t3, ratio_from_t4, ratio_from_t5, ratio_from_t6]
labels = ['L_{FUV, exact}\;/\; L_{bol, total}', 'L_{FUV, exact}\;/\; L_{bol, incident}', 'L_{FUV, mean}\;/\; L_{bol, total}', 'L_{FUV, mean}\;/\; L_{bol, incident}', 'L_{FUV, exact}\;/\; 10^{43}', 'L_{FUV, mean}\;/\; 10^{43}']
linestyles = ['solid', 'dashed', 'dotted', 'dashdot', 'solid', 'dashed']
'''
for i, label in enumerate(labels):
    ax.plot(np.log10(AGN_T), outputs[i], c=cmap(norm(i)), ls=linestyles[i], linewidth=8-i, alpha=0.8, label=rf'$\rm{label}$')
    #ax.plot(np.log10(AGN_T), interpols[i](AGN_T), '--', c='r', alpha=0.5)
'''

ax.plot(np.log10(AGN_T), bol_corr_uv, c='k', ls='-', label='UV total')
ax.plot(np.log10(AGN_T), bol_corr_uv_int, c='k', ls='--', label='UV incident')
ax.plot(np.log10(AGN_T), bol_corr_optical, c='k', ls='-', label='Optical')
ax.plot(np.log10(AGN_T), luv_integrate, c='k', ls='dashdot', label='UV integrated')

ax.set_ylabel(r"$\rm F_{\lambda, uv} \; / \; F_{\lambda, bol}$")
ax.set_xlabel(r"$\rm log_{10}[T_{AGN} \; /\; K]$")
#ax.set_xlabel(r"$\rm T_{AGN} \; /\; 10^{5}\;K$")

ax.legend(loc='best', prop={'size': 6})

fig.savefig(f'figures/bolometric_correction_uv_test2.pdf', bbox_inches='tight')


print('L_bol-total, AGN')
print(lbols)
print('L_bol-incident, AGN')
print(linc)
print('L_uv, AGN')
print(luvs)

print(min(lbols), np.median(lbols), max(lbols))
print(min(linc), np.median(linc), max(linc))

print(min(luv_integrate), np.median(luv_integrate), max(luv_integrate))

bol_corr = {'AGN_T': AGN_T, 'ratio':{'FUV': bol_corr_uv, 'optical': bol_corr_optical, 'xi_per_Lbol':bol_corr_xi, 'ionising':bol_corr_ion, 'xi':bol_corr_xi_nobolscale}}
output = {'AGN_T':AGN_T, 'ratios':{'FUV_exact':{'total':y1, 'incident':y2, 'exact_num':y5}, 'FUV_mean':{'total':y3, 'incident':y4, 'exact_num':y6}}}
pickle.dump(bol_corr, open('bolometric_correction_ion.p', 'wb'))