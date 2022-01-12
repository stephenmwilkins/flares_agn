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


bol_correction = pickle.load(open('bolometric_correction_ion.p', 'rb'))

fig = plt.figure(figsize=(3,3))
left  = 0.1
bottom = 0.1
width = 0.75
height = 0.75
ax = fig.add_axes((left, bottom, width, height))


#ax.plot(np.log10(bol_correction['AGN_T']), bol_correction['ratio']['FUV'], c='k', ls='-', label='FUV')
#ax.plot(np.log10(bol_correction['AGN_T']), bol_correction['ratio']['optical'], c='k', ls='--', label='Optical')
#ax.plot(np.log10(bol_correction['AGN_T']), bol_correction['ratio']['ionising'], c='k', ls=':', label='Ionising')
ax.plot(np.log10(bol_correction['AGN_T']), np.log10(bol_correction['ratio']['xi']), c='k', ls='-')
ax.set_ylabel(r"$\rm log_{10}[\xi_{ion} \; / \; (erg^{-1} \; Hz)]$")
ax.set_xlabel(r"$\rm log_{10}[T_{AGN} \; /\; K]$")

ax.legend(loc='best', prop={'size': 6})

fig.savefig(f'figures/xi_Tagn.pdf', bbox_inches='tight')

#print(bol_correction['ratio']['ionising'])