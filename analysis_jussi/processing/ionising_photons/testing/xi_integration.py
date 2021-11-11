from scipy.integrate import simps
import numpy as np

def xi_ion(lam, sed, lims=[0, 912]):
    """
    :param lam: wavelength array in Å
    :param sed: SED as F_nu/(erg/s/Hz)
    :param lims: limits of integration in Å
    :return: ionising photon production efficiency
    """
    s = ((lam >= 0) & (lam < 912)).nonzero()[0]
    conv = 1.98644586e-08/lam[s]
    #conv = ((constants.h * constants.c / ((lam[s] * units.AA).to(units.m))).to(units.erg)).value  # alternative using astropy.units and astropy.constants

    xi = simps(sed[s]/conv, lam[s])/np.interp(1500, lam, sed)

    return xi
