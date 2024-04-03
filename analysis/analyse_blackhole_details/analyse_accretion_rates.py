import numpy as np
import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import h5py
from scipy.stats import binned_statistic
from astropy.cosmology import Planck13 as cosmo, z_at_value
import astropy.units as u
plt.style.use('http://stephenwilkins.co.uk/matplotlibrc.txt')
from unyt import c, Msun, yr, Lsun, g, s



sims = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
           '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35',
           '36', '37', '38', '39']

# conversion from Mbh to LEdd
log10LEdd_conversion = np.log10(3.2) + 4 + np.log10((1*Lsun).to('erg/s').value)

# conversion from Mdot -> to Lbol
Lbol_conversion = 0.1 * Msun * c**2 / yr
log10Lbol_conversion = np.log10(Lbol_conversion.to('erg/s').value)


timescales = [10, 20, 50, 100, 200]



with h5py.File('data/accretion_rates.h5', 'r') as hf:

    # loop over all simulations

    masses = np.array([])
    accretion_rates = {}
    accretion_rates['instant'] = np.array([])
    for timescale in timescales:
        accretion_rates[timescale] = np.array([])

    for sim in hf.keys():

        print('-'*20, sim)
        


        masses = np.concatenate((masses, hf[f'{sim}/Mbh'][()]))

        accretion_rates_ = hf[f'{sim}/Mdot/instant'][()]
        accretion_rates['instant'] = np.concatenate((accretion_rates['instant'], accretion_rates_))

        print(len(accretion_rates_))

        print(f'instant {np.sum(accretion_rates_):.3f}')

        for timescale in timescales:

            accretion_rates_ = hf[f'{sim}/Mdot/{timescale}'][()]

            accretion_rates[timescale] = np.concatenate((accretion_rates[timescale], accretion_rates_))

            print(f'{timescale} {np.sum(accretion_rates_):.3f}')




    log10LEdd = np.log10(masses) + log10LEdd_conversion
    log10Lbol = np.log10(accretion_rates['instant']) + log10Lbol_conversion
    eddington_ratio = log10Lbol - log10LEdd


    print('-'*40, 'ALL')

    print(f'{np.min(eddington_ratio):.2f}, {np.max(eddington_ratio):.2f}')

    print(f'instant {np.sum(accretion_rates["instant"]):.3f}')

    for timescale in timescales:

        print(f'{timescale} {np.sum(accretion_rates[timescale]):.3f}')

        