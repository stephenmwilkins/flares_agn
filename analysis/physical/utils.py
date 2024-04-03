import numpy as np
import flares_utility.analyse as analyse
from unyt import (
    c,
    Msun,
    yr,
    # Lsun,
    g,
    s,
    G,
    mp,
    sigma_thompson)





# EAGLE specific quantities
# little-h
h = 0.677

# radiative efficiency
radiative_efficiency = 0.1

# accretion rate units (already corrected for h)
blackhole_mass_units = Msun

# accretion rate units (not corrected for h previously)
# accretion_rate_units = 6.446E23 * (g/s) / h
accretion_rate_units = Msun / yr / h
seed_mass = 1E5 * Msun / h

accretion_rate_units = accretion_rate_units.to('Msun/yr')
blackhole_mass_units = blackhole_mass_units.to('Msun')

def calculate_bolometric_luminosity(
        accretion_rate,
        radiative_efficiency=radiative_efficiency):

    """
    Calculate bolometric luminosity from accretion rate
    
    """

    return radiative_efficiency * accretion_rate * c**2


def calculate_eddington_accretion_rate(
        blackhole_mass,
        radiative_efficiency=radiative_efficiency):

    return ((4 * np.pi * G * blackhole_mass * mp)
            / (radiative_efficiency * sigma_thompson * c))


def load_data(
        filename='/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5',
        tag='010_z005p000',
        mass_limit=6.5):

    
    flares = analyse.analyse(filename, default_tags=False)

    quantities = []
    
    quantities.append({'path': 'Galaxy', 'dataset': f'Mstar_30',
                  'name': 'Mstar', 'log10': True})
    quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mass',
                    'name': 'BH_Mass', 'log10': True})
    quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mdot',
                    'name': 'BH_Mdot', 'log10': True})


    D = flares.get_datasets(tag, quantities)
    s = D['log10BH_Mass'] > mass_limit
 
    blackhole_accretion_rate = D['BH_Mdot'][s] * accretion_rate_units
    blackhole_mass = D['BH_Mass'][s] * blackhole_mass_units 

    eddington_accretion_rate = calculate_eddington_accretion_rate(blackhole_mass)
    bolometric_luminosity = calculate_bolometric_luminosity(blackhole_accretion_rate)

    eddington_ratio = blackhole_accretion_rate/eddington_accretion_rate

    return blackhole_mass, blackhole_accretion_rate, bolometric_luminosity, eddington_ratio