"""
Measure averaged BH accretion rates from the accretion rate history instead of the mass growth.
"""


import numpy as np
import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import h5py
from astropy.cosmology import Planck13 as cosmo, z_at_value
import astropy.units as u
from flares_utility import analyse
from unyt import g, s


plt.style.use('http://stephenwilkins.co.uk/matplotlibrc.txt')

verbose = False

# open flares for list of sims and weights
filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'
flares = analyse.analyse(filename, default_tags=False)



h = 0.677
target_z = 5.

averaging_timescales = [10, 20, 50, 100, 200]
averaging_redshifts = {t: z_at_value(cosmo.age, cosmo.age(5.0) - t * u.Myr) for t in averaging_timescales}

data_dir = f'/Users/sw376/Dropbox/Research/data/simulations/flares/'
mergers = h5py.File(f'{data_dir}/bhmergers.h5', 'r')

# this appears to be correct because it yields the correct range of Eddington ratios
instant_conversion = (6.446E23 * g/s).to('Msun/yr').value # -> Msun/yr
mass_conversion = 1E10/h # -> Msun


# output

output_file = 'data/accretion_rates_instant.h5'


with h5py.File(output_file, 'w') as hfout:

    with h5py.File(f'{data_dir}/blackhole_details.h5', 'r') as details:

        # for sim, weight in zip(['37'], flares.weights):
        for sim, weight in zip(flares.sims, flares.weights):
            
            masses = []
            instantaneous_accretion_rates = []
            averaged_accretion_rates = {t:[] for t in averaging_timescales}
           
            weights = []

            pids = np.array(list(details[sim].keys()))
            print(sim, len(pids))

            # loop over galaxies
            for pid in pids:
                
                bh = details[sim][pid]

                z = (1 / bh['a'][()]) - 1

                index_at_target = np.argmin(np.fabs(z-target_z))

                # if the blackhole is active at the target_z use those quantities

                if np.fabs(z[index_at_target]-target_z)<0.01:
                    final_mass = bh['BH_Subgrid_Mass'][index_at_target]
                    instantaneous_accretion_rate = bh['Mdot'][index_at_target] * instant_conversion
                # otherwise check if it's been swallowed and otherwise use the nearest mass and set the accretion rate to zero
                elif int(pid) not in mergers[sim]['ids_secondary'][()]:
                    final_mass = bh['BH_Subgrid_Mass'][index_at_target]
                    instantaneous_accretion_rate = 0.0
                # if swallowed ignore
                else:
                    final_mass = 0.0
                    instantaneous_accretion_rate = 0.0

                final_mass *= mass_conversion

                if np.log10(final_mass)>7.0:
                    if instantaneous_accretion_rate == 0.0:
                        print('no activity')
                    

                    # get time coordinate
                    time = cosmo.age(z).to('Myr').value

                    # the width of each bin, use for the relative weight
                    dt = time[1:]-time[:-1]

                    # get time nefore target
                    time_before_present = cosmo.age(target_z).to('Myr').value - time

                    mdot = bh['Mdot'][1:]

                    masses.append(final_mass)
                    weights.append(weight)
                    
                    instantaneous_accretion_rates.append(instantaneous_accretion_rate)

                    if verbose:
                        print('-'*20)
                        print(pid)
                        print(f'{np.log10(final_mass):.2f}')
                        print(instantaneous_accretion_rate)

                    for averaging_timescale in averaging_timescales:

                        # select bits of the accretion rate history within the averaging timescale

                        s = time_before_present < averaging_timescale

                        s = s[1:]

                        averaged_accretion_rate = np.sum(mdot[s]*dt[s])/np.sum(dt[s])

                        averaged_accretion_rate *= instant_conversion
      
                        if verbose:
                            print(f'{averaging_timescale:.2f} {np.sum(mdot[s]*dt[s])} {np.sum(dt[s])} {averaged_accretion_rate:.2f}, {instantaneous_accretion_rate/averaged_accretion_rate:.2f}')

                        averaged_accretion_rates[averaging_timescale].append(averaged_accretion_rate)
                    

            hfout[f'{sim}/Mbh'] = np.array(masses)
            hfout[f'{sim}/Mdot/instant'] = np.array(instantaneous_accretion_rates)
            for averaging_timescale in averaging_timescales:
                hfout[f'{sim}/Mdot/{averaging_timescale}'] = np.array(averaged_accretion_rates[averaging_timescale])









