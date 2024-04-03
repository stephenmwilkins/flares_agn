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
remove_mergers = False

# open flares for list of sims and weights
filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'
flares = analyse.analyse(filename, default_tags=False)


h = 0.677
target_z = 5.

averaging_timescales = [10, 20, 50, 100, 200]
averaging_redshifts = {t: z_at_value(cosmo.age, cosmo.age(5.0) - t * u.Myr) for t in averaging_timescales}

data_dir = f'/Users/sw376/Dropbox/Research/data/simulations/flares/'


# this appears to be correct because it yields the correct range of Eddington ratios
instant_conversion = (6.446E23 * g/s).to('Msun/yr').value # -> Msun/yr
mass_conversion = 1E10/h # -> Msun


mergers = h5py.File(f'{data_dir}/bhmergers.h5', 'r')

# output
if remove_mergers:
    output_file = 'data/accretion_rates.h5'
else:
    output_file = 'data/accretion_rates_no_mergers.h5'

with h5py.File(output_file, 'w') as hfout:

    with h5py.File(f'{data_dir}/blackhole_details.h5', 'r') as details:

        # for sim, weight in zip(['00'], flares.weights):
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

                

                # Only consider blackholes with entries near the target redshift
                if np.fabs(z[index_at_target]-target_z)<0.01:

                    target_mass = mass_conversion * bh['BH_Subgrid_Mass'][index_at_target]

                    if np.log10(target_mass)>7.0:

                        # get time nefore target
                        time = (cosmo.age(target_z) - cosmo.age(z)).to('Myr').value
                        # for z_, t, m in zip(z, time, bh['BH_Subgrid_Mass']):
                        #     print(f'{z_:.3f} {t:.2f}, {np.log10(m)+10.:.2f}')

                        masses.append(target_mass)
                        weights.append(weight)
                        instantaneous_accretion_rate = bh['Mdot'][index_at_target] * instant_conversion
                        instantaneous_accretion_rates.append(instantaneous_accretion_rate)

                        if verbose:
                            print('-'*20)
                            print(pid)
                            print(f'{np.log10(target_mass):.2f}')
                            print(instantaneous_accretion_rate)

                        for averaging_timescale in averaging_timescales:

                            # the mass at this redshift
                            timescale_mass = mass_conversion * np.interp(averaging_timescale, time[::-1], bh['BH_Subgrid_Mass'][()][::-1])
                            target_mass_no_mergers = False

                            # find out the contribution of mergers and exclude

                            if remove_mergers:

                                if int(pid) in mergers[sim]['ids_primary'][()]:

                                    # determine the redshift at averaging timescale
                                    z_ = np.interp(averaging_timescale, time[::-1], bh['z'][()][::-1])

                                    merged = (mergers[sim]['ids_primary'][()] == int(pid)) & ((1/mergers[sim]['a'][()])-1 < z_) & ((1/mergers[sim]['a'][()])-1 > target_z)
                                    
                                    if np.sum(merged) > 0:

                                        # calcualte the mass that has merged in over the timescale
                                        merged_mass = mass_conversion * np.sum(mergers[sim]['masses_secondary'][merged])

                                        target_mass_no_mergers = target_mass
                                        target_mass -= merged_mass
                                        

                                        if verbose:
                                            print('     --- MERGERS')
                                            print(f'    {z_:.2f}')
                                            zz =(1/mergers[sim]["a"][merged])-1
                                            print(f'    {zz}')
                                            print(f'    {np.sum(merged)}')
                                            print(f'    {np.log10(np.sum(merged_mass)):.2f}')
                                            print(f'    {target_mass/target_mass_no_mergers:.2f}')



                            averaged_accretion_rate = (target_mass - timescale_mass)/(averaging_timescale*1E6)
                            if verbose:
                                print(f'{averaging_timescale:.2f}, {np.log10(timescale_mass):.2f}, {timescale_mass/target_mass:.2f}, {averaged_accretion_rate:.2f}, {instantaneous_accretion_rate/averaged_accretion_rate:.2f}')

                            averaged_accretion_rates[averaging_timescale].append(averaged_accretion_rate)
                        

            hfout[f'{sim}/Mbh'] = np.array(masses)
            hfout[f'{sim}/Mdot/instant'] = np.array(instantaneous_accretion_rates)
            for averaging_timescale in averaging_timescales:
                hfout[f'{sim}/Mdot/{averaging_timescale}'] = np.array(averaged_accretion_rates[averaging_timescale])









