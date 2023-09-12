import flares

import numpy as np

import matplotlib.pyplot as plt


flares_dir = '/data/simulations' # change to point to the directory containing the simulations file (relative to the script location)

fl = flares.flares(f'{flares_dir}/flares_no_particlesed.hdf5', sim_type='FLARES') # this loads the hdf5 file

# this prints out the available Galaxy data in the file
fl.print_keys('00/005_z010p000/Galaxy')

# this prints out the different simulated regions
fl.print_keys()

# this prints out the available redshifts
fl.zeds

# this prints out the available redshift snapshots
fl.tags

# you can load up datasets (from all regions and all snapshots) using this
Mstar = fl.load_dataset('Mstar_30', arr_type='Galaxy')

# or you can use the following to load a single dataset from a single snapshot or region
# This specifically loads up stellar masses of region '00' from snapshot '005_z010p000' (z=10) from the container 'Galaxy', which contains the available galaxy data
x = fl._load_single_dataset(name='Mstar_30', halo='00', tag='005_z010p000', arr_type='Galaxy')

# This specifically loads up instantaneous star formation rates of region '00' from snapshot '005_z010p000' (z=10) from the container 'Galaxy', which contains the available galaxy data
y = fl._load_single_dataset(name='SFR_inst_30', halo='00', tag='005_z010p000', arr_type='Galaxy')

# example visualisation

x = np.log10(x) + 10 # the masses in these files are given in the form: M / (10^10 M_sol) so this converts it into log10(M_star)
y = np.log10(y)

plt.scatter(x, y)
plt.xlabel(r'$log_{10}(M_{*} / M_{\odot})$')
plt.ylabel(r'$log_{10}(SFR_{inst} / M_{\odot} / yr^{-1})$')
plt.show()