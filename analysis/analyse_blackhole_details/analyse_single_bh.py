import numpy as np
import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import h5py
from astropy.cosmology import Planck13 as cosmo, z_at_value
import astropy.units as u
plt.style.use('http://stephenwilkins.co.uk/matplotlibrc.txt')


h = 0.677
target_z = 5.
averaging_timescale = 100. * u.Myr 
averaging_z = z_at_value(cosmo.age, cosmo.age(5.0) - averaging_timescale)
print(averaging_z)

sim = '00'
filename = f'/Users/sw376/Dropbox/Research/data/simulations/flares/blackhole_details_h{sim}.h5'

pid = '29073816540486753'

with h5py.File(filename, 'r') as hf:

    bh = hf[pid]

    z = (1 / bh['a'][()]) - 1

    index_at_target = np.argmin(np.fabs(z-target_z))
    mass = 1E10 * bh['BH_Subgrid_Mass'][()]/h

    for z_, mass_ in zip(z, mass):
        print(z_, np.log10(mass_))