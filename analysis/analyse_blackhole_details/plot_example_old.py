
"""
Make plots for a single BH
"""

import pandas as pd
import numpy as np
import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import h5py
from astropy.cosmology import Planck13 as cosmo, z_at_value
import astropy.units as u
from unyt import g, s
from unyt import c, Msun, yr, Lsun, g, s

# conversion from EAGLE units to useful units
conversion = (6.446E23 * g/s).to('Msun/yr').value

# conversion from Mbh to LEdd
log10LEdd_conversion = np.log10(3.2) + 4 + np.log10((1*Lsun).to('erg/s').value)

# conversion from Mdot -> to Lbol
Lbol_conversion = 0.1 * Msun * c**2 / yr
log10Lbol_conversion = np.log10(Lbol_conversion.to('erg/s').value)


plt.style.use('http://stephenwilkins.co.uk/matplotlibrc.txt')

h = 0.677
target_z = 5.


sim = '00'
# pid = 29073816540486753
pid = 29077115957483601

redshift_limits = [8., 5.]


data_dir = f'/Users/sw376/Dropbox/Research/data/simulations/flares/'

details = h5py.File(f'{data_dir}/blackhole_details.h5', 'r')
mergers = h5py.File(f'{data_dir}/bhmergers.h5', 'r')

bh = details[sim][str(pid)]

# Mass growth


primary_ids = mergers[sim]['ids_primary'][()]
secondary_ids = mergers[sim]['ids_secondary'][()]

for primary_id in set(primary_ids):
    print(primary_id, np.sum(primary_ids == primary_id))

s = primary_ids == pid
print(secondary_ids[s])
print(mergers[sim]['a'][s])
print(mergers[sim]['masses_secondary'][s])

print(np.sum(mergers[sim]['masses_secondary'][s])/bh['BH_Subgrid_Mass'][-1])


colors = cmr.take_cmap_colors(cmap=cmr.neon, N=5, cmap_range=(0.2, 0.9))


fig = plt.figure(figsize=(3.5, 6.0))

left = 0.15
width = 0.8
bottom = 0.075
height = 0.3

ax = fig.add_axes((left, bottom+height*2, width, height))
ax2 = fig.add_axes((left, bottom+height, width, height))
ax3 = fig.add_axes((left, bottom, width, height))  # Eddington ratio

ax3.axhline(0.0, c='k', alpha=0.1, lw=2)


log10subgrid_mass = np.log10(bh['BH_Subgrid_Mass'][()]/h) + 10.
log10particle_mass = np.log10(bh['BH_Particle_Mass'][()]/h) + 10.
log10accretion_rate = np.log10(bh['Mdot'][()] * conversion) 

eddington_luminosity = log10subgrid_mass + log10LEdd_conversion
bolometric_luminosity = log10accretion_rate + log10Lbol_conversion
eddington_ratio = bolometric_luminosity - eddington_luminosity


for a in mergers[sim]['a'][s]:
    z = (1/a)-1
    ax.axvline(z, c='k', lw=1, alpha=0.1)
    ax2.axvline(z, c='k', lw=1, alpha=0.1)
    ax3.axvline(z, c='k', lw=1, alpha=0.1)

ax.plot(bh['z'][()], log10particle_mass, c=colors[0], lw=3, alpha=0.2)
ax.plot(bh['z'][()], log10subgrid_mass, c=colors[1], lw=2, alpha=1.0)

ax2.plot(bh['z'][()], log10accretion_rate, c=colors[2], alpha=1.0, lw=1)
ax3.plot(bh['z'][()], eddington_ratio, c=colors[3], alpha=1.0, lw=1)

ax.set_xlim(redshift_limits)
ax2.set_xlim(redshift_limits)
ax3.set_xlim(redshift_limits)
ax.set_xticks([])
ax2.set_xticks([])

ax.set_ylim([5., 9.5])


ax3.set_xlabel(r'$\rm z$')
ax.set_ylabel(r'$\rm \log_{10}(M_{\bullet}/M_{\odot})$')
ax2.set_ylabel(r'$\rm \log_{10}(\dot{M_{\bullet}}/M_{\odot}\ yr^{-1})$')
ax3.set_ylabel(r'$\rm \log_{10}(\lambda)$')

fig.savefig(f'figs/example_blackhole_growth.pdf')
fig.clf()










# ## ---- find merger redshift, dM condition
# dM = 1e-5
# merger_i = np.where(np.diff(bh['BH_Subgrid_Mass'][()]) > dM)[0]

# plt.hist(np.log10(np.diff(bh['BH_Subgrid_Mass'][()])), bins = np.arange(-10, -3, 0.1))
# plt.show()

# plt.plot(bh['a'][()], np.diff(bh['BH_Subgrid_Mass'][()]))

# plt.show()

# print(np.min(bh['BH_Subgrid_Mass'][()]), np.max(bh['BH_Subgrid_Mass'][()]))

# print(merger_i)
# print(len(merger_i))
# # print("merger redshifts:", merger_z)


# # bh_history = pd.read_hdf(bh)

# # _temp = bh_history   
# # max_mass = _temp.iloc[0]['BH_Subgrid_Mass']
# # indices = np.zeros(len(_temp), dtype=bool)
# # for i, _bhm in enumerate(_temp['BH_Subgrid_Mass']):
# #     if _temp.iloc[i]['BH_Subgrid_Mass'] >= max_mass:
# #         max_mass = _temp.iloc[i]['BH_Subgrid_Mass']
# #         indices[i] = True

# # bh2 = _temp.loc[indices]


