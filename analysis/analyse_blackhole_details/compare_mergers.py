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

make_combined_plot = True
make_individual_plot = True

sims = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
           '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35',
           '36', '37', '38', '39']

# conversion from Mbh to LEdd
log10LEdd_conversion = np.log10(3.2) + 4 + np.log10((1*Lsun).to('erg/s').value)

# conversion from Mdot -> to Lbol
Lbol_conversion = 0.1 * Msun * c**2 / yr
log10Lbol_conversion = np.log10(Lbol_conversion.to('erg/s').value)


timescales = [10, 20, 50, 100, 200]





if make_combined_plot:

    fig2 = plt.figure(figsize=(3.5, 3.5))

    left = 0.15
    height = 0.8
    bottom = 0.15
    width = 0.8

    ax2 = fig2.add_axes((left, bottom, width, height))
    ax2.axhline(0.0, c='k', lw=2, alpha=0.2)



hf = h5py.File('data/accretion_rates.h5', 'r')
hf_no_mergers = h5py.File('data/accretion_rates_no_mergers.h5', 'r')

timescale = 100


masses = np.array([])
accretion_rates = np.array([])
accretion_rates_no_mergers = np.array([])

for sim in hf.keys():
    masses = np.concatenate((masses, hf[f'{sim}/Mbh'][()]))
    accretion_rates = np.concatenate((accretion_rates, hf[f'{sim}/Mdot/{timescale}'][()]))
    
for sim in hf_no_mergers.keys():
    accretion_rates_no_mergers = np.concatenate((accretion_rates_no_mergers, hf_no_mergers[f'{sim}/Mdot/{timescale}'][()]))
       
ratio = np.log10(accretion_rates/accretion_rates_no_mergers)

    

fig = plt.figure(figsize=(3.5, 3.5))

left = 0.15
height = 0.8
bottom = 0.15
width = 0.8


ax = fig.add_axes((left, bottom, width, height))


ax.axhline(0.0, c='k', lw=2, alpha=0.2)


ax.scatter(np.log10(masses), ratio, s=3)

ax.set_xlim([7., 9.5])

# ax.set_ylabel(rf'$\rm \log_{{10}}(\dot{{M}}_{{\bullet, instant}}/\dot{{M}}_{{\bullet, {timescale}\ Myr}})$')
ax.set_xlabel(r'$\rm \log_{10}(M_{\bullet}/M_{\odot})$')

fig.savefig(f'figs/merger_impact.pdf')
fig.clf()


