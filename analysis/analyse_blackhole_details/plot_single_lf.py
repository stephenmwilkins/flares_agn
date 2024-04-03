import numpy as np
import cmasher as cmr
import matplotlib.pyplot as plt
import h5py
from flares_utility import analyse

plt.style.use('http://stephenwilkins.co.uk/matplotlibrc.txt')
from unyt import c, Msun, yr, Lsun, g, s

filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'
flares = analyse.analyse(filename, default_tags=False)

binw = 0.25
X_limits = [41., 47.]
Y_limits = [-4.9, -2.01]
bin_edges = np.arange(*X_limits, binw)
bin_centres = bin_edges[:-1]+binw/2


volume = (4./3) * np.pi * (flares.radius)**3 # Mpc^3

# conversion from Mdot -> to Lbol
Lbol_conversion = 0.1 * Msun * c**2 / yr
Lbol_conversion = Lbol_conversion.to('erg/s').value
log10Lbol_conversion = np.log10(Lbol_conversion)

fig = plt.figure(figsize=(3.5, 3.5))

left = 0.15
height = 0.8
bottom = 0.15
width = 0.8

ax = fig.add_axes((left, bottom, width, height))

ax.fill_between([41., 45], [-10, -10], [10,10], color='k', alpha=0.05)
ax.axvline(45., lw=1, c='k', alpha=0.2)

hf = h5py.File('data/accretion_rates.h5', 'r') 


# instantaneous

phi = np.zeros(len(bin_centres))
N = np.zeros(len(bin_centres))

sim = '01'



x = np.array(hf[sim]['Mdot/instant']) * Lbol_conversion

print(f'instantaneous: {np.sum(x)}')

x = x[x>0.0]

N_temp, _ = np.histogram(np.log10(x), bins=bin_edges)

N += N_temp

phi = (N_temp / volume) / binw

ax.plot(bin_centres, np.log10(phi), ls = '-', c='k', lw=2, alpha=1.0)
label = r'$\rm instantaneous$'




for timescale, c in zip([10, 20, 50, 100, 200], cmr.take_cmap_colors(cmap='cmr.sunburst', N=5, cmap_range=(0.2, 0.8))):

    
    phi = np.zeros(len(bin_centres))
    N = np.zeros(len(bin_centres))

    x = np.array(hf[sim][f'Mdot/{timescale}']) * Lbol_conversion

    print(f'{timescale}: {np.sum(x)}')
    x = x[x>0.0]

    N_temp, _ = np.histogram(np.log10(x) , bins=bin_edges)

    N += N_temp

    phi = (N_temp / volume) / binw

    label = rf'$\rm {timescale}\ Myr $'
    ax.plot(bin_centres, np.log10(phi), ls = '-', c=c, lw=2, alpha=0.5)
    


ax.legend(fontsize=7)

ax.set_xlim(X_limits)
ax.set_ylim(Y_limits)

ax.set_ylabel(r'$\rm\log_{10}(\phi/Mpc^{-3}\ dex^{-1})$')
ax.set_xlabel(r'$\rm \log_{10}(L_{bol}/erg\ s^{-1})$')

fig.savefig(f'figs/bolometric_lf_timescale_{sim}.pdf')
fig.clf()