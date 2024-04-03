import numpy as np
import matplotlib.cm as cm
import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
import scipy.stats as stats
from astropy.table import Table
import cmasher as cmr
import h5py
import flare.photom as phot
from flare.photom import M_to_lum
import flares_utility.limits
import flares_utility.plt
import flares_utility.analyse as analyse
import flares_utility.stats

from unyt import c, Msun, yr, Lsun
import utils as u


# set style
plt.style.use('../matplotlibrc.txt')


filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'

flares = analyse.analyse(filename, default_tags=False)

tag = '010_z005p000'

quantities = []

quantities.append({'path': 'Galaxy', 'dataset': f'Mstar_30',
                'name': 'Mstar', 'log10': True})
quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mass',
                'name': 'BH_Mass', 'log10': True})
quantities.append({'path': 'Galaxy', 'dataset': f'BH_Mdot',
                'name': 'BH_Mdot', 'log10': True})





base_size = 3.5
left = 0.15
height = 0.7
bottom = 0.15
width = 0.8

fig = plt.figure(figsize=(base_size, base_size*width/height))

ax = fig.add_axes((left, bottom, width, height))
cax = fig.add_axes([left, bottom+height, width, 0.03])

xlimits = np.array([9.1, 11.9])
ylimits = np.array([-5., -0.5])



for mbh, lw in zip([5,6,7,8], [1,2,3,4]):
    
    y = mbh-xlimits
    ax.plot(xlimits, y, c='k', alpha=0.1, lw=lw)

    m = (y[1]-y[0])/(xlimits[1]-xlimits[0])
    aspect = (ylimits[1]-ylimits[0])/(xlimits[1]-xlimits[0])
    angle = np.arctan(m/aspect)
    x = 9.2
    ax.text(x, mbh - x -0.4, rf'$\rm M_{{\bullet}}/M_{{\odot}}=10^{mbh}$', rotation=180*angle/np.pi, fontsize = 8, c='0.5')



norm = Normalize(vmin=4.0, vmax=11.0)
cmap = cmr.bubblegum


for tag, z in zip(flares.tags, flares.zeds):

    c = cmap(norm(z))

    D = flares.get_datasets(tag, quantities)
    s = D['log10Mstar']>9.

    stellar_mass =  D['Mstar'][s] * Msun
    blackhole_accretion_rate = D['BH_Mdot'][s] * u.accretion_rate_units
    blackhole_mass = D['BH_Mass'][s] * u.blackhole_mass_units 

    eddington_accretion_rate = u.calculate_eddington_accretion_rate(blackhole_mass)
    bolometric_luminosity = u.calcualte_bolometric_luminosity(blackhole_accretion_rate)


    s = (np.log10(stellar_mass.to('Msun'))>9.)&(np.log10(bolometric_luminosity.to('erg/s'))>44.)

    ax.scatter(np.log10(stellar_mass[s].to('Msun')), np.log10(blackhole_mass[s]/stellar_mass[s].to('Msun')), s=1, c=c, alpha=0.2)






# add obervations



# Kormendy and Ho (read off Hazboulit)


# x = np.array([9., 12.])
# y = np.array([6.3, 9.8])-x

# ax.plot(x,y,c='k',lw=1,ls='--',label=r'$\rm Kormendy\ &\ Ho\ (2013)\ [z=0]$')


# Maiolino (2023)
obs = Table.read('obs_data/Maiolino23a.ecsv', format='ascii.ecsv')

# ax.scatter(obs['log10Mstar'], obs['log10Mbh']-obs['log10Mstar'], c='k', marker='o', s=5)
s = (obs['log10Mstar'] > 9.1)
z = obs['z'][s]
x = obs['log10Mstar'][s]
y = obs['log10Mbh'][s]-obs['log10Mstar'][s]
xerr = np.array([-obs['log10Mstar_err_low'][s], obs['log10Mbh_err_upp'][s]]).T
yerr = np.array([np.sqrt(obs['log10Mstar_err_low'][s]**2 + obs['log10Mbh_err_low'][s]**2), np.sqrt(obs['log10Mstar_err_upp'][s]**2 + obs['log10Mbh_err_upp'][s]**2)]).T

for x_, y_, xerr_, yerr_, z_, in zip(x, y, xerr, yerr, z):
    xerr_ = np.array([[xerr_[0]],[xerr_[1]]])
    yerr_ = np.array([[yerr_[0]],[yerr_[1]]])
    ax.errorbar(x_, y_, xerr=xerr_, yerr=yerr_, fmt='o', markersize=5, c=cmap(norm(z_)))

# For label
ax.errorbar(0.0, 0.0, xerr=0.0, yerr=0.0, fmt='o', markersize=5, label=r'$\rm Maiolino+23a$', c='0.5')


# Harikane (2023)
obs = Table.read('obs_data/Harikane23.ecsv', format='ascii.ecsv')

# ax.scatter(obs['log10Mstar'], obs['log10Mbh']-obs['log10Mstar'], c='k', marker='o', s=5)
s = (obs['log10Mstar'] > 9.1)
z = obs['z'][s]
x = obs['log10Mstar'][s]
y = np.log10(obs['Mbh'][s])-obs['log10Mstar'][s]


log10Mbh_err_upp = np.log10(obs['Mbh'][s]+obs['Mbh_err_upp'][s]) - np.log10(obs['Mbh'][s])
log10Mbh_err_low = np.log10(obs['Mbh'][s]) - np.log10(obs['Mbh'][s]+obs['Mbh_err_low'][s])

xerr = np.array([-obs['log10Mstar_err_low'][s], log10Mbh_err_upp]).T
yerr = np.array([np.sqrt(obs['log10Mstar_err_low'][s]**2 + log10Mbh_err_low**2), np.sqrt(obs['log10Mstar_err_upp'][s]**2 + log10Mbh_err_upp**2)]).T

for x_, y_, xerr_, yerr_, z_, in zip(x, y, xerr, yerr, z):
    xerr_ = np.array([[xerr_[0]],[xerr_[1]]])
    yerr_ = np.array([[yerr_[0]],[yerr_[1]]])
    if xerr_[0] == 0.0:
        xuplims = True
        xerr_=0.15
        uplims = True
        yerr_=0.15
    else:
        xuplims = False
    ax.errorbar(x_, y_, xerr=xerr_, yerr=yerr_, uplims=uplims, xuplims=xuplims, fmt='o', markersize=5, c=cmap(norm(z_)))
    # ax.errorbar(x_, y_,  fmt='s', markersize=5, c=cmap(norm(z_)))

# For label
ax.errorbar(0.0, 0.0, xerr=0.0, yerr=0.0, fmt='s', markersize=5, label=r'$\rm Harikane+23$', c='0.5')


# Kocevski (2023)
obs = Table.read('obs_data/Kocevski23.ecsv', format='ascii.ecsv')

# ax.scatter(obs['log10Mstar'], obs['log10Mbh']-obs['log10Mstar'], c='k', marker='o', s=5)
s = (obs['log10Mstar'] > 9.1)
z = obs['z'][s]
x = obs['log10Mstar'][s]

y = np.log10(obs['Mbh'][s])-obs['log10Mstar'][s]


log10Mbh_err_upp = np.log10(obs['Mbh'][s]+obs['Mbh_err_upp'][s]) - np.log10(obs['Mbh'][s])
log10Mbh_err_low = np.log10(obs['Mbh'][s]) - np.log10(obs['Mbh'][s]+obs['Mbh_err_low'][s])

xerr = np.array([-obs['log10Mstar_err_low'][s], log10Mbh_err_upp]).T
yerr = np.array([np.sqrt(obs['log10Mstar_err_low'][s]**2 + log10Mbh_err_low**2), np.sqrt(obs['log10Mstar_err_upp'][s]**2 + log10Mbh_err_upp**2)]).T

for x_, y_, xerr_, yerr_, z_, in zip(x, y, xerr, yerr, z):
    xerr_ = np.array([[xerr_[0]],[xerr_[1]]])
    yerr_ = np.array([[yerr_[0]],[yerr_[1]]])
    if xerr_[0] == 0.0:
        xuplims = True
        xerr_=0.15
        uplims = True
        yerr_=0.15
    else:
        xuplims = False

    ax.errorbar(x_, y_, xerr=xerr_, yerr=yerr_, uplims=uplims, xuplims=xuplims, fmt='p', markersize=5, c=cmap(norm(z_)))
    # ax.errorbar(x_, y_,  fmt='s', markersize=5, c=cmap(norm(z_)))

# For label
ax.errorbar(0.0, 0.0, xerr=0.0, yerr=0.0, fmt='p', markersize=5, label=r'$\rm Kocevski+23$', c='0.5')



ax.legend(fontsize=8)
ax.set_ylim(ylimits)
ax.set_xlim(xlimits)

ax.set_xlabel(r'$\rm log_{10}(M_{\star}/M_{\odot})$')
ax.set_ylabel(r'$\rm log_{10}(M_{\bullet}/M_{\star})$')

# add colourbar
cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(cmapper, cax=cax, orientation='horizontal')
cax.xaxis.tick_top()
cax.xaxis.set_label_position('top')
cax.set_xlabel(r'$\rm z$', fontsize=8)
cax.tick_params(axis='x', labelsize=6)


filename = f'figs/Mstar_Mbh_Mbhdot_5_obs.pdf'
fig.savefig(filename)
print(filename)

