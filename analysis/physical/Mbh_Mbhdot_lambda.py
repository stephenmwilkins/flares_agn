import numpy as np
import matplotlib.cm as cm
import cmasher as cmr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import flares_utility.limits
import flares_utility.plt
import flares_utility.analyse as analyse
from unyt import Msun, yr
import utils as u


# set style
plt.style.use('../matplotlibrc.txt')


# get data
blackhole_mass, blackhole_accretion_rate, bolometric_luminosity, eddington_ratio = u.load_data()


resolved_blackhole_mass = 1E7 * Msun

resolved_eddington_accretion_rate = u.calculate_eddington_accretion_rate(resolved_blackhole_mass)

resolved_bolometric_luminosity = u.calculate_bolometric_luminosity(resolved_eddington_accretion_rate)


print(resolved_eddington_accretion_rate)
print(resolved_bolometric_luminosity)
print(np.log10(resolved_bolometric_luminosity.to('erg/s')))


xlimits = log10blackhole_mass_limit = np.array([6.51, 9.49]) 
ylimits = log10blackhole_accretion_rate_limit = np.array([-5.99, 0.99]) 
bolometric_luminosity_limits = u.calculate_bolometric_luminosity(10**log10blackhole_accretion_rate_limit*Msun/yr)
y2limits = np.log10(bolometric_luminosity_limits.to('erg/s'))





fig = plt.figure(figsize = (3.5, 3.5))

left  = 0.15
height = 0.7
bottom = 0.15
width = 0.7

ax = fig.add_axes((left, bottom, width, height))
ax2 = ax.twinx()
cax = fig.add_axes([left, bottom+height, width, 0.03])

norm = Normalize(vmin=-6., vmax=0.18)
cmap = cmr.gem

n = norm


ax.fill_between([0,7],[-10,-10],[7,7], color='k',alpha=0.05)
# ax2.axhline(log10Lbol_+7., c='k', ls=':',lw=1, alpha=0.5)

accretion_rate_unity = np.log10(u.calculate_eddington_accretion_rate(1*Msun).to('Msun/yr'))

print(accretion_rate_unity)

for ratio, lw in zip([0.001, 0.01, 0.1, 1.0], [1,2,3,4]):

    # calculate luminosity

    X = log10blackhole_mass_limit
    Y = log10blackhole_mass_limit + accretion_rate_unity + np.log10(ratio)
    print(X, Y)

    ax.plot(X, Y, lw=lw, c='k', ls='-', alpha=0.1, zorder=0)

    m = (Y[1]-Y[0])/(X[1]-X[0])
    aspect = (ylimits[1]-ylimits[0])/(xlimits[1]-xlimits[0])
    angle = np.arctan(m/aspect)

    x = 7.5
    ax.text(x, Y[0] + 1.2, rf'$\rm \lambda={ratio}$', rotation=180*angle/np.pi, fontsize = 8, c='0.5')


ax.scatter(np.log10(blackhole_mass), 
           np.log10(blackhole_accretion_rate), 
           s=1, 
           zorder=1, 
           c=cmap(norm(np.log10(eddington_ratio))))


ax.set_xlim(log10blackhole_mass_limit)
ax.set_ylim(log10blackhole_accretion_rate_limit)
ax2.set_ylim(y2limits)
ax2.grid(False)

ax.set_xlabel(r'$\rm log_{10}(M_{\bullet}/M_{\odot})$')
ax.set_ylabel(r'$\rm log_{10}(\dot{M}_{accr}/M_{\odot}\ yr^{-1})$')
ax2.set_ylabel(r'$\rm log_{10}(L_{bol}/erg\ s^{-1})$')


# add colourbar
cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(cmapper, cax=cax, orientation='horizontal')
cax.xaxis.tick_top()
cax.xaxis.set_label_position('top')
cax.set_xlabel(r'$\rm log_{10}(\lambda_{\rm Edd})$', fontsize=7)
cax.tick_params(axis='x', labelsize=6)


filename = f'figs/Mbh_Mbhdot_lambda.pdf'
print(filename)

fig.savefig(filename)

