import numpy as np
import matplotlib.cm as cm
import cmasher as cmr
from matplotlib.colors import Normalize
from scipy.stats import binned_statistic

from flares_utility.stats import weighted_median, binned_weighted_quantile
import utils as u
import matplotlib.pyplot as plt

# set style
plt.style.use('../matplotlibrc.txt')

# get data
blackhole_mass, blackhole_accretion_rate, bolometric_luminosity, eddington_ratio = u.load_data()


xlimits = Mbh_limit = [6.51, 9.49] # Msun
ylimits = np.array([-6., 0.4])

fig = plt.figure(figsize = (3.5, 3.5))

left  = 0.15
height = 0.7
bottom = 0.15
width = 0.6
hwidth = 0.2

ax = fig.add_axes((left, bottom, width, height))
cax = fig.add_axes([left, bottom+height, width, 0.03])
hax = fig.add_axes((left+width, bottom, hwidth, height))


norm = Normalize(vmin=-6., vmax=1.)
cmap = cmr.sunburst
n = norm

ax.fill_between([0,7],[-10,-10],[7,7], color='k',alpha=0.05)

ax.scatter(np.log10(blackhole_mass.value),
           np.log10(eddington_ratio),
           s=1,
           zorder=1,
           c=cmap(norm(np.log10(blackhole_accretion_rate))))


# plot median relation
mass_bins = np.arange(6.5, 9.5, 0.25)
median, bin_edges, N = binned_statistic(
    np.log10(blackhole_mass), 
    np.log10(eddington_ratio), 
    bins=mass_bins, 
    statistic='median')

ax.plot(mass_bins[:-1], median, c='k', ls='--')

# accretion weighted
median = binned_weighted_quantile(np.log10(blackhole_mass), 
                                  np.log10(eddington_ratio), 
                                  blackhole_accretion_rate.value, 
                                  mass_bins, 
                                  [0.5])


ax.plot(mass_bins[:-1], median, c='k', ls='-')

s2 = np.log10(blackhole_mass.to('Msun').value) > 7

# add histogram of Eddington ratios
bins = np.arange(-6.1, 0.6, 0.2)
hax.hist(np.log10(eddington_ratio[s2]), bins=bins, orientation="horizontal", color='0.8', density=True, histtype='stepfilled')

hax.axhline(np.percentile(np.log10(eddington_ratio[s2]), 15.8), color='k', alpha=0.5, ls=':', lw=1)
hax.axhline(np.percentile(np.log10(eddington_ratio[s2]), 84.2), color='k', alpha=0.5, ls='--', lw=1)
hax.axhline(np.median(np.log10(eddington_ratio[s2])), color='k', alpha=0.5, lw=1)


hax.set_ylim(ylimits)
hax.set_yticks([])
hax.set_xticks([])



ax.set_xlim(Mbh_limit)
ax.set_ylim(ylimits)
# ax.set_yscale('log')

ax.set_xlabel(r'$\rm log_{10}(M_{\bullet}/M_{\odot})$')
ax.set_ylabel(r'$\rm log_{10}(\lambda_{\rm Edd})$')


# add colourbar
cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(cmapper, cax=cax, orientation='horizontal')
cax.xaxis.tick_top()
cax.xaxis.set_label_position('top')
cax.set_xlabel(r'$\rm log_{10}(\dot{M}_{accr}/M_{\odot}\ yr^{-1})$', fontsize=7)
cax.tick_params(axis='x', labelsize=6)


filename = f'figs/Mbh_Eddington.pdf'
print(filename)

fig.savefig(filename)

