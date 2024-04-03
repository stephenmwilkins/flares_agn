
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import cmasher as cmr
import flares_utility.analyse as analyse
import flare.plt as fplt
import scipy.stats as stats

# set style
plt.style.use('../matplotlibrc.txt')


X_limits = [11.1, 13.4]
Y_limits = [5.99, 8.99]


filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'
flares = analyse.analyse(filename, default_tags = False)

flares.list_datasets()

fig = plt.figure(figsize = (3.5, 3.5))

left  = 0.15
height = 0.7
bottom = 0.15
width = 0.8

ax = fig.add_axes((left, bottom, width, height))
# ax2 = ax.twinx()
cax = fig.add_axes([left, bottom+height, width, 0.03])



# --- FLARES


norm = Normalize(vmin=-0.39, vmax=0.39)
cmap = cmr.get_sub_cmap('cmr.copper', 0.2, 0.8)

bins = np.arange(9., 15., 0.5)
bin_centres = 0.5*(bins[1:]+bins[:-1])

z = 5.0
tag = '010_z005p000'

tag = flares.tag_from_zed[z]
Mbh = flares.load_dataset(tag, *['Galaxy', 'BH_Mass'])
Mstar = flares.load_dataset(tag, *['Galaxy', 'Mstar_30'])
Mgal = flares.load_dataset(tag, *['Galaxy', 'SubhaloMass'])


delta_bin_edges = np.array([-0.4, -0.15, -0.05, 0.05, 0.15, 0.4])




for bin_low, bin_high in zip(delta_bin_edges[:-1], delta_bin_edges[1:]):

    X = np.array([])
    Y = np.array([])

    for i, (sim, w, delta) in enumerate(zip(flares.sims, flares.weights, flares.deltas)):

        if (np.log10(delta+1) > bin_low) & (np.log10(delta+1) < bin_high):

            x = np.log10(np.array(Mgal[sim])) + 10.
            y = np.log10(np.array(Mbh[sim])) + 10.

            s = y>6.0

            if np.sum(s)>0.:
                X = np.hstack((X, x[s]))
                Y = np.hstack((Y, y[s]))

    print(X)
    print(Y)

    med, _, _ = stats.binned_statistic(X, Y, statistic='median', bins=bins)

    d = 0.5*(bin_low+bin_high)

    print(med)

    ax.plot(bin_centres, med, c=cmap(norm(d)), lw=1)


ax.set_xlim(X_limits)
ax.set_ylim(Y_limits)

ax.set_xlabel(r'$\rm \log_{10}(M_{gal}/M_{\odot})$')
ax.set_ylabel(r'$\rm \log_{10}(M_{\bullet}/M_{\odot})$')

# add colourbar
cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(cmapper, cax=cax, orientation='horizontal')
cax.xaxis.tick_top()
cax.xaxis.set_label_position('top')
cax.set_xlabel(r'$\rm log_{10}(1+\delta_{14})$', fontsize=7)
cax.tick_params(axis='x', labelsize=6)


fig.savefig(f'figs/Mh_Mbh_environment.pdf')


fig.clf()
