
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import cmasher as cmr
import flares_utility.analyse as analyse

# set style
plt.style.use('../matplotlibrc.txt')

X_limits = [-0.39, 0.39]
Y_limits = [0, 69.]


filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'
flares = analyse.analyse(filename)

fig = plt.figure(figsize = (3.5, 3.5))

left  = 0.15
height = 0.8
bottom = 0.15
width = 0.8

ax = fig.add_axes((left, bottom, width, height))
# ax2 = ax.twinx()
# cax = fig.add_axes([left, bottom+height, width, 0.03])



# --- FLARES





norm = Normalize(vmin=-0.39, vmax=0.39)
cmap = cmr.get_sub_cmap('cmr.copper', 0.2, 0.8)


z = 5.0
tag = '010_z005p000'

tag = flares.tag_from_zed[z]
Mbh = flares.load_dataset(tag, *['Galaxy', 'BH_Mass'])
Mstar = flares.load_dataset(tag, *['Galaxy', 'Mstar_30'])

for i, (sim, w, delta) in enumerate(zip(flares.sims, flares.weights, flares.deltas)):

    x = np.log10(np.array(Mbh[sim])) + 10.
    x = x[x>0.0]

    N7 = np.sum(x>7.0)
    N8 = np.sum(x>8.0)

    ax.scatter(np.log10(1+delta), N7, c=cmap(norm(np.log10(1+delta))), s=20)

    x = np.log10(np.array(Mstar[sim])) + 10.
    x = x[x>0.0]

    # MstarN10 = np.sum(x>10.0)

    # ax.scatter(np.log10(1+delta), MstarN10, c=cmap(norm(np.log10(1+delta))), s=10, marker='*')

    # # ax.scatter(np.log10(1+delta), N8, c='k', marker='p', s=25)
    # # ax.scatter(np.log10(1+delta), N8, c=cmap(norm(np.log10(1+delta))), marker='p', s=15)
    # print(sim, delta)






delta = np.arange(-1.,10, 0.01)


for N in [1, 2, 4,8,16,32]:
    ax.plot(np.log10(delta+1), N*(delta+1), c='k', lw=1, ls=':', alpha = 0.3)


ax.set_xlim(X_limits)
ax.set_ylim(Y_limits)

ax.set_xlabel(r'$\rm \log_{10}(1+\delta_{14})$')
ax.set_ylabel(r'$\rm N(M_{\bullet}>10^{7}\ M_{\odot})$')

# # add colourbar
# cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
# fig.colorbar(cmapper, cax=cax, orientation='horizontal')
# cax.xaxis.tick_top()
# cax.xaxis.set_label_position('top')
# cax.set_xlabel(r'$\rm log_{10}(1+\delta_{14})$', fontsize=7)
# cax.tick_params(axis='x', labelsize=6)


filename = f'figs/Mbh_N_environment.pdf'
print(filename)
fig.savefig(filename)


fig.clf()
