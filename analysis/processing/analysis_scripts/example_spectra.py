import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt
import h5py

# load the grid
hf = h5py.File('agn_grid.h5', 'r')

# Load model temperatures + wavelength (same for each spectrum)
T_AGN = hf['T_AGN'][()]
lam = hf['lambda'][()]

cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin = np.log10(min(T_AGN)), vmax = np.log10(max(T_AGN)))

fig = plt.figure(figsize=(3,3))
left  = 0.1
bottom = 0.1
width = 0.75
height = 0.75
ax = fig.add_axes((left, bottom, width, height))

for i in range(len(T_AGN)):
    # Plotting total spectrum for each temperature
    ax.plot(np.log10(lam), np.log10(hf['total'][i][()]), c = cmap(norm(np.log10(T_AGN[i]))))

ax.axvline(np.log10(1500), c='k', ls='--')
ax.axvline(np.log10(2500), c='k', ls='--')

cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
cmapper.set_array([])

cax = fig.add_axes([width+left, bottom, 0.05, height])
bar = fig.colorbar(cmapper, cax=cax, orientation='vertical', format='%d')
bar.set_ticks([4, 5, 6])
cax.set_ylabel(r'$\rm log_{10}[T_{AGN} \; /\; K]$')


ax.set_ylim(40, 44)
ax.set_xlim(3, 3.6)

ax.set_ylabel(r"$\rm log_{10}[\lambda F_{\lambda} \; / \; erg \;s^{-1}]$")
ax.set_xlabel(r"$\rm log_{10}[\lambda \; / \; \AA]$")

fig.savefig(f'figures/total_test.pdf', bbox_inches='tight')
