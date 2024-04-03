import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cmasher as cmr
import flares_utility.analyse as analyse
from flares_utility.stats import poisson_confidence_interval
from unyt import c, Msun, yr, g, s, erg

# set style
plt.style.use('../matplotlibrc.txt')

tags = ['005_z010p000', '006_z009p000', '007_z008p000',
        '008_z007p000', '009_z006p000', '010_z005p000']
redshifts = [10, 9, 8, 7, 6, 5]


binw = 0.25
X_limits = [6.625, 9.99]
Y_limits = [-8.5, -3.49]


filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'
flares = analyse.analyse(filename)



N = len(redshifts)
left = 0.1
top = 0.95
bottom = 0.15
right = 0.975
panel_width = (right-left)/N
panel_height = top-bottom
fig, axes = plt.subplots(2, 3, figsize = (7,5), sharey = True, sharex = True)
plt.subplots_adjust(left=left, top=top, bottom=bottom, right=right, wspace=0.0, hspace=0.1)



# --- FLARES


# flares.list_datasets()

V = (4./3) * np.pi * (flares.radius)**3 # Mpc^3


bin_edges = np.arange(*X_limits, binw)
bin_centres = bin_edges[:-1]+binw/2

quantity = ['Galaxy', 'BH_Mass']





# converting MBHacc units to M_sol/yr
h = 0.6777  # Hubble parameter
bhacc_conversion = 6.446E23 * g / s / h

conversion = 0.1 * bhacc_conversion * c**2
conversion = conversion
log10conversion = np.log10(conversion.to('erg/s').value)
print(log10conversion)
# log10conversion = np.log10(conversion.to('Lsun').value)
# print(log10conversion)

for z, c, ax in zip(redshifts, cmr.take_cmap_colors('cmr.bubblegum_r', len(redshifts)), axes.flatten()):

    tag = flares.tag_from_zed[z]    

    X = flares.load_dataset(tag, *quantity)
    Mstar = flares.load_dataset(tag, 'Galaxy', 'Mstar_30')
    BH_Mdot = flares.load_dataset(tag, *['Galaxy', 'BH_Mdot'])
    
    limits = [(30., 45.0), (45., 46.), (46., 49.0), (30.,47.)]
    # limits = [(10., 11.5), (11.5, 12.0), (12, 13), (10, 15)]
    lw = [1,1,1,2]
    ls = ['--','-.',':','-']

    for limits_, lw_, ls_ in zip(limits, lw, ls): 

        phi = np.zeros(len(bin_centres))
        N = np.zeros(len(bin_centres))

        for i, (sim, w) in enumerate(zip(flares.sims, flares.weights)):

            x = np.log10(np.array(X[sim])) + 10.

            Lbol = np.log10(np.array(BH_Mdot[sim])) + log10conversion

            s = (Lbol>limits_[0])&(Lbol<=limits_[1])

            x = x[s]

            N_temp, _ = np.histogram(x, bins = bin_edges)

            N += N_temp

            phi_temp = (N_temp / V) / binw

            phi += phi_temp * w


        if lw_ == 2:

            upper = np.array([poisson_confidence_interval(n)[1] for n in N])
            lower = np.array([poisson_confidence_interval(n)[0] for n in N])

            phi_upper = phi * upper / N
            phi_lower = phi * lower / N
            ax.fill_between(bin_centres, np.log10(phi_upper), np.log10(phi_lower), alpha=0.1, color=c)

        ax.plot(bin_centres, np.log10(phi), ls = ls_, c=c, lw=lw_, alpha=1.0, zorder=2)
        # ax.plot(bin_centres[N>4], np.log10(phi[N>4]), ls = ls_, c=c, lw=lw_)

    if z==5:
        for ax in axes.flatten():
            ax.plot(bin_centres, np.log10(phi), ls = '-', c='k', lw=2, alpha=0.2, zorder=-1)

    ax.text(0.5, 1.02, rf'$\rm z={z}$', horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes, fontsize = 7)

    ax.set_xlim(X_limits)
    ax.set_ylim(Y_limits)

    ax.legend(fontsize=8)


fig.text(0.04, bottom+(top-bottom)/2, r'$\rm\log_{10}[\phi/Mpc^{-3}\ dex^{-1}]$', rotation = 90, va='center', fontsize = 9)
fig.text(left+(right-left)/2, 0.08, r'$\rm \log_{10}(M_{\bullet}/M_{\odot})$', ha='center', fontsize = 9)



handles = []

# limits = [(44.5, 45.0), (45.0, 45.5), (45.5, 46.0)]
#     lw = [1,1,1,2]
#     ls = ['--','-.',':','-']

handles.append(Line2D([0], [0], label=r'$\rm L_{bol}/erg\ s^{-1}<10^{45}$', color='0.5', lw=1, alpha=1.0, ls='--', c='k'))
handles.append(Line2D([0], [0], label=r'$\rm 10^{45}<L_{bol}/erg\ s^{-1}<10^{46}$', color='0.5', lw=1, alpha=1.0, ls='-.', c='k'))
handles.append(Line2D([0], [0], label=r'$\rm L_{bol}/erg\ s^{-1}>10^{46}$', color='0.5', lw=1, alpha=1.0, ls=':', c='k'))
handles.append(Line2D([0], [0], label=r'$\rm all$', color='0.5', lw=2, alpha=1.0, ls='-', c='k'))
handles.append(Line2D([0], [0], label=r'$\rm z=5$', color='0.5', lw=2, alpha=0.2, ls='-', c='k'))

fig.legend(handles=handles, fontsize=8, labelspacing=0.1, loc = 'outside lower center', ncol=len(handles))



fig.savefig(f'figs/Mbh_DF_Lbol.pdf')


fig.clf()
