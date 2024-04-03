import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from flares_utility.stats import poisson_confidence_interval
import cmasher as cmr
import h5py
import flares_utility.analyse as analyse

from unyt import c, Msun, yr, g, s, erg, Lsun

# set style
plt.style.use('../matplotlibrc.txt')

tags = ['005_z010p000', '006_z009p000', '007_z008p000',
        '008_z007p000', '009_z006p000', '010_z005p000']
redshifts = [10, 9, 8, 7, 6, 5]


binw = 0.2
X_limits = [44.6, 46.49]
Y_limits = [-7.9, -2.01]


filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'
flares = analyse.analyse(filename)


class DPL_LF:

    def __init__(self, phi_star, L_star, gamma_1, gamma_2, log = False):

        if log:
            self.log10phi_star = phi_star
            self.phi_star = 10**phi_star
            self.log10L_star = L_star
            self.L_star = 10**L_star

        else:
            self.phi_star = phi_star
            self.L_star = L_star

        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
           
    def phi(self, L):
        return self.phi_star/((L/self.L_star)**self.gamma_1+(L/self.L_star)**self.gamma_2)

    # def log10phi(self, log10L):
    #     return self.log10phi_star - ((L/self.L_star)**self.gamma_1+(L/self.L_star)**self.gamma_2)




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


h = 0.6777  # Hubble parameter
blackhole_accretion_rate_units = 6.446E23 * g / s / h
radiative_efficiency = 0.1
bolometric_luminosity_conversion = radiative_efficiency * c**2


obs_colours = cmr.take_cmap_colors('cmr.infinity', 5, cmap_range = (0.1,0.9))

colours = cmr.take_cmap_colors('cmr.bubblegum_r', len(redshifts))

for z, c, ax in zip(redshifts, colours, axes.flatten()):


    ax.fill_between([0, 45.],[-9,-9],[0,0],color='k',alpha=0.05)

    tag = flares.tag_from_zed[z]

    ## ---- get data
    phi = np.zeros(len(bin_centres))
    N = np.zeros(len(bin_centres))

    # Lbol BH

    blackhole_accretion_rate = flares.load_dataset(tag, *['Galaxy', 'BH_Mdot'])

    for i, (sim, w) in enumerate(zip(flares.sims, flares.weights)):

        x = np.array(blackhole_accretion_rate[sim]) * blackhole_accretion_rate_units * bolometric_luminosity_conversion
        x = x.to('erg/s').value
        x = x[x>0.0]
        x = np.log10(x) 

        N_temp, _ = np.histogram(x, bins = bin_edges)
        N += N_temp
        phi_temp = (N_temp / V) / binw
        phi += phi_temp * w

    upper = np.array([poisson_confidence_interval(n)[1] for n in N])
    lower = np.array([poisson_confidence_interval(n)[0] for n in N])

    phi_upper = phi * upper / N
    phi_lower = phi * lower / N
    ax.fill_between(bin_centres, np.log10(phi_upper), np.log10(phi_lower), alpha=0.1, color=c)

    ax.plot(bin_centres, np.log10(phi), ls ='-', c=c, lw=2, alpha=1.0, zorder=2)
    # ax.plot(bin_centres[N>4], np.log10(phi[N>4]), ls = '-', c=c, lw=2)

    # plot z=5 LF on all plots
    if (z == 5):
        for ax_ in axes.flatten()[:-1]:
            ax_.plot(bin_centres, np.log10(phi), ls='-', c=colours[-1], lw=1, alpha=0.1, zorder=2)


    # for log10Lbolsun in [11.5, 12, 12.5]:

    #     log10Lbol = log10Lbolsun + np.log10((1*Lsun).to('erg/s').value)
    #     ax.axvline(log10Lbol, lw=1, c='k', alpha=0.1)
    #     ax.text(log10Lbol-0.1, -7.6, rf'$\rm 10^{{ {log10Lbolsun} }}\ L_{{\odot}}$', horizontalalignment='left', verticalalignment='bottom', rotation=90.,fontsize = 6, c='k',alpha=0.5)



    ax.text(0.5, 1.02, rf'$\rm z={z}$', horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes, fontsize = 7)

    ax.set_xlim(X_limits)
    ax.set_ylim(Y_limits)

    # Kokerev observations 

    if (z == 5) or (z == 6):
        x = [44, 45, 46]
        y = [-4.09, -4.20, -4.82]
        yerr = [0.2, 0.1, 0.18]
        # ax.errorbar(x, y, fmt="o", c='k', markersize=7, zorder=0)
        ax.errorbar(x, y, yerr=yerr, fmt="o", c='0.5', markersize=4, label = r'$\rm Kokorev+2023\ (4.5<z<6.5)$', zorder=1)
        

    if (z == 7) or (z == 8):
        x = [45, 46, 47]
        y = [-4.24, -4.48, -5.19]
        yerr = [0.23, 0.15, 0.29]
        # ax.errorbar(x, y, fmt="o", c='k', markersize=7, zorder=0)
        ax.errorbar(x, y, yerr=yerr, fmt="o", c='0.5', markersize=4, label = r'$\rm Kokorev+2023\ (6.5<z<8.5)$', zorder=1)


    # Greene observations

    # if z==5 or z==6:

    #     x = [44, 45, 46]
    #     y = np.log10(np.array([1, 4.2, 1]))-5.
    #     # xerr = [0.5, 0.5, 0.5]
    #     yerr = [0.2]*3
    #     lolims = [1,1,1]
    #     # ax.errorbar(x, y, yerr=yerr, fmt="s", c='k', markersize=5)
    #     ax.errorbar(x, y, yerr=yerr, lolims=lolims, fmt="s", c='0.5', markersize=3, label = r'$\rm Greene+2023\ (4.5<z<6.5)$', zorder=1)

    
    # Shen observations

    if z==5 or z==6:

        shen = {}
        shen['free'] = {}
        shen['free'][5.] = (-5.258, 12.319, 0.26, 1.916)
        shen['free'][6.] = (-8.019, 13.709, 1.196, 2.349)
        shen['polished'] = {}
        shen['polished'][5.] = (-5.243, 12.309, 0.245, 1.912)
        shen['polished'][6.] = (-5.452, 11.978, 1.509, 1.509)

        for model, ls in zip(['free', 'polished'],['-','-.']):
            lf = DPL_LF(*shen[model][z], log=True)

            log10L = np.arange(44, 47., 0.01) 
            L = 10**log10L * erg/s
            phi = lf.phi(L.to('Lsun').value)

            ax.plot(log10L, np.log10(phi), label = rf'$\rm Shen+2020\ ({model})$', c='0.5', ls=ls, lw=1, zorder=1) 



    # Add Illustris
    # include_illustris = True
    # if include_illustris:       
    #     z_to_snaphots = {5: 49, 6: 45, 7: 41}
    #     if z in [5,6, 7]:

    #         data_dir = '/Users/sw376/Dropbox/Research/data/simulations/illustris'
    #         simulation = 'Illustris-1'
    #         fields = ['SubhaloBHMass']
    #         X = il.groupcat.loadSubhalos(f'{data_dir}/{simulation}/outputs',z_to_snaphots[z],fields=fields)
    #         X = X[X>0.0]
    #         X /= 0.978 # -> (10^10 Msol)/(Gyr)
    #         log10X = np.log10(X) + 10. - 9. # Msol/yr
    #         log10X += log10conversion2 # bolometric luminosity

    #         N, bin_edges = np.histogram(log10X, bins=bin_edges)
    #         print('Illustris', z, N)
    #         phi = N/(100**3)/binw
    #         ax.plot(bin_centres, np.log10(phi), c='0.7', lw=1, ls=':', zorder=0)


    # include_tng100 = True
    # if include_tng100:
    #     # Add illustris TNG100
    #     z_to_snaphots = {5: 17, 6: 13, 7: 11}
    #     if z in [5,6,7]:
    #         data_dir = '/Users/sw376/Dropbox/Research/data/simulations/illustris'
    #         simulation = 'TNG100-1'
            
    #         # Sum of the instantaneous accretion rates 𝑀 of all blackholes in this subhalo.
    #         # (10^10 Msol/h)/(0.978 Gyr/h)
    #         fields = ['SubhaloBHMdot']


    #         X = il.groupcat.loadSubhalos(f'{data_dir}/{simulation}/outputs',z_to_snaphots[z],fields=fields)
    #         X = X[X>0.0]
    #         X /= 0.978 # -> (10^10 Msol)/(Gyr)
    #         log10X = np.log10(X) + 10. - 9. # Msol/yr
    #         log10X += log10conversion2 # bolometric luminosity

    #         print(np.min(log10X), np.median(log10X), np.max(log10X))

    #         N, bin_edges = np.histogram(log10X, bins=bin_edges)
    #         print('TNG100', z, N)
    #         phi = N/(100**3)/binw
    #         ax.plot(bin_centres, np.log10(phi), c='0.7', lw=1, ls='--', zorder=0)

    # include_tng300 = True
    # if include_tng300:
    #     z_to_snaphots = {5: 17, 6: 13, 7: 11}
    #     if z in [5, 6, 7]:

    #         data_dir = '/Users/sw376/Dropbox/Research/data/simulations/illustris'
    #         simulation = 'TNG300-1'
    #         fields = ['SubhaloBHMass']
    #         X = il.groupcat.loadSubhalos(
    #             f'{data_dir}/{simulation}/outputs',
    #             z_to_snaphots[z],
    #             fields=fields)
    #         X = X[X>0.0]
    #         X /= 0.978 # -> (10^10 Msol)/(Gyr)
    #         log10X = np.log10(X) + 10. - 9. # Msol/yr
    #         log10X += log10conversion2 # bolometric luminosity

    #         N, bin_edges = np.histogram(log10X, bins=bin_edges)
    #         print('TNG300', z, N)
    #         phi = N/(300**3)/binw
    #         ax.plot(bin_centres, np.log10(phi), c='0.7', lw=1, ls='-.', zorder=0)


    ax.legend(fontsize=7)





fig.text(0.04, bottom+(top-bottom)/2, r'$\rm\log_{10}(\phi/Mpc^{-3}\ dex^{-1})$', rotation = 90, va='center', fontsize = 9)
fig.text(left+(right-left)/2, 0.08, r'$\rm \log_{10}(L_{bol}/erg\ s^{-1})$', ha='center', fontsize = 9)


# handles = []
# handles.append(Line2D([0], [0], label=r'$\rm stars$', color='0.5', lw=1, alpha=0.5, ls='--', c='k'))
# handles.append(Line2D([0], [0], label=r'$\rm blackholes$', color='0.5', lw=2, alpha=0.5, ls='-', c='k'))
# handles.append(Line2D([0], [0], label=r'$\rm total$', color='0.5', lw=1, alpha=0.5, ls=':', c='k'))

# fig.legend(handles=handles, fontsize=8, labelspacing=0.1, loc = 'outside lower center', ncol=len(handles))

fig.savefig(f'figs/Lbol_DF.pdf')


fig.clf()
