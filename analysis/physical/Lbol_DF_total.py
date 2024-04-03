
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
import cmasher as cmr
import h5py
import flares_utility.analyse as analyse
from unyt import c, Msun, yr, g, s, erg, Lsun

# set style
plt.style.use('../matplotlibrc.txt')

tags = ['005_z010p000', '006_z009p000', '007_z008p000',
        '008_z007p000', '009_z006p000', '010_z005p000']
redshifts = [10, 9, 8, 7, 6, 5]


binw = 0.25
X_limits = [44.61, 46.49]
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



for z, c, ax in zip(redshifts, cmr.take_cmap_colors('cmr.bubblegum_r', len(redshifts)), axes.flatten()):


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



    ax.plot(bin_centres, np.log10(phi), ls = ':', c=c, lw=1)
    # ax.plot(bin_centres[N>4], np.log10(phi[N>4]), ls = '-', c=c, lw=2)



    # Lbol * 

    phi = np.zeros(len(bin_centres))
    N = np.zeros(len(bin_centres))

    LbolStar = flares.load_dataset(tag, *['Galaxy/BPASS_2.2.1/Chabrier300/Indices/Lbol', 'DustModelI'])

    for i, (sim, w) in enumerate(zip(flares.sims, flares.weights)):

        x = np.array(LbolStar[sim])
        x = x[x>0.0]
        x = np.log10(x)


        N_temp, _ = np.histogram(x, bins = bin_edges)

        N += N_temp

        phi_temp = (N_temp / V) / binw

        phi += phi_temp * w

    ax.plot(bin_centres, np.log10(phi), ls = '--', c=c, lw=1)
    # ax.plot(bin_centres[N>4], np.log10(phi[N>4]), ls = '--', c=c, lw=1)


    # TOTAL

    phi = np.zeros(len(bin_centres))
    N = np.zeros(len(bin_centres))

    for i, (sim, w) in enumerate(zip(flares.sims, flares.weights)):

        x1 = np.array(blackhole_accretion_rate[sim]) * blackhole_accretion_rate_units * bolometric_luminosity_conversion

        x = np.array(LbolStar[sim]) + x1.to('erg/s').value
        x = x[x>0.0]
        x = np.log10(x)

        N_temp, _ = np.histogram(x, bins = bin_edges)

        N += N_temp

        phi_temp = (N_temp / V) / binw

        phi += phi_temp * w

    ax.plot(bin_centres, np.log10(phi), ls = '-', c=c, lw=2)
    # ax.plot(bin_centres[N>4], np.log10(phi[N>4]), ls = ':', c=c, lw=1, alpha=1)



    for log10Lbolsun in [11.5, 12, 12.5]:

        log10Lbol = log10Lbolsun + np.log10((1*Lsun).to('erg/s').value)
        ax.axvline(log10Lbol, lw=1, c='k', alpha=0.1)
        ax.text(log10Lbol-0.1, -7.6, rf'$\rm 10^{{ {log10Lbolsun} }}\ L_{{\odot}}$', horizontalalignment='left', verticalalignment='bottom', rotation=90.,fontsize = 6, c='k',alpha=0.5)



    ax.text(0.5, 1.02, rf'$\rm z={z}$', horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes, fontsize = 7)

    ax.set_xlim(X_limits)
    ax.set_ylim(Y_limits)

fig.text(0.04, bottom+(top-bottom)/2, r'$\rm\log_{10}(\phi/Mpc^{-3}\ dex^{-1})$', rotation = 90, va='center', fontsize = 9)
fig.text(left+(right-left)/2, 0.08, r'$\rm \log_{10}(L_{bol}/erg\ s^{-1})$', ha='center', fontsize = 9)


handles = []
handles.append(Line2D([0], [0], label=r'$\rm stars$', color='0.5', lw=1, alpha=0.5, ls='--', c='k'))
handles.append(Line2D([0], [0], label=r'$\rm blackholes$', color='0.5', lw=1, alpha=0.5, ls=':', c='k'))
handles.append(Line2D([0], [0], label=r'$\rm total$', color='0.5', lw=2, alpha=0.5, ls='-', c='k'))

fig.legend(handles=handles, fontsize=8, labelspacing=0.1, loc = 'outside lower center', ncol=len(handles))

fig.savefig(f'figs/Lbol_DF_total.pdf')


fig.clf()
