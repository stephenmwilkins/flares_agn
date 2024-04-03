import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cmasher as cmr
import h5py
import flares_utility.analyse as analyse
from flares_utility.stats import poisson_confidence_interval
from unyt import c, Msun, yr, g, s, erg

import illustris_python as il

# set style
plt.style.use('../matplotlibrc.txt')

tags = ['005_z010p000', '006_z009p000', '007_z008p000',
        '008_z007p000', '009_z006p000', '010_z005p000']
redshifts = [10, 9, 8, 7, 6, 5]


binw = 0.25
X_limits = [6.625, 9.99]
Y_limits = [-8.5, -3.01]


filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'
flares = analyse.analyse(filename)



N = len(redshifts)
left = 0.1
top = 0.975
bottom = 0.05
right = 0.975
# panel_width = (right-left)/N
# panel_height = top-bottom
fig, axes = plt.subplots(6, 2, figsize = (7,9), sharey = True, sharex = True)
plt.subplots_adjust(left=left, top=top, bottom=bottom, right=right, wspace=0.0, hspace=0.0)


eagle = h5py.File('/Users/sw376/Dropbox/Research/data/simulations/flares/EAGLE_REF_sp_info.hdf5', 'r')




# --- FLARES


# flares.list_datasets()

V = (4./3) * np.pi * (flares.radius)**3 # Mpc^3


bin_edges = np.arange(*X_limits, binw)
bin_centres = bin_edges[:-1]+binw/2

quantity = ['Galaxy', 'BH_Mass']

# converting MBHacc units to M_sol/yr
h = 0.6777  # Hubble parameter
bhacc_conversion = h * 6.446E23 * g / s
conversion = 0.1 * bhacc_conversion * c**2
conversion = conversion
log10conversion = np.log10(conversion.to('erg/s').value)
print(log10conversion)
# log10conversion = np.log10(conversion.to('Lsun').value)
# print(log10conversion)


for zi, (z, c) in enumerate(zip(redshifts, cmr.take_cmap_colors('cmr.bubblegum_r', len(redshifts)))):

    ax0 = axes[zi,0]
    ax1 = axes[zi,1]

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
            ax0.fill_between(bin_centres, np.log10(phi_upper), np.log10(phi_lower), alpha=0.1, color=c)

        ax0.plot(bin_centres, np.log10(phi), ls = ls_, c=c, lw=lw_, alpha=1.0, zorder=2)
        
        # ax.plot(bin_centres[N>4], np.log10(phi[N>4]), ls = ls_, c=c, lw=lw_)

    if z==5:
        for zi, z in enumerate(redshifts):
            axes[zi, 0].plot(bin_centres, np.log10(phi), ls = '-', c='k', lw=2, alpha=0.1, zorder=-1)

    ax0.plot(bin_centres, np.log10(phi), ls='-', c=c, lw=2, alpha=1.0, zorder=2)
    ax1.plot(bin_centres, np.log10(phi), ls = ls_, c=c, lw=lw_, alpha=1.0, zorder=2)
    # ax.plot(bin_centres[N>4], np.log10(phi[N>4]), ls = ls_, c=c, lw=lw_)


    # ------------------------------------------------------------------------------------
    # Add observations

    co = '0.7'

    # Matthee+2023
    if z==5:
        x = [7.5, 8.1]
        y = [-4.29, -5.05]
        xerr = [0.2, 0.4]
        yerr = [[0.15, 0.30], [0.11, 0.18]]
        ax1.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", c=co, markersize=5, label=r'$\rm Matthee+23$', zorder=0)

    # He+
    if z==5:
        x = np.arange(7.5, 10.7, 0.3)
        y = np.array([12.14, 9.92, 11.30, 13.93, 8.07, 4.46, 0.49, 1.61, 0.19, 0.04, 0.01])
        yerr = np.array([8.7, 6.24, 4.34, 3.68, 2.87, 1.89, 0.28, 1.08, 0.01, 0.01, 0.01])
        # ax.scatter(x, np.log10(y)-7, marker="s", c=c, s=5, label=r'$\rm He+23\ (z=4)$', alpha=0.5)
        yerr_u = np.log10(y+yerr)-np.log10(y)
        yerr_l = np.log10(y)-np.log10(y-yerr)
        print(yerr_l)
        yerr = np.array([yerr_l, yerr_u])
        ax1.errorbar(x, np.log10(y)-7., yerr=yerr, fmt="s", c=co, markersize=3, label=r'$\rm He+23\ (z=4)$', zorder=0)


    # ------------------------------------------------------------------------------------
    # Add other models
    include_astrid = True
    include_eagle = True
    include_bluetides = True
    include_illustris = True
    include_tng100 = True
    include_tng300 = True
    include_simba = True

    if include_simba:
        z_to_snaphots = {
            5: '042', 
            6: '036', 
            7: '030',
            8: '026',
            9: '022',
            10: '019'
            }
        snapshot_z = {
            '019': 9.966759,
            '022': 9.033071,
            '026': 7.962628,
            '030': 7.054463,
            '036': 5.929968,
            '042': 5.024400,
            }

    if z in [5,6,7,8]:

        data_dir = '/Users/sw376/Dropbox/Research/data/simulations/simba'

        snapshot = z_to_snaphots[z]
        h = 0.7

        with h5py.File(f'{data_dir}/m100n1024_{snapshot}.hdf5') as hf:
            print(z)
            X = hf['galaxy_data/dicts/masses.bh'][()]
            log10X = np.log10(X/h)
            N, bin_edges = np.histogram(log10X, bins=bin_edges)
            print('Simba', z, N)
            phi = N/((100/h)**3)/binw
            ax1.plot(bin_centres, np.log10(phi), c='0.7', lw=1, ls=(0, (3, 5, 1, 5, 1, 5)), zorder=0)


    # Add Bluetides
    if include_bluetides:        
        
        if z in [7, 8, 9, 10]:

            data_dir = '/Users/sw376/Dropbox/Research/data/simulations/bluetides/bhcat'
            
            d = np.load(f'{data_dir}/BT_bh_cat_z{z}.0.npy')

            mass = []
            accretion_rate = []

            for mass_, accretion_rate_ in d:
                mass.append(mass_)
                accretion_rate.append(accretion_rate_)

            mass =np.array(mass)
            accretion_rate = np.array(accretion_rate)
            
            log10X = np.log10(mass) 

            N, bin_edges = np.histogram(log10X, bins=bin_edges)
            print('Bluetides', z, N)
            phi = N/((400/h)**3)/binw
            ax1.plot(bin_centres, np.log10(phi), c='0.7', lw=1, ls=(0, (3, 1, 1, 1, 1, 1)), zorder=0)


    # Add Astrid
    if include_astrid:        
        
        if z in [5,6, 7, 8, 9, 10]:

            data_dir = '/Users/sw376/Dropbox/Research/data/simulations/astrid/bhcat'
            
            d = np.load(f'{data_dir}/Astrid_bh_cat_z{z}.0.npy')

            print(d)

            mass = []
            accretion_rate = []

            for mass_, accretion_rate_ in d:
                mass.append(mass_)
                accretion_rate.append(accretion_rate_)

            mass =np.array(mass)
            accretion_rate = np.array(accretion_rate)
            
            log10X = np.log10(mass) 

            N, bin_edges = np.histogram(log10X, bins=bin_edges)
            print('Astrid', z, N)
            phi = N/((250/h)**3)/binw
            ax1.plot(bin_centres, np.log10(phi), c='0.7', lw=1, ls=(5, (10, 3)), zorder=0)

    
    # Add EAGLE
    
    if include_eagle:
        # not exact
        z_to_snaphots = {
            5: '008_z005p037', 
            6: '006_z005p971', 
            7: '005_z007p050',
            8: '004_z008p075',
            9: '003_z008p988',
            10:  '002_z009p993'
            }
        snapshot_z = {
            '002_z009p993': 9.993,
            '003_z008p988': 8.988,
            '004_z008p075': 8.075,
            '005_z007p050': 7.050,
            '006_z005p971': 5.971,
            '008_z005p037': 5.037,
            }

        if z in [5, 6, 7, 8, 9, 10]:
            # get tag
            tag = z_to_snaphots[z]
            log10X = np.log10(eagle[f'{tag}/Galaxy/BH_Mass'][()]) + 10.
            N, bin_edges = np.histogram(log10X, bins=bin_edges)
            phi = N/(100**3)/binw
            print('EAGLE', z, N)
            ax1.plot(bin_centres, np.log10(phi), c='0.7', lw=1, ls='-', zorder=0)

    # Add Illustris
    
    if include_illustris:        
        
        z_to_snaphots = {5: 49, 6: 45, 7: 41}
        if z in [5,6, 7]:

            data_dir = '/Users/sw376/Dropbox/Research/data/simulations/illustris'
            simulation = 'Illustris-1'
            fields = ['SubhaloBHMass']
            X = il.groupcat.loadSubhalos(f'{data_dir}/{simulation}/outputs',z_to_snaphots[z],fields=fields)
            X = X[X>0.0]
            X = X/0.7
            log10X = np.log10(X) + 10. 

            N, bin_edges = np.histogram(log10X, bins=bin_edges)
            print('Illustris', z, N)
            phi = N/(100**3)/binw
            ax1.plot(bin_centres, np.log10(phi), c='0.7', lw=1, ls=':', zorder=0)

    
    if include_tng100:
        # Add illustris TNG100
        z_to_snaphots = {5: 17, 6: 13, 7: 11}
        if z in [5,6,7]:
            data_dir = '/Users/sw376/Dropbox/Research/data/simulations/illustris'
            simulation = 'TNG100-1'
            fields = ['SubhaloBHMass']
            X = il.groupcat.loadSubhalos(f'{data_dir}/{simulation}/outputs',z_to_snaphots[z],fields=fields)
            X = X[X>0.0]
            X = X/0.7
            log10X = np.log10(X) + 10. 

            N, bin_edges = np.histogram(log10X, bins=bin_edges)
            print('TNG100', z, N)
            phi = N/(100**3)/binw
            ax1.plot(bin_centres, np.log10(phi), c='0.7', lw=1, ls='--', zorder=0)

    
    if include_tng300:
        z_to_snaphots = {5: 17, 6: 13, 7: 11}
        if z in [5, 6, 7]:

            data_dir = '/Users/sw376/Dropbox/Research/data/simulations/illustris'
            simulation = 'TNG300-1'
            fields = ['SubhaloBHMass']
            X = il.groupcat.loadSubhalos(
                f'{data_dir}/{simulation}/outputs',
                z_to_snaphots[z],
                fields=fields)
            X = X[X>0.0]
            X = X/0.7
            log10X = np.log10(X) + 10. 

            N, bin_edges = np.histogram(log10X, bins=bin_edges)
            print('TNG300', z, N)
            phi = N/(300**3)/binw
            ax1.plot(bin_centres, np.log10(phi), c='0.7', lw=1, ls='-.', zorder=0)





    ax0.text(0.075, 
             0.85, 
             rf'$\rm z={z}$', 
             horizontalalignment='center', 
             verticalalignment='bottom', 
             transform=ax0.transAxes,
             color=c, 
             fontsize=10)

    ax0.set_xlim(X_limits)
    ax0.set_ylim(Y_limits)


    # for observations
    ax1.legend(fontsize=8)


# for bins
handles = []
handles.append(Line2D([0], [0], label=r'$\rm L_{bol}/erg\ s^{-1}<10^{45}$', color='0.5', lw=1, alpha=1.0, ls='--', c='k'))
handles.append(Line2D([0], [0], label=r'$\rm 10^{45}<L_{bol}/erg\ s^{-1}<10^{46}$', color='0.5', lw=1, alpha=1.0, ls='-.', c='k'))
handles.append(Line2D([0], [0], label=r'$\rm L_{bol}/erg\ s^{-1}>10^{46}$', color='0.5', lw=1, alpha=1.0, ls=':', c='k'))
handles.append(Line2D([0], [0], label=r'$\rm all$', color='0.5', lw=2, alpha=1.0, ls='-', c='k'))
handles.append(Line2D([0], [0], label=r'$\rm z=5$', color='0.5', lw=2, alpha=0.1, ls='-', c='k'))
axes[0,0].legend(loc='upper right', handles=handles, fontsize=8, labelspacing=0.1)

# for simulations
handles = []
handles.append(Line2D([0], [0], label=r'$\rm ASTRID$', color='0.7', lw=1, ls=(5, (10, 3))))
handles.append(Line2D([0], [0], label=r'$\rm Bluetides$', color='0.7', lw=1, ls=(0, (3, 1, 1, 1, 1, 1))))
handles.append(Line2D([0], [0], label=r'$\rm EAGLE$', color='0.7', lw=1, ls='-'))
handles.append(Line2D([0], [0], label=r'$\rm Illustris$', color='0.7', lw=1, ls=':'))
handles.append(Line2D([0], [0], label=r'$\rm Simba$', color='0.7', lw=1, ls=(0, (3, 5, 1, 5, 1, 5))))
handles.append(Line2D([0], [0], label=r'$\rm TNG100$', color='0.7', lw=1, ls='--'))
handles.append(Line2D([0], [0], label=r'$\rm TNG300$', color='0.7', lw=1, ls='-.'))



axes[0,1].legend(handles=handles, fontsize=8, labelspacing=0.1)


fig.text(0.04, bottom+(top-bottom)/2, r'$\rm\log_{10}[\phi/Mpc^{-3}\ dex^{-1}]$', rotation = 90, va='center', fontsize = 11)
fig.text(left+(right-left)/2, 0.015, r'$\rm \log_{10}(M_{\bullet}/M_{\odot})$', ha='center', fontsize = 11)

fig.savefig(f'figs/Mbh_DF_combined.pdf')


fig.clf()
