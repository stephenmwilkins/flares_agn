import numpy as np
import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import h5py
from scipy.stats import binned_statistic
from astropy.cosmology import Planck13 as cosmo, z_at_value
import astropy.units as u
plt.style.use('http://stephenwilkins.co.uk/matplotlibrc.txt')
from unyt import c, Msun, yr, Lsun, g, s

make_combined_plot = False
make_individual_plot = False

sims = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
           '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35',
           '36', '37', '38', '39']

# conversion from Mbh to LEdd
log10LEdd_conversion = np.log10(3.2) + 4 + np.log10((1*Lsun).to('erg/s').value)

# conversion from Mdot -> to Lbol
Lbol_conversion = 0.1 * Msun * c**2 / yr
log10Lbol_conversion = np.log10(Lbol_conversion.to('erg/s').value)


timescales = [10, 20, 50, 100, 200]


if make_combined_plot:

    fig2 = plt.figure(figsize=(3.5, 3.5))

    left = 0.15
    height = 0.8
    bottom = 0.15
    width = 0.8

    ax2 = fig2.add_axes((left, bottom, width, height))
    ax2.axhline(0.0, c='k', lw=2, alpha=0.2)



with h5py.File('data/accretion_rates.h5', 'r') as hf:

    # loop over all simulations

    masses = np.array([])
    accretion_rates = {}
    accretion_rates['instant'] = np.array([])
    for timescale in timescales:
        accretion_rates[timescale] = np.array([])

    for sim in hf.keys():
        masses = np.concatenate((masses, hf[f'{sim}/Mbh'][()]))
        accretion_rates['instant'] = np.concatenate((accretion_rates['instant'], hf[f'{sim}/Mdot/instant'][()]))
        for timescale in timescales:
            accretion_rates[timescale] = np.concatenate((accretion_rates[timescale], hf[f'{sim}/Mdot/{timescale}'][()]))

    log10LEdd = np.log10(masses) + log10LEdd_conversion
    log10Lbol = np.log10(accretion_rates['instant']) + log10Lbol_conversion
    eddington_ratio = log10Lbol - log10LEdd

    print('instant:',
          np.sum(accretion_rates['instant']),
          np.min(eddington_ratio),
          np.max(eddington_ratio))

    for timescale in timescales:

        s = (accretion_rates['instant'] != 0.0) & (accretion_rates[timescale] != 0.0)

        ratio = np.log10(accretion_rates['instant']/accretion_rates[timescale])

        print(timescale, np.sum(accretion_rates[timescale]))

        norm = Normalize(vmin=-6., vmax=0.0)
        cmap = cmr.gem


        mass_bins = np.arange(7., 9.5, 0.25)

        median, bin_edges, N = binned_statistic(np.log10(masses), ratio, bins=mass_bins, statistic='median')
        mean, bin_edges, N = binned_statistic(np.log10(masses), ratio, bins=mass_bins, statistic='mean')


        if make_combined_plot:

            # plot median

            
            ax2.plot(mass_bins[:-1], median, label=f'{timescale}')
            # ax2.plot(mass_bins[:-1], mean, label=f'{timescale}', ls=':')

        if make_individual_plot:

            fig = plt.figure(figsize=(3.5, 3.))

            left = 0.15
            height = 0.8
            bottom = 0.15
            width = 0.6
            hwidth = 0.15

            ax = fig.add_axes((left, bottom, width, height))
            axh = fig.add_axes((left+width, bottom, hwidth, height))

            ax.axhline(0.0, c='k', lw=2, alpha=0.2)
            axh.axhline(0.0, c='k', lw=2, alpha=0.2)

            # calculate the ratio of instantaneous accretion to averaged 

            accretion_ratio = np.sum(accretion_rates['instant'])/np.sum(accretion_rates[timescale])

            # ax.text(7.2, 3.3, rf'$\rm \dot{{M}}_{{\bullet, instant}}/\dot{{M}}_{{\bullet, {timescale}\ Myr}}={accretion_ratio:.2f}$', fontsize=10)

            ax.scatter(np.log10(masses[s]), ratio[s], s=3, c=cmap(norm(eddington_ratio[s])), alpha=0.5)

            ax.plot(mass_bins[:-1], median, label=f'{timescale}', c='k', lw=2)


            axh.hist(ratio[s], bins=np.arange(-5., 5., 0.2), orientation='horizontal', color='0.8')
            axh.axhline(np.median(ratio), c='k', ls='-', lw=1)
            axh.axhline(np.mean(ratio), c='k', ls=':', lw=1)

            ax.set_xlim([7.01, 9.24])
            ax.set_xticks(np.arange(7.5, 9.5, 0.5))
            ylim = [-5., 2.5]
            ax.set_ylim(ylim)
            axh.set_ylim(ylim)

            axh.xaxis.set_ticks([])
            axh.yaxis.set_label_position("right")
            axh.yaxis.tick_right()

            ax.set_ylabel(rf'$\rm \log_{{10}}(\dot{{M}}_{{\bullet, instant}}/\dot{{M}}_{{\bullet, {timescale}\ Myr}})$')
            ax.set_xlabel(r'$\rm \log_{10}(M_{\bullet}/M_{\odot})$')

            fig.savefig(f'figs/accretion_rates_{timescale}.pdf')
            fig.clf()



if make_combined_plot:

    ax2.legend(fontsize=8)

    ax2.set_xlim([7., 9.49])
    ax2.set_ylim([-4., 4.])

    ax2.set_ylabel(rf'$\rm \log_{{10}}(\dot{{M}}_{{\bullet, instant}}/\dot{{M}}_{{\bullet, averaged}})$')
    ax2.set_xlabel(r'$\rm \log_{10}(M_{\bullet}/M_{\odot})$')

    fig2.savefig(f'figs/accretion_rates_combined.pdf')
    fig2.clf()