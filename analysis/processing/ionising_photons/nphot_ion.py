import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt

import flares

import astropy.constants as constants
import astropy.units as units

import FLARE.photom as photom
import FLARE.plt as fplt

from scipy.integrate import simps, trapz

import astropy.constants as const
import astropy.units as u

from scipy.interpolate import interp1d

import _pickle as pickle

bol_correction = pickle.load(open('bolometric_correction_ion.p', 'rb'))
ratio_from_t = interp1d(bol_correction['AGN_T'], bol_correction['ratio']['xi'])
fuv_bolcorr = interp1d(bol_correction['AGN_T'], bol_correction['ratio']['FUV'])

spec_type = 'Pure_Stellar' #['DustModelI','Intrinsic','No_ISM','Pure_Stellar']

mass_cut = 5.

def t_bb(m, m_dot):
    return 2.24*10**9*(m_dot)**(1/4)*(m)**(-1/2) #2.24*10**9*m_dot**(4)*m**(-8) #

def l_agn(m_dot, etta=0.1):
    m_dot = (m_dot*u.M_sun/u.yr).to(u.kg / u.s) # accretion rate in SI
    c = const.c #speed of light
    etta = etta
    l = (etta*m_dot*c**2).to(u.erg/u.s)
    return np.log10(l.value) # output in log10(erg/s)


cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=5., vmax=10.)

cmap2 = mpl.cm.cool
norm2 = mpl.colors.Normalize(vmin=5., vmax=10.)


flares_dir = '../../../../data/simulations'

fl = flares.flares(f'{flares_dir}/flares_no_particlesed.hdf5', sim_type='FLARES') #_no_particlesed

halo = fl.halos

tags = fl.tags

MBH = fl.load_dataset('BH_Mass', arr_type='Galaxy') # Black hole mass of galaxy
MDOT = fl.load_dataset('BH_Mdot', arr_type='Galaxy') # Black hole accretion rate
MS = fl.load_dataset('Mstar_30', arr_type='Galaxy') # Black hole accretion rate
LFUV = fl.load_dataset('FUV', arr_type=f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic/')
LBOL = fl.load_dataset('Intrinsic', arr_type=f'Galaxy/BPASS_2.2.1/Chabrier300/Indices/Lbol/')

#LUM = fl.load_dataset('DustModelI', arr_type=f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity')


xi_gal = pickle.load(open(flares_dir+'/xi_new.p', 'rb'))

Y = MBH
X = MDOT

df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
weights = np.array(df['weights'])


fig, axes = plt.subplots(2, 3, figsize = (6, 4), sharex = True, sharey=True)
fig.subplots_adjust(left=0.07, bottom=0.15, top=1.0, right=0.85, wspace=0.0, hspace=0.0)

for i, tag in enumerate(np.flip(fl.tags)):


    z = np.flip(fl.zeds)[i]
    ws, x, y, mstar, lstar, lbol, xi= np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    for ii in range(len(halo)):
        s = (np.log10(MS[halo[ii]][tag])+10 > 7)
        ws = np.append(ws, np.ones(np.shape(X[halo[ii]][tag][s]))*weights[ii])
        x = np.append(x, X[halo[ii]][tag][s])
        y = np.append(y, Y[halo[ii]][tag][s])
        mstar = np.append(mstar, np.log10(MS[halo[ii]][tag][s])+10)
        lstar = np.append(lstar, np.log10(LFUV[halo[ii]][tag][s]))
        lbol = np.append(lbol, np.log10(LBOL[halo[ii]][tag][s]))
        xi = np.append(xi, xi_gal[halo[ii]][tag][spec_type][s])

    h = 0.6777  # Hubble parameter


    # converting MBHacc units to M_sol/yr
    x *= h * 6.445909132449984E23  # g/s
    x = x/constants.M_sun.to('g').value  # convert to M_sol/s
    x *= units.yr.to('s')  # convert to M_sol/yr


    y *= 10**10
    b = t_bb(y, x)

    s_t = np.array(b) > 10**4

    q = np.array([l_agn(g, etta=0.1) for g in x[s_t]])

    ws = ws[s_t]


    xi_agn = (ratio_from_t(b[s_t])) #* 10 ** q
    uv_agn = (fuv_bolcorr(b[s_t]))*10**q / ((const.c/(1500*u.AA).to(u.m)).to(u.Hz)).value

    x = mstar[s_t]

    xi = np.log10(xi[s_t]*10**lstar[s_t])

    y = np.log10(xi_agn*uv_agn)-xi

    # -- this will calculate the weighted quantiles of the distribution
    quantiles = [0.84, 0.50, 0.16]  # quantiles for range
    bins = np.arange(7.5, 12, 0.25)  #  x-coordinate bins
    bincen = (bins[:-1] + bins[1:]) / 2.
    out = flares.binned_weighted_quantile(x, y, ws, bins, quantiles)

    #out2 = flares.binned_weighted_quantile(x, xi, ws, bins, quantiles)

    # --- plot the median and quantiles for bins with >10 galaxies

    N, bin_edges = np.histogram(x, bins=bins)
    Ns = N > 10

    '''
    axes.flatten()[i].plot(bincen, out[:, 1], c='xkcd:darkgreen', ls=':')
    axes.flatten()[i].plot(bincen[Ns], out[:, 1][Ns], c='xkcd:darkgreen', ls='-')
    axes.flatten()[i].fill_between(bincen[Ns], out[:, 0][Ns], out[:, 2][Ns], color='xkcd:darkgreen',
                                   alpha=0.4)

    axes.flatten()[i].plot(bincen, out2[:, 1], c='xkcd:darkgreen', ls=':')
    axes.flatten()[i].plot(bincen[Ns], out2[:, 1][Ns], c='xkcd:darkgreen', ls='--')
    axes.flatten()[i].fill_between(bincen[Ns], out2[:, 0][Ns], out2[:, 2][Ns], color='xkcd:darkgreen',
                                   alpha=0.4)

    '''
    axes.flatten()[i].axhline(0, alpha=0.8, c='k', ls='--', linewidth=1)

    s_outshine = (y > 0)

    print(np.sum(s_outshine))

    axes.flatten()[i].scatter(x[s_outshine], y[s_outshine], s=2, color=cmap(norm(z)), alpha=0.5)

    axes.flatten()[i].plot(bincen, out[:, 1], c=cmap(norm(z)), ls=':')
    axes.flatten()[i].plot(bincen[Ns], out[:, 1][Ns], c=cmap(norm(z)), ls='-')
    axes.flatten()[i].fill_between(bincen[Ns], out[:, 0][Ns], out[:, 2][Ns], color=cmap(norm(z)),
                                   alpha=0.4)

    #axes.flatten()[i].plot(bincen, out2[:, 1], c=cmap(norm(z)), ls=':')
    #axes.flatten()[i].plot(bincen[Ns], out2[:, 1][Ns], c=cmap(norm(z)), ls='--')
    #axes.flatten()[i].fill_between(bincen[Ns], out2[:, 0][Ns], out2[:, 2][Ns], color=cmap(norm(z)),
    #                               alpha=0.4)




    axes.flatten()[i].set_ylim(-8, 1.5)
    axes.flatten()[i].set_xlim(7.9, 11.5)
    #axes.flatten()[i].set_ylim(24, 26.5)

    axes.flatten()[i].set_xticks([8, 9, 10, 11])
    #axes.flatten()[i].set_yticks([24, 25, 26])

    axes.flatten()[i].text(0.1, 0.9, r'$\rm z={0:.0f}$'.format(z), fontsize=8, transform=axes.flatten()[i].transAxes,
                           color=cmap(norm(z)), ha='left')

#axes.flatten()[i].plot(-99, -99, color='k', ls='-', label='AGN', alpha=0.8)
#axes.flatten()[i].plot(-99, -99, color='k', ls='--', label='Stellar', alpha=0.8)
axes.flatten()[i].legend(loc='lower right')

fig.text(0.01, 0.55, r'$\rm log_{10}[N_{\gamma_{ion}, AGN} \; / \; N_{\gamma_{ion}, Stellar}]$', ha = 'left', va = 'center', rotation = 'vertical', fontsize=10)
fig.text(0.45,0.05, r'$\rm log_{10}[M_{*}\;/\;M_{\odot}]$', ha = 'center', va = 'bottom', fontsize=10)

fig.savefig(f'figures/Nphot_grid.pdf', bbox_inches='tight')
fig.clf()

