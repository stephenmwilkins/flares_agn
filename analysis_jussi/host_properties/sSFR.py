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

mass_cut = 5.

def t_bb(m, m_dot):
    return 2.24*10**9*(m_dot)**(1/4)*(m)**(-1/2) #2.24*10**9*m_dot**(4)*m**(-8) #

def l_agn(m_dot, etta=0.1):
    m_dot = 10**m_dot*1.98847*10**30/(365*24*60*60) # accretion rate in SI
    c = 299792458 #speed of light
    etta = etta
    return np.log10(etta*m_dot*c**2*10**7) # output in log10(erg/s)


cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=5., vmax=10.)

flares_dir = '../../../data/simulations'

fl = flares.flares(f'{flares_dir}/flares.hdf5', sim_type='FLARES')
df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
halo = fl.halos
print(fl.tags) # print list of available snapshots
tags = fl.tags #This would be z=5

MBH = fl.load_dataset('BH_Mass', arr_type='Galaxy') # Black hole mass of galaxy
MS = fl.load_dataset('Mstar_30', arr_type='Galaxy') # Stellar mass
SFR = fl.load_dataset('SFR_inst_30', arr_type='Galaxy') # instantaneous star formation rate

Y = SFR
X = MS


df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
weights = np.array(df['weights'])

fig, axes = plt.subplots(2, 3, figsize = (6, 4), sharex = True, sharey=True)
fig.subplots_adjust(left=0.07, bottom=0.15, top=1.0, right=0.85, wspace=0.0, hspace=0.0)


for i, tag in enumerate(np.flip(fl.tags)):
    z = np.flip(fl.zeds)[i]
    ws, x, y, mbh = np.array([]), np.array([]), np.array([]), np.array([])
    for ii in range(len(halo)):
        s = (np.log10(X[halo[ii]][tag])+10 > 7)
        ws = np.append(ws, np.ones(np.shape(X[halo[ii]][tag][s]))*weights[ii])
        x = np.append(x, X[halo[ii]][tag][s])
        y = np.append(y, Y[halo[ii]][tag][s])
        mbh = np.append(mbh, MBH[halo[ii]][tag][s])

    y = np.log10(y) - np.log10(x*1E10) + 9

    w = np.nonzero(mbh) # this line selects hosts
    w_nonhost = np.invert(w)

    # --- simply print the ranges of the quantities

    y_nonhost = y[w_nonhost]
    x_nonhost = np.log10(x[w_nonhost]) +10
    ws_nonhost = ws[w_nonhost]

    y = y[w]
    x = np.log10(x[w]) +10
    ws = ws[w]



    print(z)
    print(np.min(x),np.median(x),np.max(x))
    print(np.min(y),np.median(y),np.max(y))

    '''
    binw = 0.25
    bins = np.arange(5, 10, binw)
    b_c = bins[:-1] + binw / 2

    N_weighted, edges = np.histogram(x, bins=bins, weights=ws)

    N_weighted_nonhost, edges = np.histogram(x_nonhost, bins=bins, weights=ws_nonhost)

    h = 0.6777
    vol = (4 / 3) * np.pi * (14 / h) ** 3

    phi = N_weighted / (binw * vol)
    phi_nonhost = N_weighted_nonhost / (binw * vol)

    axes.flatten()[i].plot(bins[:-1] + binw / 2, np.log10(phi), c=cmap(norm(z)))
    axes.flatten()[i].plot(bins[:-1] + binw / 2, np.log10(phi_nonhost), '--', c=cmap(norm(z)))

    '''
    # -- this will calculate the weighted quantiles of the distribution
    quantiles = [0.84,0.50,0.16] # quantiles for range
    bins = np.arange(7,11, 0.1) # x-coordinate bins
    bincen = (bins[:-1]+bins[1:])/2.
    out = flares.binned_weighted_quantile(x,y,ws,bins,quantiles)
    out_nonhost = flares.binned_weighted_quantile(x_nonhost,y_nonhost,ws_nonhost,bins,quantiles)

    # --- plot the median and quantiles for bins with >10 galaxies

    N, bin_edges = np.histogram(x, bins=bins)
    Ns = N > 10
    axes.flatten()[i].plot(bincen, out[:, 1], c=cmap(norm(z)), ls=':')
    axes.flatten()[i].plot(bincen[Ns], out[:, 1][Ns], c=cmap(norm(z)))
    axes.flatten()[i].fill_between(bincen[Ns], out[:, 0][Ns], out[:, 2][Ns], color=cmap(norm(z)), alpha=0.2)

    # non-hosts

    N_nonhost, bin_edges_nonhost = np.histogram(x_nonhost, bins=bins)
    Ns_nonhost = N_nonhost > 10
    axes.flatten()[i].plot(bincen, out_nonhost[:, 1], c=cmap(norm(z)), ls='dashdot', alpha=0.6)
    axes.flatten()[i].plot(bincen[Ns_nonhost], out_nonhost[:, 1][Ns_nonhost], c=cmap(norm(z)), ls='--', alpha=0.6)
    axes.flatten()[i].fill_between(bincen[Ns_nonhost], out_nonhost[:, 0][Ns_nonhost], out_nonhost[:, 2][Ns_nonhost], color=cmap(norm(z)), alpha=0.1)

    #axes.flatten()[i].scatter(x, y, c=cmap(norm(z)), alpha=0.5, s=5)

    axes.flatten()[i].text(0.7, 0.9, r'$\rm z={0:.0f}$'.format(z), fontsize=8, transform=axes.flatten()[i].transAxes,
                           color=cmap(norm(z)))
    axes.flatten()[i].set_ylim(-0.1,1.5)
    axes.flatten()[i].set_xlim(7, 11)

    axes.flatten()[i].set_yticks([0, 0.4, 0.8, 1.2])
    axes.flatten()[i].set_xticks([7, 8, 9, 10])



axes.flatten()[3].plot(-99, -99, '-', c='k', alpha=0.6, label="AGN hosts")
axes.flatten()[3].plot(-99, -99, '--', c='k', alpha=0.6, label="non-hosts")
axes.flatten()[3].legend(loc='lower left')
# --- scatter plot

#ax.scatter(x,y,s=3,c='k',alpha=0.1)

#fig.text(0.01, 0.55, r'$\rm log_{10}[T_{BB}\;/\;K]$', ha = 'left', va = 'center', rotation = 'vertical', fontsize=10)
fig.text(0.01, 0.55, r'$\rm \log_{10}[sSFR\;/\;Gyr^{-1}]$', ha = 'left', va = 'center', rotation = 'vertical', fontsize=10)
fig.text(0.45,0.05, r'$\rm log_{10}[M_{*}\;/\;M_{\odot}]$', ha = 'center', va = 'bottom', fontsize=10)

fig.savefig(f'figures/comparisons/sSFR.pdf', bbox_inches='tight')
fig.clf()

