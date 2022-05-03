import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt
import astropy.constants as const
import astropy.units as u

from uncertainties import unumpy

import flares

import FLARE.photom as photom
import FLARE.plt as fplt

mass_cut = 5.

def l_agn(m_dot, etta=0.1, etta_r=0.15):
    m_dot = 10**m_dot*1.98847*10**30/(365*24*60*60) # accretion rate in SI
    c = 299792458 #speed of light
    etta = etta*(1-etta_r)
    return np.log10(etta*m_dot*c**2*10**7) # output in log10(erg/s)

def M_edd(m_bh):
    m_bh = m_bh * u.solMass
    m_edd = (4*np.pi*const.G*m_bh*const.m_p)/(0.1*const.c*const.sigma_T)
    return (m_edd.to(u.solMass / u.yr)).value

def L_edd(m_bh):
    return 3*10**4*(const.L_sun.to(u.erg*u.s**(-1))).value*10**m_bh



# Observed quasars (MOVE SOMEWHERE LATER)

# Onoue
onoue = {'label': 'Onoue+2019'}
onoue['lum'] = np.array([8.96*10**45, 1.03*10**45, 6.77*10**45, 4.44*10**45, 4.38*10**45, 2.66*10**45])
onoue['mass'] = np.array([22.*10**8, 0.38*10**8, 6.3*10**8, 11*10**8, 7.1*10**8, 7.0*10**8])
onoue['edd_rate'] = np.array([0.16, 1.1, 0.43, 0.17, 0.24, 0.24])
onoue['err_edd_u'] = np.array([0.04, 0.5, 0.08, 0.04, 0.18, 0.06])
onoue['err_edd_l'] = np.array([0.02, 0.3, 0.05, 0.05, 0.08, 0.01])
onoue['err_mass_u'] = np.array([2*10**8, 0.1*10**8, 0.8*10**8, 3*10**8, 2.4*10**8, 1.4*10**8])
onoue['err_mass_l'] = np.array([6*10**8, 0.18*10**8, 1.2*10**8, 2*10**8, 5.2*10**8, 2.3*10**8])




cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=5., vmax=10.)

flares_dir = '../../../data/simulations'

fl = flares.flares(f'{flares_dir}/flares_no_particlesed.hdf5', sim_type='FLARES')
df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
halo = fl.halos
print(fl.tags) # print list of available snapshots
tags = fl.tags #This would be z=5

Mdot = fl.load_dataset('BH_Mdot', arr_type='Galaxy') # Black hole mass of galaxy
MBH = fl.load_dataset('BH_Mass', arr_type='Galaxy') # Black hole mass of galaxy

Y = MBH
X = Mdot



# --- FLARES is composed of 40 individual simulations. Each simulation has a statistical weight (basically the likelhood a similar region would appear in a volume of the real Universe). To get true distribution functions we need to weight each galaxy by the weight of the simulation.

# --- This bit of codes reads in the properties for all of the galaxies in every simulation and also produces a corresponding weight array. These can be combined later to get a true relationship.

df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
weights = np.array(df['weights'])

deltas = np.array(df['delta'])
deltas = np.log10(deltas+1)

import h5py
eagle = h5py.File(f'{flares_dir}/EAGLE_REF_sp_info.hdf5')




cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=-0.3, vmax=0.3)

m_bh = np.linspace(5., 10, 100)
accr = np.linspace(-10, 1, 100)

eagletags = np.flip(np.array(['002_z009p993', '003_z008p988', '004_z008p075', '005_z007p050', '006_z005p971', '008_z005p037']))



for i, tag in enumerate(np.flip(fl.tags)):
    fig = plt.figure(figsize=(3, 3))
    left = 0.1
    bottom = 0.1
    width = 0.75
    height = 0.75
    ax = fig.add_axes((left, bottom, width, height))

    z = np.flip(fl.zeds)[i]
    ws, dt, x, y = np.array([]), np.array([]), np.array([]), np.array([])

    eagletag = eagletags[i]

    for ii in range(len(halo)):
        ws = np.append(ws, np.ones(np.shape(X[halo[ii]][tag]))*weights[ii])
        dt = np.append(dt, np.ones(np.shape(X[halo[ii]][tag])) * deltas[ii])
        x = np.append(x, X[halo[ii]][tag])
        y = np.append(y, np.log10(Y[halo[ii]][tag]))

    eagle_mbh = np.log10(eagle[eagletag]['Galaxy']['BH_Mass'].value)+10
    eagle_maccr = eagle[eagletag]['Galaxy']['BH_Mdot'].value

    h = 0.6777  # Hubble parameter

    # converting MBHacc units to M_sol/yr
    x *= h #* 6.445909132449984E23  # g/s
    #x = x/const.M_sun.to('g').value  # convert to M_sol/s
    #x *= u.yr.to('s')  # convert to M_sol/yr
    x = np.log10(x)

    # converting MBHacc units to M_sol/yr
    eagle_maccr *= h * 6.445909132449984E23  # g/s
    eagle_maccr = eagle_maccr/const.M_sun.to('g').value  # convert to M_sol/s
    eagle_maccr *= u.yr.to('s')  # convert to M_sol/yr


    y += 10 # units are 1E10 M_sol
    b = np.array([l_agn(q, etta_r=0) for q in x])
    x = b

    # --- simply print the ranges of the quantities

    print(z)
    print(np.min(x),np.median(x),np.max(x))
    print(np.min(y),np.median(y),np.max(y))

    '''

    # -- this will calculate the weighted quantiles of the distribution
    quantiles = [0.84,0.50,0.16] # quantiles for range
    bins = np.arange(36, 49, 0.1) # x-coordinate bins, in this case stellar mass
    bincen = (bins[:-1]+bins[1:])/2.
    out = flares.binned_weighted_quantile(x,y,ws,bins,quantiles)

    # --- plot the median and quantiles for bins with >10 galaxies

    N, bin_edges = np.histogram(x, bins=bins)
    Ns = N > 10
    ax.plot(bincen, out[:, 1], c=cmap(norm(z)), ls=':')
    ax.plot(bincen[Ns], out[:, 1][Ns], c=cmap(norm(z)), label=rf'$\rm z={int(z)}$')
    ax.fill_between(bincen[Ns], out[:, 0][Ns], out[:, 2][Ns], color=cmap(norm(z)), alpha=0.2)

    '''
    bins = np.arange(5, 10, 0.5)  #  x-coordinate bins, in this case stellar mass
    bincen = (bins[:-1] + bins[1:]) / 2.

    N_flares, binedges = np.histogram(y, bins=bins)
    N_eagle, binedges =  np.histogram(eagle_mbh, bins=bins)

    l_edd_agn = np.array([l_agn(maccr, etta=0.1, etta_r=0) for maccr in np.log10(M_edd(10**m_bh))])
    l_agn_eagle = np.array([l_agn(maccr, etta=0.1, etta_r=0) for maccr in np.log10(eagle_maccr)])
    l_agn_onoue = onoue['edd_rate']*np.array([10**l_agn(maccr, etta=0.1, etta_r=0) for maccr in np.log10(M_edd(onoue['mass']))])

    print(np.log10(l_agn_onoue))

    print(f'FLARES z = {z}, N_BH = {sum(N_flares)}')
    print(f'EAGLE  z = {z}, N_BH = {sum(N_eagle)}')

    print('bin centres:')
    print(bincen)
    print('FLARES:')
    print(N_flares)
    print('EAGLE:')
    print(N_eagle)

    def log10err(err, val):
        return err/(np.log(10)*val)


    def calc(BH_Mass, f_Edd):

        L_Edd = (1.3e46) * (BH_Mass / 1e8)  # in erg/s
        L_BH = f_Edd * L_Edd

        return unumpy.log10(L_BH)

    ax.plot(m_bh, l_edd_agn, 'k--', alpha=0.5, zorder=-99)
    ax.scatter(y, x, c=cmap(norm(dt)), alpha=0.7, s=4, zorder=-50)
    #axes.flatten()[i].scatter(eagle_mbh, l_agn_eagle, c='teal', marker='D', alpha=1, s=4, zorder=10)
    #axes.flatten()[i].scatter(np.log10(onoue['mass']), np.log10(l_agn_onoue), c='firebrick', marker='*', alpha=1, s=10, zorder=15)


    ax.text(0.2, 0.8, r'$\rm z={0:.0f}$'.format(z), fontsize=8, transform=ax.transAxes, color='k')

    ax.set_xticks([6, 7, 8, 9])
    ax.set_yticks([45,46,47])
    ax.set_xlim(5.9,9.5)
    ax.set_ylim(44.5, 47.1)

    cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    cmapper.set_array([])

    cax = fig.add_axes([width + left, bottom, 0.05, height])
    bar = fig.colorbar(cmapper, cax=cax, orientation='vertical', format='%.1f')
    bar.set_ticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    cax.set_ylabel(r'$\rm log_{10}[\delta + 1]$')

    ax.set_xlabel(r'$\rm log_{10}[M_{BH}\;/\;M_{\odot}]$')
    ax.set_ylabel(r'$\rm log_{10}[L_{AGN}\;/\;erg\,s^{-1}]$')

    fig.savefig(f'figures/LAGN_MBH/LAGN_MBH_{z}.pdf', bbox_inches='tight')
    fig.clf()

