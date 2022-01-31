import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt
import astropy.constants as const
import astropy.units as u

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




cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=5., vmax=10.)

flares_dir = '../../../data/simulations'

fl = flares.flares(f'{flares_dir}/flares_no_particlesed_old.hdf5', sim_type='FLARES')
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

import h5py
eagle = h5py.File(f'{flares_dir}/EAGLE_REF_sp_info.hdf5')

fig, axes = plt.subplots(2, 3, figsize = (6, 4), sharex = True, sharey=True)
fig.subplots_adjust(left=0.07, bottom=0.15, top=1.0, right=0.85, wspace=0.0, hspace=0.0)

m_bh = np.linspace(5., 10, 100)
accr = np.linspace(-10, 1, 100)

eagletags = np.flip(np.array(['002_z009p993', '003_z008p988', '004_z008p075', '005_z007p050', '006_z005p971', '008_z005p037']))

for i, tag in enumerate(np.flip(fl.tags)):
    z = np.flip(fl.zeds)[i]
    ws, x, y = np.array([]), np.array([]), np.array([])

    eagletag = eagletags[i]

    for ii in range(len(halo)):
        ws = np.append(ws, np.ones(np.shape(X[halo[ii]][tag]))*weights[ii])
        x = np.append(x, X[halo[ii]][tag])
        y = np.append(y, np.log10(Y[halo[ii]][tag]))

    eagle_mbh = np.log10(eagle[eagletag]['Galaxy']['BH_Mass'].value)+10
    eagle_maccr = eagle[eagletag]['Galaxy']['BH_Mdot'].value

    h = 0.6777  # Hubble parameter

    # converting MBHacc units to M_sol/yr
    x *= h * 6.445909132449984E23  # g/s
    x = x/const.M_sun.to('g').value  # convert to M_sol/s
    x *= u.yr.to('s')  # convert to M_sol/yr
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


    l_edd_agn = np.array([l_agn(maccr, etta=0.1, etta_r=0) for maccr in np.log10(M_edd(10**m_bh))])
    l_agn_eagle = np.array([l_agn(maccr, etta=0.1, etta_r=0) for maccr in np.log10(eagle_maccr)])
    l_agn_onoue = onoue['edd_rate']*np.array([10**l_agn(maccr, etta=0.1, etta_r=0) for maccr in np.log10(M_edd(onoue['mass']))])

    bins = np.arange(44, 47, 0.25)  #  x-coordinate bins, in this case stellar mass
    bincen = (bins[:-1] + bins[1:]) / 2.

    N_flares, binedges = np.histogram(x, bins=bins)
    N_eagle, binedges =  np.histogram(l_agn_eagle, bins=bins)

    print(f'FLARES z = {z}, N_BH = {sum(N_flares)}')
    print(f'EAGLE  z = {z}, N_BH = {sum(N_eagle)}')

    print('bin centres:')
    print(bincen)
    print('FLARES:')
    print(N_flares)
    print('EAGLE:')
    print(N_eagle)

    axes.flatten()[i].step(bincen, N_flares, c='k', where='mid')
    axes.flatten()[i].step(bincen, N_eagle, where='mid', c='teal', alpha=0.7)

    #axes.flatten()[i].plot(m_bh, l_edd_agn, 'k--', alpha=0.5, zorder=0)
    #axes.flatten()[i].scatter(y, x, c=cmap(norm(z)), alpha=0.7, s=4, zorder=5)
    #axes.flatten()[i].scatter(eagle_mbh, l_agn_eagle, c='teal', marker='D', alpha=1, s=4, zorder=10)
    #axes.flatten()[i].scatter(np.log10(onoue['mass']), np.log10(l_agn_onoue), c='teal', marker='*', alpha=1, s=4, zorder=15)

    axes.flatten()[i].text(0.8, 0.8, r'$\rm z={0:.0f}$'.format(z), fontsize=8, transform=axes.flatten()[i].transAxes, color=cmap(norm(z)))

    #axes.flatten()[i].set_xticks([6, 7, 8, 9])
    axes.flatten()[i].set_xticks([45,46,47])
    axes.flatten()[i].set_xlim(44, 47)
    axes.flatten()[i].set_ylim(0)
    #ax.legend(prop={'size': 6})

    # --- scatter plot

    #ax.scatter(x,y,s=3,c='k',alpha=0.1)

#axes.flatten()[i].scatter(-99, -99, c='teal', marker='*', alpha=1, s=4, label=onoue['label'])
axes.flatten()[i].step(-99, -99, c='k', alpha=1, label='FLARES')
axes.flatten()[i].step(-99, -99, c='teal', alpha=0.8, label='EAGLE REF')
axes.flatten()[i].legend(loc='lower right', prop={'size': 6})

fig.text(0.45, 0.05, r'$\rm log_{10}[L_{AGN}\;/\;erg\,s^{-1}]$', ha = 'center', va = 'bottom', fontsize=10)
fig.text(0.01,0.55, r'$\rm N_{AGN}$', ha = 'left', va = 'center', rotation = 'vertical', fontsize=10)

fig.savefig(f'figures/l_agn/L_AGN_hist.pdf', bbox_inches='tight')
fig.clf()
