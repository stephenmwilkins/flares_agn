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

mass_cut = 7.

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

Mdot = fl.load_dataset('BH_Mdot', arr_type='Galaxy') # Black hole mass of galaxy
Mstar = fl.load_dataset('Mstar_30', arr_type='Galaxy') # stellar mass of galaxy

Y = Mdot
X = Mstar



# --- FLARES is composed of 40 individual simulations. Each simulation has a statistical weight (basically the likelhood a similar region would appear in a volume of the real Universe). To get true distribution functions we need to weight each galaxy by the weight of the simulation.

# --- This bit of codes reads in the properties for all of the galaxies in every simulation and also produces a corresponding weight array. These can be combined later to get a true relationship.

df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
weights = np.array(df['weights'])

fig = plt.figure(figsize=(3,3))
left  = 0.2
bottom = 0.2
width = 0.75
height = 0.75
ax = fig.add_axes((left, bottom, width, height))


for i, tag in enumerate(fl.tags):
    z = fl.zeds[i]
    ws, x, y = np.array([]), np.array([]), np.array([])
    for ii in range(len(halo)):
        s = (np.log10(X[halo[ii]][tag])+10 > mass_cut)
        ws = np.append(ws, np.ones(np.shape(X[halo[ii]][tag][s]))*weights[ii])
        x = np.append(x, np.log10(X[halo[ii]][tag][s]))
        y = np.append(y, Y[halo[ii]][tag][s])

    h = 0.6777  # Hubble parameter


    # converting MBHacc units to M_sol/yr
    y *= h * 6.445909132449984E23  # g/s
    y = y/constants.M_sun.to('g').value  # convert to M_sol/s
    y *= units.yr.to('s')  # convert to M_sol/yr
    y = np.log10(y)

    x += 10 # units are 1E10 M_sol
    b = np.array([l_agn(q) for q in y])
    y = b

    # --- simply print the ranges of the quantities

    print(z)
    print(np.min(x),np.median(x),np.max(x))
    print(np.min(y),np.median(y),np.max(y))


    # -- this will calculate the weighted quantiles of the distribution
    quantiles = [0.84,0.50,0.16] # quantiles for range
    bins = np.arange(7, 11.5, 0.1) # x-coordinate bins, in this case stellar mass
    bincen = (bins[:-1]+bins[1:])/2.
    out = flares.binned_weighted_quantile(x,y,ws,bins,quantiles)

    # --- plot the median and quantiles for bins with >10 galaxies

    N, bin_edges = np.histogram(x, bins=bins)
    Ns = N > 10
    ax.plot(bincen, out[:, 1], c=cmap(norm(z)), ls=':')
    ax.plot(bincen[Ns], out[:, 1][Ns], c=cmap(norm(z)), label=rf'$\rm z={int(z)}$')
    ax.fill_between(bincen[Ns], out[:, 0][Ns], out[:, 2][Ns], color=cmap(norm(z)), alpha=0.2)



ax.legend(prop={'size': 6})

# --- scatter plot

#ax.scatter(x,y,s=3,c='k',alpha=0.1)

ax.set_xlabel(r'$\rm log_{10}[M_{\star}\;/\;M_{\odot}]$')
ax.set_ylabel(r'$\rm log_{10}[L_{AGN}\;/\;erg\,s^{-1}]$')
fig.savefig(f'figures/L_AGN.pdf', bbox_inches='tight')
fig.clf()
