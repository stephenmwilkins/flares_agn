import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt

import flares

import FLARE.photom as photom
import FLARE.plt as fplt

mass_cut = 5.

def t_bb(m, m_dot):
    return 2.24*10**9*m_dot**(1/4)*m**(-1/2)

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
MDOT = fl.load_dataset('BH_Mdot', arr_type='Galaxy') # Black hole accretion rate

Y = MBH
X = MDOT

df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
weights = np.array(df['weights'])

fig, axes = plt.subplots(2, 3, figsize = (6, 4), sharex = True, sharey=True)
fig.subplots_adjust(left=0.07, bottom=0.15, top=1.0, right=0.85, wspace=0.0, hspace=0.0)


for i, tag in enumerate(np.flip(fl.tags)):
    z = np.flip(fl.zeds)[i]
    ws, x, y = np.array([]), np.array([]), np.array([])
    for ii in range(len(halo)):
        s = (np.log10(Y[halo[ii]][tag])+10 > mass_cut)
        ws = np.append(ws, np.ones(np.shape(X[halo[ii]][tag][s]))*weights[ii])
        x = np.append(x, X[halo[ii]][tag][s])
        y = np.append(y, Y[halo[ii]][tag][s])

    b = t_bb(y, x)
    x = b

    y = np.log10(y)
    y += 10  # units are 1E10 M_sol

    # --- simply print the ranges of the quantities

    print(z)
    print(np.min(x),np.median(x),np.max(x))
    print(np.min(y),np.median(y),np.max(y))

    axes.flatten()[i].scatter(x, y, c=cmap(norm(z)), alpha=0.5, s=5)

    axes.flatten()[i].text(0.7, 0.9, r'$\rm z={0:.0f}$'.format(z), fontsize=8, transform=axes.flatten()[i].transAxes,
                           color=cmap(norm(z)))

# --- scatter plot

#ax.scatter(x,y,s=3,c='k',alpha=0.1)

fig.text(0.01, 0.55, r'$\rm log_{10}[M_{BH}\;/\;M_{\odot}]$', ha = 'left', va = 'center', rotation = 'vertical', fontsize=10)
fig.text(0.45,0.05, r'$\rm T_{BB}\;/\;K$', ha = 'center', va = 'bottom', fontsize=10)

fig.savefig(f'figures/t_agn/T_AGN_M_BH_zeds_reversed.pdf', bbox_inches='tight')
fig.clf()

