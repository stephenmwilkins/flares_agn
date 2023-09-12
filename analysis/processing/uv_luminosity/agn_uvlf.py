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

from scipy.interpolate import interp1d


spectrum_index = 0

spectrum_parts = {'uv':[1400, 1600], 'optical':[4000, 8000]}
spectrum_part = list(spectrum_parts.keys())[spectrum_index]
lims = spectrum_parts[spectrum_part]

import astropy.units as u
import astropy.constants as c
lower_lim = ((c.c/(((1 * u.Ry).to(u.J))/c.h)).to(u.AA)).value
upper_lim = ((c.c/((((7.354e6) * u.Ry).to(u.J))/c.h)).to(u.AA)).value

#uvlims = [upper_lim, lower_lim] # limits Ross used (trying to see what he actually did)

cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin = np.log10(1e4), vmax = np.log10(1.5e6))

output_dir_bol = '../cloudy/output/linear/total'

AGN_T = np.linspace(10000, 1.5e6, 100)  # range of AGN temperatures to model

y = np.array([])
lbols = np.array([])
luvs = np.array([])

for i in range(len(AGN_T)):
    lam_bol, incident_bol, transmitted_bol, nebular_bol, total_bol, linecont_bol = np.loadtxt(f'{output_dir_bol}/{i+1}.cont', delimiter='\t',
                                                                      usecols=(0, 1, 2, 3, 4, 8)).T

    s = (lam_bol > lims[0])&(lam_bol<lims[1])

    lbols = np.append(lbols, trapz(np.flip(total_bol/lam_bol), np.flip(lam_bol)))
    luvs = np.append(luvs, trapz(np.flip(total_bol[s]/lam_bol[s]), np.flip(lam_bol[s])))

    y = np.append(y, simps(total_bol[s]/lam_bol[s], lam_bol[s])/simps(total_bol/lam_bol, lam_bol))

print(y)

ratio_from_t = interp1d(AGN_T, y)

'''
cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
cmapper.set_array([])

cax = fig.add_axes([width+left, bottom, 0.05, height])
bar = fig.colorbar(cmapper, cax=cax, orientation='vertical', format='%d')
bar.set_ticks([4, 5, 6])
cax.set_ylabel(r'$\rm log_{10}[T_{AGN} \; /\; K]$')
'''


print('L_bol, AGN')
print(lbols)
print('L_uv, AGN')
print(luvs)

print(min(lbols), np.median(lbols), max(lbols))





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

flares_dir = '../../../../data/simulations'

fl = flares.flares(f'{flares_dir}/flares.hdf5', sim_type='FLARES')
df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
halo = fl.halos
print(fl.tags) # print list of available snapshots
tags = fl.tags #This would be z=5

MBH = fl.load_dataset('BH_Mass', arr_type='Galaxy') # Black hole mass of galaxy
MDOT = fl.load_dataset('BH_Mdot', arr_type='Galaxy') # Black hole accretion rate
MS = fl.load_dataset('Mstar_30', arr_type='Galaxy') # Black hole accretion rate

#LUM = fl.load_dataset('DustModelI', arr_type=f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity')

Y = MBH
X = MDOT

df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
weights = np.array(df['weights'])

fig = plt.figure(figsize=(3,3))
left  = 0.2
bottom = 0.2
width = 0.75
height = 0.75
ax = fig.add_axes((left, bottom, width, height))


for i, tag in enumerate(np.flip(fl.tags)):
    z = np.flip(fl.zeds)[i]
    ws, x, y, mstar = np.array([]), np.array([]), np.array([]), np.array([])
    for ii in range(len(halo)):
        s = (np.log10(MS[halo[ii]][tag])+10 > 8)
        ws = np.append(ws, np.ones(np.shape(X[halo[ii]][tag][s]))*weights[ii])
        x = np.append(x, X[halo[ii]][tag][s])
        y = np.append(y, Y[halo[ii]][tag][s])
        mstar = np.append(mstar, np.log10(MS[halo[ii]][tag][s])+10)

    h = 0.6777  # Hubble parameter


    # converting MBHacc units to M_sol/yr
    x *= h * 6.445909132449984E23  # g/s
    x = x/constants.M_sun.to('g').value  # convert to M_sol/s
    x *= units.yr.to('s')  # convert to M_sol/yr


    y *= 10**10


    b = t_bb(y, x)

    s_t = np.array(b) > 10**4

    ws = ws[s_t]

    x = np.log10(x[s_t])
    q = np.array([l_agn(g) for g in x])

    x = np.array(mstar)[s_t]
    #y = np.log10(ratio_from_t(b[s_t])*10**q)
    y = q

    # --- simply print the ranges of the quantities

    print(z)
    print(np.min(x),np.median(x),np.max(x))
    print(np.min(y),np.median(y),np.max(y))

    '''
    # -- this will calculate the weighted quantiles of the distribution
    quantiles = [0.84,0.50,0.16] # quantiles for range
    bins = np.arange(30,49, 0.1) # x-coordinate bins
    bincen = (bins[:-1]+bins[1:])/2.
    out = flares.binned_weighted_quantile(x,y,ws,bins,quantiles)

    # --- plot the median and quantiles for bins with >10 galaxies

    N, bin_edges = np.histogram(x, bins=bins)
    Ns = N > 10
    axes.flatten()[i].plot(bincen, out[:, 1]/10**6, c=cmap(norm(z)), ls=':')
    axes.flatten()[i].plot(bincen[Ns], out[:, 1][Ns]/10**6, c=cmap(norm(z)), label=rf'$\rm z={int(z)}$')
    axes.flatten()[i].fill_between(bincen[Ns], out[:, 0][Ns]/10**6, out[:, 2][Ns]/10**6, color=cmap(norm(z)), alpha=0.2)

    '''

    binw = 0.5
    bins = np.arange(30,47,binw)
    b_c = bins[:-1]+binw/2

    N_weighted_bol, edges_bol = np.histogram(q, bins = bins, weights = ws)

    N_weighted, edges = np.histogram(y, bins = bins, weights = ws)

    vol = h = 0.6777
    vol = (4/3)*np.pi*(14/h)**3

    phi_bol = N_weighted_bol/(binw*vol)
    phi = N_weighted/(binw*vol)

    #ax.plot(bins[:-1] + binw / 2, np.log10(phi_bol), '--', c=cmap(norm(z)), label=rf'$\rm z={int(z)}$', alpha=0.6)
    ax.plot(bins[:-1] + binw / 2, np.log10(phi), c=cmap(norm(z)), label = rf'$\rm z={int(z)}$')

# --- scatter plot

#ax.scatter(x,y,s=3,c='k',alpha=0.1)

#fig.text(0.01, 0.55, r'$\rm log_{10}[T_{BB}\;/\;K]$', ha = 'left', va = 'center', rotation = 'vertical', fontsize=10)
#fig.text(0.01, 0.55, r'$\rm log_{10}[L_{AGN, UV}\;/\;erg\,s^{-1}]$', ha = 'left', va = 'center', rotation = 'vertical', fontsize=10)
#fig.text(0.45,0.05, r'$\rm log_{10}[L_{AGN, bol}\;/\;erg\,s^{-1}]$', ha = 'center', va = 'bottom', fontsize=10)

ax.legend(prop={'size': 6})

#ax.set_xlim(5.5, 9.5)
ax.set_ylim(-8)

ax.set_xlabel(r'$\rm log_{10}[L_{AGN, bol}\;/\;erg\,s^{-1}]$')
ax.set_ylabel(r'$\rm log_{10}[\phi\;/\;Mpc^{-3}\, dex^{-1}]$')

fig.savefig(f'figures/agn_bollf.pdf', bbox_inches='tight')
fig.clf()
