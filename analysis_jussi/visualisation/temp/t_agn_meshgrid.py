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


mass_cut = 5.

def t_bb(m, m_dot):
    return 2.24*10**9*(m_dot)**(1/4)*(m)**(-1/2) #2.24*10**9*m_dot**(4)*m**(-8) #

def l_agn(m_dot, etta=0.1):
    m_dot = (m_dot*u.M_sun/u.yr).to(u.kg / u.s) # accretion rate in SI
    c = const.c #speed of light
    etta = etta
    l = (etta*m_dot*c**2).to(u.erg/u.s)
    return np.log10(l.value) # output in log10(erg/s)

def tau_dust(bh_los, dtm_ratio, lam=1500, kappa=0.0795, gamma=-1):
    # Defaults are for 1500Å (FUV), kappa_ISM from Vijayan et al. 2021, gamma from Wilkins et al. 2017
    tau = kappa * dtm_ratio * bh_los * (lam/5500)**gamma
    return tau

def attn(lum, bh_los, dtm_ratio, lam=1500, kappa=0.0795, gamma=-1):
    # lum = log10(L)
    return np.log10(10**lum*np.exp(-1*tau_dust(bh_los, dtm_ratio, lam, kappa, gamma)))

def M_edd(m_bh):
    m_bh = m_bh * u.solMass
    m_edd = (4*np.pi*const.G*m_bh*const.m_p)/(0.1*const.c*const.sigma_T)
    return (m_edd.to(u.solMass / u.yr)).value

cmap = mpl.cm.YlOrRd
norm = mpl.colors.Normalize(vmin=4, vmax=np.log10(1.5*10**6))

flares_dir = '../../../../data/simulations'

#fl = flares.flares(f'{flares_dir}/flares_old.hdf5', sim_type='FLARES') #_no_particlesed
fl = flares.flares(f'{flares_dir}/flares_no_particlesed.hdf5', sim_type='FLARES') #_no_particlesed
df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
halo = fl.halos
print(fl.tags) # print list of available snapshots
tags = fl.tags #This would be z=5

MBH = fl.load_dataset('BH_Mass', arr_type='Galaxy') # Black hole mass of galaxy
MDOT = fl.load_dataset('BH_Mdot', arr_type='Galaxy') # Black hole accretion rate
MS = fl.load_dataset('Mstar_30', arr_type='Galaxy') # Black hole accretion rate
LFUV = fl.load_dataset('FUV', arr_type=f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI/')
LBOL = fl.load_dataset('DustModelI', arr_type=f'Galaxy/BPASS_2.2.1/Chabrier300/Indices/Lbol/')
BHLOS = fl.load_dataset('BH_los', arr_type='Galaxy')
DTM = fl.load_dataset('DTM', arr_type='Galaxy')


df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
weights = np.array(df['weights'])


attn_out = {}


fig = plt.figure(figsize=(3,3))
left  = 0.1
bottom = 0.1
width = 0.75
height = 0.75
ax = fig.add_axes((left, bottom, width, height))


m_bh = np.linspace(5., 10, 100)
accr = np.linspace(-10, 1, 100)

X, Y = np.meshgrid(m_bh, accr)
C = np.log10(t_bb(10**X, 10**Y))

ax.pcolormesh(X, Y, C, cmap=cmap, norm=norm)

z_in = 10

ztag = z_in - 5

ax.plot(m_bh, np.log10(M_edd(10**m_bh)), ls='--', c='k', lw=1, alpha=0.7)

for tag in [np.flip(fl.tags)[ztag]]:

    z = np.flip(fl.zeds)[ztag]

    ws, x, y, mstar, lstar, los, dtm = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    for ii in range(len(halo)):
        s = (np.log10(MS[halo[ii]][tag])+10 > 8)&(np.log10(MBH[halo[ii]][tag]) +10 > 5.5)
        ws = np.append(ws, np.ones(np.shape(MBH[halo[ii]][tag][s]))*weights[ii])
        x = np.append(x, MDOT[halo[ii]][tag][s])
        y = np.append(y, MBH[halo[ii]][tag][s])
        mstar = np.append(mstar, np.log10(MS[halo[ii]][tag][s])+10)
        lstar = np.append(lstar, np.log10(LFUV[halo[ii]][tag][s]))
        los = np.append(los, BHLOS[halo[ii]][tag][s])
        dtm = np.append(dtm, DTM[halo[ii]][tag][s])

    h = 0.6777  # Hubble parameter


    # converting MBHacc units to M_sol/yr
    x *= h * 6.445909132449984E23  # g/s
    x = x/constants.M_sun.to('g').value  # convert to M_sol/s
    x *= units.yr.to('s')  # convert to M_sol/yr


    y *= 10**10


    b = t_bb(y, x)

    s_t = np.array(b) > 10**4

    ws = ws[s_t]

    print(z)
    print('Accr')
    print(f'min = {np.log10(np.min(x[s_t]))}, max = {np.log10(np.max(x[s_t]))}')
    print('M_BH')
    print(f'min = {np.log10(np.min(y[s_t]))}, max = {np.log10(np.max(y[s_t]))}')
    ax.scatter(np.log10(y[s_t]), np.log10(x[s_t]), c='k', s=2, alpha=0.5)


ax.text(0.8, 0.2, r'$\rm z={0:.0f}$'.format(z), fontsize=8, transform=ax.transAxes,
                           color='k', ha='right')


cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
cmapper.set_array([])

cax = fig.add_axes([width + left, bottom, 0.05, height])
bar = fig.colorbar(cmapper, cax=cax, orientation='vertical', format='%d')
bar.set_ticks([4, 5, 6])
cax.set_ylabel(r'$\rm log_{10}[T_{AGN} \; /\; K]$')

ax.set_xlim(5, 10)
ax.set_ylim(-10, 1)

ax.set_ylabel(r"$\rm log_{10}[\dot{M}_{BH} \; / \; M_{\odot} \;yr^{-1}]$")
ax.set_xlabel(r"$\rm log_{10}[M_{BH} \; / \; M_{\odot}]$")

fig.savefig(f'figures/t_bb_{z}.pdf', bbox_inches='tight')