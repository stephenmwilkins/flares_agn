import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import flares

import sys

sys.path.insert(1, '../../../../data/literature')

from agn_obs import literature

import astropy.constants as constants
import astropy.units as units

import FLARE.photom as photom
import FLARE.plt as fplt

from scipy.integrate import simps, trapz, quad

import astropy.constants as const
import astropy.units as u

from scipy.interpolate import interp1d

import _pickle as pickle

bol_correction = pickle.load(open('bolometric_correction.p', 'rb'))
ratio_from_t = interp1d(bol_correction['AGN_T'], bol_correction['ratio']['FUV'])


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

def ax_M_to_lum(x):
    return np.log10(photom.M_to_lum(x))

def ax_lum_to_M(x):
    return photom.lum_to_M(10**x)

def ax_y_M_to_lum(x):
    return np.log10(10**x/0.4)

def ax_y_lum_to_M(x):
    return np.log10(10**x*0.4)

def lum_density(lum, phi):
    return trapz(lum*phi, lum)

cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=5., vmax=10.)

flares_dir = '../../../../data/simulations'

#fl = flares.flares(f'{flares_dir}/flares_old.hdf5', sim_type='FLARES') #_no_particlesed
fl = flares.flares(f'{flares_dir}/flares_no_particlesed.hdf5', sim_type='FLARES') #_no_particlesed
df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
halo = fl.halos
print(fl.tags) # print list of available snapshots
tags = fl.tags

MBH = fl.load_dataset('BH_Mass', arr_type='Galaxy') # Black hole mass of galaxy
MDOT = fl.load_dataset('BH_Mdot', arr_type='Galaxy') # Black hole accretion rate
MS = fl.load_dataset('Mstar', arr_type='Galaxy') # Black hole accretion rate
LFUV = fl.load_dataset('FUV', arr_type=f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/DustModelI/')
LBOL = fl.load_dataset('DustModelI', arr_type=f'Galaxy/BPASS_2.2.1/Chabrier300/Indices/Lbol/')
BHLOS = pickle.load(open(f'{flares_dir}/bh_los.p', 'rb')) #fl.load_dataset('BH_los', arr_type='Galaxy')
DTM = fl.load_dataset('DTM', arr_type='Galaxy')

#LUM = fl.load_dataset('DustModelI', arr_type=f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity')

Y = MBH
X = MDOT

df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
weights = np.array(df['weights'])

attn_out = {}

density = {}
density["bolometric"] = np.array([])
density["uv_intrinsic"] = np.array([])
density["uv_attenuated"] = np.array([])

for i, tag in enumerate(np.flip(fl.tags)):

    z = np.flip(fl.zeds)[i]

    ws, x, y, mstar, lstar, los, dtm, lbol = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    for ii in range(len(halo)):
        s = (np.log10(MS[halo[ii]][tag])+10 > 8)
        ws = np.append(ws, np.ones(np.shape(X[halo[ii]][tag][s]))*weights[ii])
        x = np.append(x, X[halo[ii]][tag][s])
        y = np.append(y, Y[halo[ii]][tag][s])
        mstar = np.append(mstar, np.log10(MS[halo[ii]][tag][s])+10)
        lstar = np.append(lstar, np.log10(LFUV[halo[ii]][tag][s]))
        los = np.append(los, BHLOS[halo[ii]][tag][s])
        dtm = np.append(dtm, DTM[halo[ii]][tag][s])
        lbol = np.append(lbol, np.log10(LBOL[halo[ii]][tag][s]))

    h = 0.6777  # Hubble parameter

    x *= h

    # converting MBHacc units to M_sol/yr
    #x *= h * 6.445909132449984E23  # g/s
    #x = x/constants.M_sun.to('g').value  # convert to M_sol/s
    #x *= units.yr.to('s')  # convert to M_sol/yr


    y *= 10**10


    b = t_bb(y, x)

    s_t = np.array(b) > 10**4

    ws = ws[s_t]

    q = np.array([l_agn(g, etta=0.1) for g in x[s_t]])

    x = np.array(mstar)[s_t]


    y = (ratio_from_t(b[s_t]))*10**q
    #y = (1/4.4) * 10 ** q

    y = np.log10(y /  ((const.c/(1500*u.AA).to(u.m)).to(u.Hz)).value)

    y_intrinsic = y[:]

    y = attn(y, los[s_t], dtm[s_t])

    attn_out[str(z)] = {'attn': (attn(0, los[s_t], dtm[s_t])), 'mstar': mstar[s_t], 'ws': ws, 'tau': tau_dust(los[s_t], dtm[s_t])}

    lstar_uv = lstar[s_t]
    lstar_bolometric = lbol[s_t]

    obj_bolometric = np.log10(10**lstar_bolometric + 10**q)
    obj_uv_intrinsic = np.log10(10**lstar_uv + 10**y_intrinsic)
    obj_uv_dust = np.log10(10**lstar_uv + 10**y)

    binw = 1
    bins = np.arange(25,32,binw)
    b_c = bins[:-1]+binw/2

    binw2 = 1
    bins2 = np.arange(35, 50, binw2)
    b_c2 = bins2[:-1] + binw2 / 2


    N_weighted_dusty, edges_dusty = np.histogram(obj_uv_dust, bins = bins, weights = ws)

    N_weighted_intrinsic, edges_intrinsic = np.histogram(obj_uv_intrinsic, bins=bins, weights=ws)

    N_weighted_bolometric, edges_bolometric = np.histogram(obj_bolometric, bins=bins2, weights=ws)

    h = 0.6777
    vol = (4/3)*np.pi*(14/h)**3

    phi_bolometric = N_weighted_bolometric / (binw2*vol)
    phi_intrinsic = N_weighted_intrinsic / (binw * vol)
    phi_dusty = N_weighted_dusty / (binw * vol)

    density["bolometric"] = np.append(density["bolometric"], lum_density(b_c2, phi_bolometric))
    density["uv_intrinsic"] = np.append(density["uv_intrinsic"], lum_density(b_c, phi_intrinsic))
    density["uv_attenuated"] = np.append(density["uv_attenuated"], lum_density(b_c, phi_dusty))

print(density["bolometric"])

plt.figure()
plt.plot(np.flip(fl.zeds), np.log10(density["bolometric"]))
plt.plot(np.flip(fl.zeds), np.log10(density["uv_intrinsic"]))
plt.plot(np.flip(fl.zeds), np.log10(density["uv_attenuated"]), '--')
plt.show()