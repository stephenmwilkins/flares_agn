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


cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=5., vmax=10.)

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

#LUM = fl.load_dataset('DustModelI', arr_type=f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity')

Y = MBH
X = MDOT

df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
weights = np.array(df['weights'])

output = {}

fig, axes = plt.subplots(2, 3, figsize = (6, 4), sharex = True, sharey=True)
fig.subplots_adjust(left=0.07, bottom=0.15, top=1.0, right=0.85, wspace=0.0, hspace=0.0)

h = 0.6777  # Hubble parameter

for ii in range(len(halo)):

    output[halo[ii]] = {}

    for i, tag in enumerate(np.flip(fl.tags)):

        ws = np.ones(np.shape(X[halo[ii]][tag]))*weights[ii]
        x = X[halo[ii]][tag]
        y = Y[halo[ii]][tag]
        mstar = np.log10(MS[halo[ii]][tag])+10
        lstar = np.log10(LFUV[halo[ii]][tag])
        los = BHLOS[halo[ii]][tag]
        dtm = DTM[halo[ii]][tag]

        # converting MBHacc units to M_sol/yr
        x *= h * 6.445909132449984E23  # g/s
        x = x/constants.M_sun.to('g').value  # convert to M_sol/s
        x *= units.yr.to('s')  # convert to M_sol/yr


        y *= 10**10


        b = t_bb(y, x)

        s_t = np.array(b) > 10**4

        ws = ws[s_t]

        q = np.array([l_agn(g, etta=0.1) for g in x[s_t]])


        y = (ratio_from_t(b[s_t]))*10**q
        #y = (1/4.4) * 10 ** q

        y_intrinsic = np.log10(y /  ((const.c/(1500*u.AA).to(u.m)).to(u.Hz)).value)

        y = attn(y_intrinsic, los[s_t], dtm[s_t])

        #yy = np.log10(10**lstar[s_t] + 10**y)

        output[halo[ii]][tag] = {}
        output[halo[ii]][tag]['mask'] = s_t
        output[halo[ii]][tag]['log10L'] = {}
        output[halo[ii]][tag]['log10L']['bolometric'] = q
        output[halo[ii]][tag]['log10L']['uv_intrinsic'] = y_intrinsic
        output[halo[ii]][tag]['log10L']['uv_dust'] = y

pickle.dump(output, open('l_agn_dust.p', 'wb'))