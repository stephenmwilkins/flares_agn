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
LFUV_INT = fl.load_dataset('FUV', arr_type=f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity/Intrinsic/')

#LUM = fl.load_dataset('DustModelI', arr_type=f'Galaxy/BPASS_2.2.1/Chabrier300/Luminosity')

Y = MBH
X = MDOT

df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
weights = np.array(df['weights'])

attn_out = {}

density = {}


for i, tag in enumerate(np.flip(fl.tags)):

    z = np.flip(fl.zeds)[i]

    ws, x, y, mstar, lstar, los, dtm, lbol, lstar_int = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
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
        lstar_int = np.append(lstar_int, np.log10(LFUV_INT[halo[ii]][tag][s]))

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

    agn_bolometric = 10**q[:]

    x = np.array(mstar)[s_t]


    y = (ratio_from_t(b[s_t]))*10**q
    #y = (1/4.4) * 10 ** q

    y = np.log10(y /  ((const.c/(1500*u.AA).to(u.m)).to(u.Hz)).value)

    agn_intrinsic = 10**y[:]
    agn_dust = 10**attn(y, los[s_t], dtm[s_t])

    lstar_uv = 10**lstar[s_t]
    lstar_uv_int = 10**lstar_int[s_t]
    lstar_bolometric = 10**lbol[s_t]


    bol_lim = 43.

    s_bol_agn = (np.log10(agn_bolometric) > bol_lim)
    s_bol_gal = (np.log10(lstar_bolometric) > bol_lim)

    ws_agn = ws[s_bol_agn]
    ws_gal = ws[s_bol_gal]

    obj_bolometric = np.sort(np.concatenate((np.log10(lstar_bolometric[s_bol_gal]), np.log10(agn_bolometric[s_bol_agn]))))
    obj_uv_intrinsic = np.sort(np.log10(lstar_uv + agn_intrinsic))
    obj_uv_dust = np.sort(np.log10(lstar_uv + agn_dust))

    agn_bolometric = np.sort(np.log10(agn_bolometric[s_bol_agn]))
    agn_intrinsic = np.sort( np.log10(agn_intrinsic))
    agn_dust = np.sort(np.log10(agn_dust))

    gal_bolometric = np.sort(np.log10(lstar_bolometric[s_bol_gal]))
    gal_intrinsic = np.sort(np.log10(lstar_uv_int))
    gal_dust = np.sort(np.log10(lstar_uv))


    print(f"z = {z}, lum bolometric obj:\nmin = {min(obj_bolometric):.4f}, max = {max(obj_bolometric):.4f}")
    print(f"z = {z}, lum uv intrins obj:\nmin = {min(obj_uv_intrinsic):.4f}, max = {max(obj_uv_intrinsic):.4f}")
    print(f"z = {z}, lum uv dusty   obj:\nmin = {min(obj_uv_dust):.4f}, max = {max(obj_uv_dust):.4f}\n")

    print(f"z = {z}, lum bolometric agn:\nmin = {min(agn_bolometric):.4f}, max = {max(agn_bolometric):.4f}")
    print(f"z = {z}, lum uv intrins agn:\nmin = {min(agn_intrinsic):.4f}, max = {max(agn_intrinsic):.4f}")
    print(f"z = {z}, lum uv dusty   agn:\nmin = {min(agn_dust):.4f}, max = {max(agn_dust):.4f}\n")

    print(f"z = {z}, lum bolometric gal:\nmin = {min(gal_bolometric):.4f}, max = {max(gal_bolometric):.4f}")
    print(f"z = {z}, lum uv intrins gal:\nmin = {min(gal_intrinsic):.4f}, max = {max(gal_intrinsic):.4f}")
    print(f"z = {z}, lum uv dusty   gal:\nmin = {min(gal_dust):.4f}, max = {max(gal_dust):.4f}\n")

    bins = np.linspace(25,32,100)
    binw = bins[1]-bins[0]
    b_c = bins[:-1]+binw/2

    bins2 = np.linspace(bol_lim, 46, 100)
    binw2 = bins2[1] - bins2[0]
    b_c2 = bins2[:-1] + binw2 / 2

    def cumsum_(x, bins, ws):
        N, edges = np.histogram(x, bins=bins, weights=ws)
        sum_ = np.cumsum(np.sort(N))
        return sum_/sum_[-1]

    #obj_cum_bol = cumsum_(obj_bolometric, bins2)
    obj_cum_int = cumsum_(obj_uv_intrinsic, bins, ws)
    obj_cum_dust = cumsum_(obj_uv_dust, bins, ws)

    agn_cum_bol = cumsum_(agn_bolometric, bins2, ws_agn)
    agn_cum_int = cumsum_(agn_intrinsic, bins, ws)
    agn_cum_dust = cumsum_(agn_dust, bins, ws)

    gal_cum_bol = cumsum_(gal_bolometric, bins2, ws_gal)
    gal_cum_int = cumsum_(gal_intrinsic, bins, ws)
    gal_cum_dust = cumsum_(gal_dust, bins, ws)

    density[str(z)] = {}

    density[str(z)]["obj"] = {}
    density[str(z)]["gal"] = {}
    density[str(z)]["agn"] = {}

    density[str(z)]["obj"]["bolometric"] = (b_c2, (gal_cum_bol+agn_cum_bol)/2)
    density[str(z)]["obj"]["uv_intrinsic"] = (b_c, obj_cum_int)
    density[str(z)]["obj"]["uv_attenuated"] = (b_c, obj_cum_dust)

    density[str(z)]["agn"]["bolometric"] = (b_c2, agn_cum_bol)
    density[str(z)]["agn"]["uv_intrinsic"] = (b_c, agn_cum_int)
    density[str(z)]["agn"]["uv_attenuated"] = (b_c, agn_cum_dust)

    density[str(z)]["gal"]["bolometric"] = (b_c2, gal_cum_bol)
    density[str(z)]["gal"]["uv_intrinsic"] = (b_c, gal_cum_int)
    density[str(z)]["gal"]["uv_attenuated"] = (b_c, gal_cum_dust)

fig = plt.figure(figsize=(3,3))
ax1 = fig.add_axes( [0., 0., 1., 1.] )
ax1.plot(density["5.0"]["obj"]["bolometric"][0], density["5.0"]["obj"]["bolometric"][1], 'k-', label=fr"Stellar+AGN, bolometric")
ax1.plot(density["5.0"]["gal"]["bolometric"][0], density["5.0"]["gal"]["bolometric"][1], 'k--', label=fr"Stellar, bolometric")
ax1.plot(density["5.0"]["agn"]["bolometric"][0], density["5.0"]["agn"]["bolometric"][1], 'k:', label=fr"AGN, bolometric")
ax1.set_xlim(bol_lim, 46)
ax1.legend(loc="best")
ax1.set_xlabel(r"$\rm log_{10}[L_{bol} \, / \, erg \, s^{-1}]$")
ax1.set_ylabel(fr"$\rm f(L_{{bol}} \, / \, 10^{{{bol_lim:.0f}}} \, erg \, s^{{-1}} )$")

plt.savefig('figures/lum_cumulative_bolometric.pdf', bbox_inches='tight')
