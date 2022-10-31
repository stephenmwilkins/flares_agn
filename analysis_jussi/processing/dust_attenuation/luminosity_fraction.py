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

bol_correction_ion = pickle.load(open('bolometric_correction_ion.p', 'rb'))
ratio_from_t_ion = interp1d(bol_correction_ion['AGN_T'], bol_correction_ion['ratio']['ionising'])

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

def lum_frac(agn, gal):
    return np.log10(agn/(agn+gal))

cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=5., vmax=10.)

flares_dir = '../../../../data/simulations'

#fl = flares.flares(f'{flares_dir}/flares_old.hdf5', sim_type='FLARES') #_no_particlesed
fl = flares.flares(f'{flares_dir}/flares_no_particlesed.hdf5', sim_type='FLARES') #_no_particlesed
df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
halo = fl.halos
print(fl.tags) # print list of available snapshots
tags = fl.tags

LION = pickle.load(open(f"{flares_dir}/ion_luminosity.p", 'rb'))
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

output = {}
output["bolometric"] = {}
output["uv_intrinsic"] = {}
output["uv_attenuated"] = {}
output["uv_agn_int_gal_attenuated"] = {}
output["ionising"] = {}

for i, tag in enumerate(np.flip(fl.tags)):

    z = np.flip(fl.zeds)[i]

    ws, x, y, mstar, lstar, los, dtm, lbol, lstar_int, lstar_ion = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
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
        lstar_ion = np.append(lstar_ion, np.log10(LION[halo[ii]][tag]["Intrinsic"][s]))

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
    agn_ionising = (ratio_from_t_ion(b[s_t]))*10**q

    y = np.log10(y /  ((const.c/(1500*u.AA).to(u.m)).to(u.Hz)).value)

    agn_intrinsic = 10**y[:]

    agn_dust = 10**attn(y, los[s_t], dtm[s_t])

    lstar_uv = 10**lstar[s_t]
    lstar_uv_int = 10**lstar_int[s_t]
    lstar_bolometric = 10**lbol[s_t]
    lstar_ion = 10**lstar_ion[s_t]

    obj_bolometric = lstar_bolometric + agn_bolometric
    obj_uv_intrinsic = lstar_uv + agn_intrinsic
    obj_uv_dust = lstar_uv + agn_dust

    bins = np.arange(25,32,0.25)
    binw = bins[1]-bins[0]
    b_c = bins[:-1]+binw/2

    bins2 = np.arange(35, 50, 0.25)
    binw2 = bins2[1] - bins2[0]
    b_c2 = bins2[:-1] + binw2 / 2

    bins3 = np.arange(35, 46, 0.25)
    binw3 = bins3[1] - bins3[0]
    b_c3 = bins3[:-1] + binw3 / 2

    N_agn_dusty, edges_dusty = np.histogram(np.log10(agn_dust), bins=bins)
    N_agn_intrinsic, edges_intrinsic = np.histogram(np.log10(agn_intrinsic), bins=bins)
    N_agn_bolometric, edges_bolometric = np.histogram(np.log10(agn_bolometric), bins=bins2)
    N_agn_ion, edges_ion = np.histogram(np.log10(agn_ionising), bins=bins3)

    N_gal_dusty, edges_dusty = np.histogram(np.log10(lstar_uv), bins=bins)
    N_gal_intrinsic, edges_intrinsic = np.histogram(np.log10(lstar_uv_int), bins=bins)
    N_gal_bolometric, edges_bolometric = np.histogram(np.log10(lstar_bolometric), bins=bins2)
    N_gal_ion, edges_ion = np.histogram(np.log10(lstar_ion), bins=bins3)

    threshold = 4
    Ns_agn_bolometric = (N_agn_bolometric > threshold)
    Ns_agn_intrinsic = (N_agn_intrinsic > threshold)
    Ns_agn_dusty = (N_agn_dusty > threshold)
    Ns_agn_ion = (N_agn_ion > threshold)

    Ns_gal_bolometric = (N_gal_bolometric > threshold)
    Ns_gal_intrinsic = (N_gal_intrinsic > threshold)
    Ns_gal_dusty = (N_gal_dusty > threshold)
    Ns_gal_ion = (N_gal_ion > threshold)

    frac_bolometric = lum_frac(agn_bolometric, lstar_bolometric)
    frac_uv_int = lum_frac(agn_intrinsic, lstar_uv_int)
    frac_uv_dust = lum_frac(agn_dust, lstar_uv)
    frac_agn_int_gal_attn = lum_frac(agn_intrinsic, lstar_uv)
    frac_ion = lum_frac(agn_ionising, lstar_ion)

    out_bolometric = flares.binned_weighted_quantile(np.log10(lstar_bolometric), frac_bolometric, ws,
                                                     bins2, [0.84, 0.50, 0.16])
    out_uv_int = flares.binned_weighted_quantile(np.log10(lstar_uv_int), frac_uv_int, ws, bins,
                                                 [0.84, 0.50, 0.16])
    out_uv_dust = flares.binned_weighted_quantile(np.log10(lstar_uv), frac_uv_dust, ws, bins,
                                                 [0.84, 0.50, 0.16])
    out_agn_int_gal_attn = flares.binned_weighted_quantile(np.log10(lstar_uv), frac_agn_int_gal_attn, ws, bins,
                                                  [0.84, 0.50, 0.16])
    out_ion = flares.binned_weighted_quantile(np.log10(lstar_ion), frac_ion, ws,
                                                     bins3, [0.84, 0.50, 0.16])

    s1 = (frac_bolometric >= 0.)
    s2 = (frac_uv_int >= 0.)
    s3 = (frac_uv_dust >= 0.)
    s4 = (frac_agn_int_gal_attn >= 0.)
    s5 = (frac_ion >= 0.)

    output["bolometric"][str(z)] = (b_c2, out_bolometric, np.log10(lstar_bolometric[s1]), frac_bolometric[s1], Ns_gal_bolometric)
    output["uv_intrinsic"][str(z)] = (b_c, out_uv_int, np.log10(lstar_uv_int[s2]), frac_uv_int[s2], Ns_gal_intrinsic)
    output["uv_attenuated"][str(z)] = (b_c, out_uv_dust, np.log10(lstar_uv[s3]), frac_uv_dust[s3], Ns_gal_dusty)
    output["uv_agn_int_gal_attenuated"][str(z)] = (b_c, out_agn_int_gal_attn, np.log10(lstar_uv[s4]), frac_agn_int_gal_attn[s4], Ns_gal_dusty)
    output["ionising"][str(z)] = (b_c3, out_ion, lstar_ion[s5], frac_ion[s5], Ns_gal_ion)

for key in output.keys():

    fig = plt.figure(figsize=(3, 3))
    ax1 = fig.add_axes([0., 0., 1., 1.])

    for z in [5.0, 6.0, 7.0]:
        Ns = output[key][str(z)][4]
        ax1.plot(output[key][str(z)][0], output[key][str(z)][1][:, 1], ls="--", color=cmap(norm(z)), alpha=0.5)
        ax1.plot(output[key][str(z)][0][Ns], output[key][str(z)][1][:, 1][Ns], color=cmap(norm(z)), label=fr"$\rm z = {z:.0f}$")
        ax1.fill_between(output[key][str(z)][0][Ns], output[key][str(z)][1][:, 0][Ns], output[key][str(z)][1][:, 2][Ns], color=cmap(norm(z)), alpha=0.15)
        #ax1.scatter(output[key][str(z)][2], output[key][str(z)][3], color=cmap(norm(z)))
        print(z, key)
        print(10**output[key][str(z)][1][:, 1][Ns])
        print("max = ", max(10**output[key][str(z)][1][:, 1][Ns]))
        print("mean = ", np.mean(10**output[key][str(z)][1][:, 1][Ns]))
        s_lion = output[key][str(z)][0][Ns] > 42.8
        print("mean L_ion > 43", np.mean(10**output[key][str(z)][1][:, 1][Ns][s_lion]))

    if key == "bolometric":
        ax1.set_xlabel(r"$\rm log_{10}[L^{*}_{bol} \, / \, erg \, s^{-1}]$")
        ax1.set_ylabel(r"$\rm log_{10}[L^{AGN}_{bol} \, / \, (L^{AGN}_{bol} + L^{*}_{bol})]$")
        ax1.set_xticks([43.5, 44, 44.5, 45, 45.5, 46])
        ax1.set_xlim(43.3, 46.2)

    elif key == "ionising":
        ax1.set_xlabel(r"$\rm log_{10}[L^{*}_{ion} \, / \, erg \, s^{-1}]$")
        ax1.set_ylabel(r"$\rm log_{10}[L^{AGN}_{ion} \, / \, (L^{AGN}_{ion} + L^{*}_{ion})]$")
        ax1.set_xticks([41, 42, 43, 44])
        ax1.set_xlim(40.5, 44.5)

    elif key == "uv_intrinsic":
        ax1.set_xlabel(r"$\rm log_{10}[L^{*}_{FUV} \, / \, erg \, s^{-1} \, Hz^{-1}]$")
        ax1.set_ylabel(r"$\rm log_{10}[L^{AGN}_{FUV} \, / \, (L^{AGN}_{FUV} + L^{*}_{FUV})]$")
        ax1.set_xticks([28, 28.5, 29, 29.5, 30, 30.5, 31])
        ax1.set_xlim(27.5, 31.5)

    else:
        ax1.set_xlabel(r"$\rm log_{10}[L^{*}_{FUV} \, / \, erg \, s^{-1} \, Hz^{-1}]$")
        ax1.set_ylabel(r"$\rm log_{10}[L^{AGN}_{FUV} \, / \, (L^{AGN}_{FUV} + L^{*}_{FUV})]$")
        ax1.set_xticks([28, 28.5, 29, 29.5, 30, 30.])
        ax1.set_xlim(27.5, 30.5)

    ax1.axhline(0, alpha=0.8, c='k', ls='--', linewidth=1)

    ax1.set_ylim(-8, 0.5)

    if key == "bolometric":
        ax1.legend(loc="center left")
    if key == "uv_intrinsic":
        ax1.legend(loc="center left")
    if key == "uv_attenuated":
        ax1.legend(loc="center left")
    if key == "uv_agn_int_gal_attenuated":
        ax1.legend(loc="center left")
    if key == "ionising":
        ax1.legend(loc="center left")

    plt.savefig(f'figures/lum_frac/{key}.pdf', bbox_inches='tight')
