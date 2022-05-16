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


# Literature quasar LFs
literature = {}

# Niida et al. 2020
literature['Niida+2020'] = {}
literature['Niida+2020']['HSC'] = {}
literature['Niida+2020']['HSC']['M'] = np.array([-22.57, -23.07, -23.57, -24.07, -24.57, -25.07, -25.57, -26.07, -26.57, -27.07])
literature['Niida+2020']['HSC']['phi'] = 10**(-8)*np.array([68.9, 23.1, 12.5, 10.7, 5.39, 7.78, 2.98, 1.80, 0.60, 1.20])
literature['Niida+2020']['HSC']['phi_lo'] = 10**(-8)*np.array([68.9-6.9, 23.1-3.7, 12.5-2.7, 10.7-2.5, 5.39-1.76, 7.78-2.13, 2.98-1.29, 1.80-0.98, 0.60-0.50, 1.20-0.78])
literature['Niida+2020']['HSC']['phi_hi'] = 10**(-8)*np.array([68.9+6.9, 23.1+4.4, 12.5+3.4, 10.7+3.2, 5.39+2.46, 7.78+2.81, 2.98+2.01, 1.80+1.75, 0.60+1.39, 1.20+1.58])

literature['Harikane+2021'] = {}
literature['Harikane+2021']['Galaxy+AGN'] = {}
literature['Harikane+2021']['Galaxy+AGN']['z5'] = {}
literature['Harikane+2021']['Galaxy+AGN']['z5']['M'] = np.array([
    -20.30, -20.55, -20.80, -21.05, -21.30, -21.55, -21.80, -22.05, -22.30,
    -22.55, -22.80, -23.05, -23.42, -23.92, -24.42, -24.92, -25.42
    ])

"""
z=4 M_UV
    -19.99, -20.09, -20.19, -20.29, -20.39, -20.49, -20.59, -20.69, -20.79, -20.89, -20.99, -21.09, -21.19, -21.29, 
    -21.39, -21.49, -21.59, -21.69, -21.79, -21.89, -21.99, -22.09, -22.19, -22.29, -22.44, -22.64, -22.89, -23.19,
    -23.49, -23.79, -24.09, -24.69, -24.99, -25.29, -25.59, -25.89, -26.19, -26.49, -26.79
"""

literature['Harikane+2021']['Galaxy+AGN']['z5']['phi'] = np.array([
    6.29 * 10 ** -4, 4.45 * 10 ** -4, 4.18 * 10 ** -4, 2.97 * 10 ** -4, 1.79 * 10 ** -4, 1.01 * 10 ** -4,
    5.00 * 10 ** -5, 3.22 * 10 ** -5, 1.40 * 10 ** -5,
    6.39 * 10 ** -6, 1.76 * 10 ** -6, 9.87 * 10 ** -7, 3.19 * 10 ** -7, 1.78 * 10 ** -7, 6.84 * 10 ** -8,
    1.90 * 10 ** -8, 7.70 * 10 ** -9
    ])
literature['Harikane+2021']['Galaxy+AGN']['z5']['phi_lo'] = np.array([
    (6.29 - 0.24) * 10 ** -4, (4.45 - 0.17) * 10 ** -4, (4.18 - 0.14) * 10 ** -4, (2.97 - 0.11) * 10 ** -4,
    (1.79 - 0.06) * 10 ** -4, (1.01 - 0.04) * 10 ** -4,
    (5.00 - 0.22) * 10 ** -5, (3.22 - 0.13) * 10 ** -5, (1.40 - 0.07) * 10 ** -5, (6.39 - 0.43) * 10 ** -6,
    (1.76 - 0.19) * 10 ** -6, (9.87 - 1.6) * 10 ** -7,
    (3.19 - 0.61) * 10 ** -7, (1.78 - 0.64) * 10 ** -7, (6.84 - 4.85) * 10 ** -8, (1.900001 - 1.90) * 10 ** -8,
    (7.700001 - 7.7) * 10 ** -9
    ])
literature['Harikane+2021']['Galaxy+AGN']['z5']['phi_hi'] = np.array([
    (6.29 + 0.24) * 10 ** -4, (4.45 + 0.17) * 10 ** -4, (4.18 + 0.14) * 10 ** -4, (2.97 + 0.11) * 10 ** -4,
    (1.79 + 0.06) * 10 ** -4, (1.01 + 0.04) * 10 ** -4,
    (5.00 + 0.22) * 10 ** -5, (3.22 + 0.13) * 10 ** -5, (1.40 + 0.07) * 10 ** -5, (6.39 + 0.43) * 10 ** -6,
    (1.76 + 0.23) * 10 ** -6, (9.87 + 1.6) * 10 ** -7,
    (3.19 + 0.65) * 10 ** -7, (1.78 + 0.68) * 10 ** -7, (6.84 + 5.14) * 10 ** -8, (1.90 + 2.92) * 10 ** -8,
    (7.70 + 25.20) * 10 ** -9
])



attn_out = {}

fig, axes = plt.subplots(2, 3, figsize = (6, 4), sharex = True, sharey=True)
fig.subplots_adjust(left=0.07, bottom=0.15, top=1.0, right=0.85, wspace=0.0, hspace=0.0)
for i, tag in enumerate(np.flip(fl.tags)):

    z = np.flip(fl.zeds)[i]

    ws, x, y, mstar, lstar, los, dtm = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    for ii in range(len(halo)):
        s = (np.log10(MS[halo[ii]][tag])+10 > 8)
        ws = np.append(ws, np.ones(np.shape(X[halo[ii]][tag][s]))*weights[ii])
        x = np.append(x, X[halo[ii]][tag][s])
        y = np.append(y, Y[halo[ii]][tag][s])
        mstar = np.append(mstar, np.log10(MS[halo[ii]][tag][s])+10)
        lstar = np.append(lstar, np.log10(LFUV[halo[ii]][tag][s]))
        los = np.append(los, BHLOS[halo[ii]][tag][s])
        dtm = np.append(dtm, DTM[halo[ii]][tag][s])

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

    yy = lstar[s_t]

    intrinsic_lum = photom.lum_to_M(10**yy + 10**y_intrinsic)
    dusty_lum = photom.lum_to_M(10**yy + 10**y)

    yy = photom.lum_to_M(10**yy)

    # --- simply print the ranges of the quantities

    print(f'z={z}')
    print(f'm_star,min = {np.min(x)}, m_star,med = {np.median(x)}, m_star,max = {np.max(x)}')
    print(f'L_agn,bol,min = {np.min(q)}, L_agn,bol,med = {np.median(q)}, L_agn,bol,max = {np.max(q)}')
    print('Dust:')
    print(f'L_agn,uv,min = {np.min(y)}, L_agn,uv,med = {np.median(y)}, L_agn,uv,max = {np.max(y)}')
    print('Dust, Total:')
    print(f'L_total,uv,min = {np.min(dusty_lum)}, L_total,uv,med = {np.median(dusty_lum)}, L_total,uv,max = {np.max(dusty_lum)}')
    print('Intrinsic:')
    print(f'L_agn,uv,min = {np.min(y_intrinsic)}, L_agn,uv,med = {np.median(y_intrinsic)}, L_agn,uv,max = {np.max(y_intrinsic)}')
    print('Intrinsic, Total:')
    print(f'L_total,uv,min = {np.min(intrinsic_lum)}, L_total,uv,med = {np.median(intrinsic_lum)}, L_total,uv,max = {np.max(intrinsic_lum)}')

    binw = 1.
    bins = np.arange(-28.5,-17,binw)
    b_c = bins[:-1]+binw/2

    N_weighted_gal, edges_gal = np.histogram(yy, bins = bins, weights = ws)

    N_weighted_dusty, edges_dusty = np.histogram(dusty_lum, bins = bins, weights = ws)

    N_weighted_intrinsic, edges_intrinsic = np.histogram(intrinsic_lum, bins=bins, weights=ws)

    N_weighted_dusty += 0.000000000001

    test_y1, edges1 = np.histogram(photom.lum_to_M(10**y), bins = bins, weights = ws)
    test_y2, edges2 = np.histogram(photom.lum_to_M(10**y_intrinsic), bins=bins, weights=ws)

    N_number_of_agn, edges =  np.histogram(y, bins = bins)
    Ns_agn = (N_number_of_agn > 5)
    not_Ns_agn = np.invert(Ns_agn)
    uplims_agn = np.ones_like(not_Ns_agn)
    err = np.sqrt(np.histogram(y, bins = bins, weights = ws**2)[0])
    err_lo = err
    err_hi = err


    N_number_of_gal, edges = np.histogram(yy, bins=bins)
    Ns_gal = (N_number_of_gal > 5)
    not_Ns_gal = np.invert(Ns_gal)

    N_number_of_total, edges = np.histogram(yy, bins=bins)
    Ns_total = (N_number_of_total > 5)
    not_Ns_total = np.invert(Ns_total)

    N_number_of_intrinsic, edges = np.histogram(y_intrinsic, bins=bins)
    Ns_intrinsic = (N_number_of_intrinsic > 5)
    not_Ns_intrinsic = np.invert(Ns_intrinsic)

    h = 0.6777
    vol = (4/3)*np.pi*(14/h)**3

    phi_intrinsic = N_weighted_intrinsic / (binw * vol)
    phi_dusty = N_weighted_dusty / (binw * vol)
    phi_gal = N_weighted_gal / (binw * vol)

    phitest1 = test_y1 / (binw * vol)
    phitest2 = test_y2 / (binw * vol)

    try:
        axes.flatten()[i].fill_between(bins[:-1] + binw / 2, np.log10(phi_dusty), np.log10(phi_intrinsic), color=cmap(norm(z)), alpha=0.5)
    except:
        print('No shaded region')

    axes.flatten()[i].plot(bins[:-1] + binw / 2, np.log10(phi_gal), ls='-', c=cmap(norm(z)))


    axes.flatten()[i].plot(bins[:-1] + binw / 2, np.log10(phitest1), ls='--', c=cmap(norm(z)), alpha=0.5)
    axes.flatten()[i].plot(bins[:-1] + binw / 2, np.log10(phitest2), ls='--', c=cmap(norm(z)), alpha=0.5)

    if z == 5.0:
        for jj in range(len(literature['Niida+2020']['HSC']['M'])):
            M_ = [literature['Niida+2020']['HSC']['M'][jj], literature['Niida+2020']['HSC']['M'][jj]]
            err = [np.log10(literature['Niida+2020']['HSC']['phi_lo'][jj]), np.log10(literature['Niida+2020']['HSC']['phi_hi'][jj])]
            axes.flatten()[i].plot(M_, err, lw=1, ls='-', c='purple', alpha=0.8)

        axes.flatten()[i].scatter(literature['Niida+2020']['HSC']['M'],
                                  np.log10(literature['Niida+2020']['HSC']['phi']), s=5, marker='^', c='purple')

        for jj in range(len(literature['Harikane+2021']['Galaxy+AGN']['z5']['M'])):
            M_ = [literature['Harikane+2021']['Galaxy+AGN']['z5']['M'][jj], literature['Harikane+2021']['Galaxy+AGN']['z5']['M'][jj]]
            err = [np.log10(literature['Harikane+2021']['Galaxy+AGN']['z5']['phi_lo'][jj]),
                   np.log10(literature['Harikane+2021']['Galaxy+AGN']['z5']['phi_hi'][jj])]
            axes.flatten()[i].plot(M_, err, lw=1, ls='-', c='black', alpha=0.8)

        axes.flatten()[i].scatter(literature['Harikane+2021']['Galaxy+AGN']['z5']['M'],
                                  np.log10(literature['Harikane+2021']['Galaxy+AGN']['z5']['phi']), s=5, marker='v', c='black')



    axes.flatten()[i].text(0.7, 0.9, r'$\rm z={0:.0f}$'.format(z), fontsize=8, transform=axes.flatten()[i].transAxes,
                           color=cmap(norm(z)))

    axes.flatten()[i].set_ylim(-8, -2.1)
    axes.flatten()[i].set_xlim(-26.5, -18.5)
    axes.flatten()[i].set_xticks([-25, -23, -21, -19])

    axes.flatten()[i].invert_xaxis()


axes.flatten()[5].plot(-99, -99, ls='-', c='k', alpha=0.6, label=rf'Stellar')
axes.flatten()[5].fill_between([-99, -90], [-99, -99], [-90, -90], color='k', alpha=0.4, label=rf'Stellar + $\rm AGN_{{intrinsic, dust}}$')

axes.flatten()[0].errorbar(-99, -99, 1, ms=2, marker='^', ls='none', c='purple', mew=2, label='Niida+20, z=5')
axes.flatten()[0].errorbar(-99, -99, 1, ms=2, marker='v', ls='none', c='black', mew=2, label='Harikane+21, z=5')

axes.flatten()[0].legend(loc='upper left', prop={'size': 6})
axes.flatten()[5].legend(loc='upper left', prop={'size': 6})

fig.text(0.01, 0.55, r'$\rm log_{10}[\phi\;/\;Mpc^{-3}\, mag^{-1}]$', ha = 'left', va = 'center', rotation = 'vertical', fontsize=10)
fig.text(0.45,0.05, r'$\rm M_{FUV}$', ha = 'center', va = 'bottom', fontsize=10)

fig.savefig(f'figures/uvlf__agn_dust__intrinsic_comparisons_grid_magnitude.pdf', bbox_inches='tight')
fig.clf()
