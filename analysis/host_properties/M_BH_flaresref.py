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

fl = flares.flares(f'{flares_dir}/flares_no_particlesed.hdf5', sim_type='FLARES')
eagle = flares.flares(f'{flares_dir}/EAGLE_REF_sp_info.hdf5', sim_type='PERIODIC')
df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
halo = fl.halos
print(fl.tags) # print list of available snapshots
tags = fl.tags #This would be z=5

Mdot = fl.load_dataset('BH_Mdot', arr_type='Galaxy') # Black hole mass of galaxy
MBH = fl.load_dataset('BH_Mass', arr_type='Galaxy') # Black hole mass of galaxy
MBH_ref = eagle.load_dataset('BH_Mass', arr_type='Galaxy')

Y = MBH
X = Mdot



# --- FLARES is composed of 40 individual simulations. Each simulation has a statistical weight (basically the likelhood a similar region would appear in a volume of the real Universe). To get true distribution functions we need to weight each galaxy by the weight of the simulation.

# --- This bit of codes reads in the properties for all of the galaxies in every simulation and also produces a corresponding weight array. These can be combined later to get a true relationship.

df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
weights = np.array(df['weights'])

fig = plt.figure(figsize=(2,2))
ax1 = fig.add_axes( [0., 0., 1., 1.] )

m_bh = np.linspace(5., 10, 100)
accr = np.linspace(-10, 1, 100)

eagle_tag = eagle.tags[-1]
flares_tag = fl.tags[-1]
eagle_z = eagle.zeds[-1]
flares_z = fl.zeds[-1]

eagle_mbh = np.log10(MBH_ref[eagle_tag])+10

flares_mbh, ws = np.array([]), np.array([])
for ii in range(len(halo)):
    ws = np.append(ws, np.ones(np.shape(MBH[halo[ii]][flares_tag])) * weights[ii])
    flares_mbh = np.append(flares_mbh, np.log10(MBH[halo[ii]][flares_tag])+10)


bins = np.arange(5, 10, 0.25)  #  x-coordinate bins, in this case stellar mass
bincen = (bins[:-1] + bins[1:]) / 2.

N_flares, binedges = np.histogram(flares_mbh, bins=bins)
N_eagle, binedges =  np.histogram(eagle_mbh, bins=bins)

print(f'FLARES z = {flares_z}, N_BH = {sum(N_flares)}')
print(f'EAGLE  z = {eagle_z}, N_BH = {sum(N_eagle)}')

print('bin centres:')
print(bincen)
print('FLARES:')
print(N_flares)
print('EAGLE:')
print(N_eagle)

ax1.step(bincen, np.log10(N_flares), ls='-', c='k', where='mid', label=fr"FLARES")
ax1.step(bincen, np.log10(N_eagle), ls=':', c='k', where='mid', label=fr"Eagle REF")

ax1.text(0.1, 0.9, r'$\rm z=5$', fontsize=12, transform=ax1.transAxes, color='k')

ax1.set_xticks([6, 7, 8, 9])
ax1.set_xlim(5.5, 9.5)
ax1.set_ylim(0, 3.4)

ax1.set_xlabel(r'$\rm log_{10}[M_{SMBH}\;/\;M_{\odot}]$')
ax1.set_ylabel(r'$\rm log_{10}[N_{SMBH}]$')
ax1.legend(loc="best")

fig.savefig(f'figures/l_agn/M_BH_hist_comparison.pdf', bbox_inches='tight')
fig.clf()
