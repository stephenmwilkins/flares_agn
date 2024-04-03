
import numpy as np
from astropy.io import ascii
from astropy.table import Table, Column
import flares_utility.analyse as analyse
from unyt import c, Msun, yr, g, s, erg, Lsun


tags = ['005_z010p000', '006_z009p000', '007_z008p000',
        '008_z007p000', '009_z006p000', '010_z005p000']
redshifts = [10, 9, 8, 7, 6, 5]


binw = 0.2
xrange = [45., 46.5]
bin_edges = np.arange(*xrange, binw)
bin_centres = bin_edges[:-1]+binw/2

# define conversion from bhacc to bolometric luminosity
bhacc_conversion = 6.446E23 * g / s
conversion = 0.1 * bhacc_conversion * c**2
conversion = conversion.to('erg/s').value
log10conversion = np.log10(conversion)



filename = '/Users/sw376/Dropbox/Research/data/simulations/flares/flares_no_particlesed.hdf5'
flares = analyse.analyse(filename)

V = (4./3) * np.pi * (flares.radius)**3  # total volume of FLARES Mpc^3

quantity = ['Galaxy', 'BH_Mdot']

# initialise output arrays, for single column tble
z_ = np.array([])
N_ = np.array([])
phi_ = np.array([])
bin_centres_ = np.array([])

# table with a different column for each redshift
# initialise table object and arrays
t2 = Table()
t2.add_column(Column(data=np.round(bin_centres, 2),
                    name='log10Lbol',
                    unit='dex(erg/s)',
                    description=r'log10(bolometric luminosity/erg/s)'))

for z in redshifts:

    tag = flares.tag_from_zed[z]

    quantities = [{'path': quantity[0],
                   'dataset': quantity[1],
                   'name': 'X',
                   'log10': True}]
    
    D = flares.get_datasets(tag, quantities)

    w = D['weight']
    log10X = D['log10X'] + log10conversion


    N, _ = np.histogram(log10X, bins=bin_edges)
    Neff, _ = np.histogram(log10X, bins=bin_edges, weights=w)
    phi = Neff/V/binw


    t2.add_column(Column(data=np.round(np.log10(phi), 2),
                        name=f'log10phi_{int(z)}',
                        unit='dex(Mpc^-3 dex^-1)',
                        description=rf'log_{{10}}(\phi({int(z)}/Mpc^{{-3}}\ dex^{{-1}})'))

    # exclude bins with no objects
    s = N>0

    # stack
    z_ = np.hstack((z_, np.ones(len(N[s]))*z))
    bin_centres_ = np.hstack((bin_centres_, bin_centres[s]))
    N_ = np.hstack((N_, N[s]))
    phi_ = np.hstack((phi_, phi[s]))


# initialise table object and arrays
t = Table()
t.add_column(Column(data=z_, name='z'))
t.add_column(Column(data=np.round(bin_centres_, 2),
                    name='log10Lbol',
                    unit='dex(erg/s)',
                    description=r'log10(bolometric luminosity/erg/s)'))
t.add_column(Column(data=N_.astype(int), 
                    name='N', 
                    description='Number of galaxies in bin'))
t.add_column(Column(data=np.round(np.log10(phi_), 2),
                    name='log10phi', 
                    unit='dex(Mpc^-3 dex^-1)', 
                    description=r'log_{10}(\phi/Mpc^{-3}\ dex^{-1})'))

t.meta['name'] = 'FLARES'
t.meta['references'] = []
t.write('../../flares_agn_data/Lbol_DF.ecsv', format='ascii.ecsv', overwrite=True)

t2.meta['name'] = 'FLARES'
t2.meta['references'] = []
t2.write('../../flares_agn_data/Lbol_DF2.ecsv', format='ascii.ecsv', overwrite=True)




# Now convert to LaTeX for the paper

# ascii.write(t, include_names=['log10Mbh','N','log10phi'], format='latex')  
ascii.write(t2, format='latex')  