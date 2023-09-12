def L_AGN(Mdot):
    # This function will calculate the bolometric luminosity of an
    # AGN from an input of accretion rate input of log10(Mdot/(M_sol/yr))
    # the output will be log10(Lagn/(M_sol/yr))
    import numpy as np
    import astropy.constants as const
    import astropy.units as u
    Msun = const.M_sun
    Lsun = const.L_sun
    c = const.c
    Mdot_SI = ((10 ** Mdot) * Msun * (u.yr ** -1))
    n = 0.1
    Lagn_SI = Mdot_SI * (c ** 2)
    Lagn = np.log10(Lagn_SI / Lsun)
    return Lagn


def L_AGN_erg(Mdot):
    # This function will calculate the bolometric luminosity of an
    # AGN from an input of accretion rate input of log10(Mdot/(M_sol/yr))
    # the output will be log10(Lagn/(erg/s))
    import numpy as np
    import astropy.constants as const
    import astropy.units as u
    Msun = const.M_sun
    Lsun = const.L_sun
    c = const.c
    Mdot_SI = ((10 ** Mdot) * Msun * (u.yr ** -1))
    n = 0.1
    Lagn_SI = Mdot_SI * (c ** 2)
    Lagn_CGS = Lagn_SI.to(u.erg / u.s)
    Lagn = np.log10(Lagn_CGS / (u.erg / u.s))
    return Lagn


def Lagn_T_individual(wanted_zed_tag):
    '''Plots T_BB against L_AGN individually, takes single input, must be string:
    'all' gives Lagn vs T for all possible redshifts
    '-1' gives redshift 5
    '-2' gives redshift 6
    '-3' gives redshift 7
    '-4' gives redshift 8
    '-5' gives redshift 9
    '-6' gives redshift 10
        '''
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.cm as cm
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import flares
    import astropy.constants as constants
    import astropy.units as units

    # sorting which zed tags are needed
    if wanted_zed_tag == 'all':
        zed_tags = (-1, -2, -3, -4, -5, -6)
    else:
        zed_tags = [int(wanted_zed_tag)]

    flares_dir = '../../../data/simulations'

    # importing flares data
    fl = flares.flares(f'{flares_dir}/flares.hdf5', sim_type='FLARES')
    halo = fl.halos

    # defining colormap
    cmap = mpl.cm.plasma

    # loading required datasets
    Mstar = fl.load_dataset('Mstar_30', arr_type='Galaxy')  # stellar mass of galaxy
    MBHdot_ = fl.load_dataset('BH_Mdot', arr_type='Galaxy')  # Black accretion rate
    MBH_ = fl.load_dataset('BH_Mass', arr_type='Galaxy')  # Black hole mass of galaxy

    X = Mstar
    fig = plt.figure(figsize=(14, 6))
    plot_no = 1
    temp_max = []
    temp_min = []

    for zed_tag in zed_tags:
        tag = fl.tags[zed_tag]
        z = fl.zeds[zed_tag]  # converting zedtag to redshift number

        # loading in weights
        df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
        weights = np.array(df['weights'])

        # creating data arrays for plotting
        ws, x, MBH, MBHdot = np.array([]), np.array([]), np.array([]), np.array([])
        for ii in range(len(halo)):
            ws = np.append(ws, np.ones(np.shape(MBH_[halo[ii]][tag])) * weights[ii])
            x = np.append(x, np.log10(X[halo[ii]][tag]))
            MBH = np.append(MBH, np.log10(MBH_[halo[ii]][tag]))
            MBHdot = np.append(MBHdot, np.log10(MBHdot_[halo[ii]][tag]))
        MBH += 10.  # converting MBH units to M_sol
        h = 0.6777  # Hubble parameter

        # converting MBHacc units to M_sol/yr
        MBHacc = MBHdot + np.log10(h * 6.445909132449984E23)  # g/s
        MBHacc -= np.log10(constants.M_sun.to('g').value)  # convert to M_sol/s
        MBHacc += np.log10(units.yr.to('s'))  # convert to M_sol/yr

        x += 10  # changing units of Mstar to M_sol

        # cutting out galaxies below mstar = 10**8
        y = MBH
        y_dot = MBHacc
        xcut, ycut, y_dotcut = np.array([]), np.array([]), np.array([])
        for i in range(len(y)):
            if x[i] > 8:
                xcut = np.append(xcut, x[i])
                ycut = np.append(ycut, y[i])
                y_dotcut = np.append(y_dotcut, y_dot[i])
            else:
                continue

        x = xcut
        y = ycut
        y_dot = y_dotcut

        # calculating luminosities
        lums = np.array([])
        for l in range(len(x)):
            lum = L_AGN(y_dot[l])
            lums = np.append(lums, lum)
        # calculating big bump temperatures
        temps = np.array([])
        for t in range(len(x)):
            temp = ((2.24e9) * ((10 ** y_dot[t]) ** (0.25)) * ((10 ** y[t]) ** (-0.5)))
            temps = np.append(temps, temp)

        # plotting data individually at each redshift
        plt.scatter(lums, temps / 10 ** 6, alpha=0.25)  # ,s=3,c=temps, cmap = cmap,alpha=0.25)
        plt.title('z = ' + str(z))
        plt.xlabel(r'AGN Luminosity ($\rm log_{10}(L_{AGN}/L_{\odot}yr))$')
        plt.ylabel(r'Temperature (T$_{AGN}$/$10^6$ K)')
        plt.ylim(0)
        plt.xlim(0)
        plt.grid()
        # plt.ticklabel_format(axis = 'y', style = 'sci', scilimits=(6,6))

        fig.savefig(f'figures/ROSS_LAGN_T_' + str(int(z)) + '.png')
        plt.clf()
        fig.clf()


Lagn_T_individual(-1)