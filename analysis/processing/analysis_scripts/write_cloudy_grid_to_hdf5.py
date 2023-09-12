import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt
import h5py

cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin = np.log10(1e4), vmax = np.log10(1.5e6))

output_dir = '../cloudy/output/linear'

hf = h5py.File('agn_grid.h5', 'w')

AGN_T = np.linspace(10000, 1.5e6, 100)  # range of AGN temperatures to model

hf.create_dataset('T_AGN', data=AGN_T)

incidents = []
transmitteds = []
nebulars = []
totals = []
lineconts = []

for i in range(len(AGN_T)):
    lam, incident, transmitted, nebular, total, linecont = np.loadtxt(f'{output_dir}/{i+1}.cont', delimiter='\t',
                                                                      usecols=(0, 1, 2, 3, 4, 8)).T
    if i == 0:
        hf.create_dataset('lambda', data=lam)

    incidents.append(incident)
    transmitteds.append(transmitted)
    nebulars.append(nebular)
    totals.append(total)
    lineconts.append(linecont)

hf.create_dataset('total', data=np.array(totals))

hf.close()
