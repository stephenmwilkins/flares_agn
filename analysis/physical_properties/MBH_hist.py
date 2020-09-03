import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt

import flares

flares_dir = '/research/astro/flare/'

fl = flares.flares(f'{flares_dir}/simulations/FLARES/data/flares.hdf5', sim_type='FLARES')
halo = fl.halos
print(fl.tags) # print list of available snapshots
tag = fl.tags[-1]  #This would be z=5

z = fl.zeds[-1] # get redshift



MBH = fl.load_dataset('BH_Mass', arr_type='Galaxy') # Black hole mass of galaxy


X = MBH


# --- FLARES is composed of 40 individual simulations. Each simulation has a statistical weight (basically the likelhood a similar region would appear in a volume of the real Universe). To get true distribution functions we need to weight each galaxy by the weight of the simulation.

# --- This bit of codes reads in the properties for all of the galaxies in every simulation and also produces a corresponding weight array. These can be combined later to get a true relationship.

df = pd.read_csv(f'{flares_dir}/modules/flares/weight_files/weights_grid.txt')
weights = np.array(df['weights'])
ws, x = np.array([]), np.array([])
for ii in range(len(halo)):
    # ws = np.append(ws, np.ones(np.shape(X[halo[ii]][tag]))*weights[ii])
    x = np.append(x, np.log10(X[halo[ii]][tag]))


x += 10 # units are 1E10 M_sol

binw = 0.25
bins = np.arange(5,10,binw)
b_c = bins[:-1]+binw/2

N, edges = np.histogram(x, bins = bins)


# -- simpy make a scatter plot

fig = plt.figure(figsize=(3,3))
left  = 0.2
bottom = 0.2
width = 0.75
height = 0.75
ax = fig.add_axes((left, bottom, width, height))
# ax.legend()

# --- plot the histogram

ax.plot(bins[:-1]+binw/2, np.log10(N))



ax.set_xlabel(r'$\rm log_{10}(M_{BH}/M_{\odot})$')
ax.set_ylabel(r'$\rm log_{10}(N)$')
fig.savefig(f'figures/MBH_hist.pdf')
fig.clf()
