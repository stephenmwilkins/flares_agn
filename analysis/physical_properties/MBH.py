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

# --- define redshift colour scale

cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=5., vmax=10.)




Mstar = fl.load_dataset('Mstar_30', arr_type='Galaxy') # stellar mass of galaxy
MBH = fl.load_dataset('BH_Mass', arr_type='Galaxy') # Black hole mass of galaxy
# Mdot = fl.load_dataset('BH_Mass', arr_type='Galaxy') # Black hole acretion rate

X = Mstar
Y = MBH



# --- FLARES is composed of 40 individual simulations. Each simulation has a statistical weight (basically the likelhood a similar region would appear in a volume of the real Universe). To get true distribution functions we need to weight each galaxy by the weight of the simulation.

# --- This bit of codes reads in the properties for all of the galaxies in every simulation and also produces a corresponding weight array. These can be combined later to get a true relationship.

df = pd.read_csv(f'{flares_dir}/modules/flares/weight_files/weights_grid.txt')
weights = np.array(df['weights'])
ws, x, y = np.array([]), np.array([]), np.array([])
for ii in range(len(halo)):
    ws = np.append(ws, np.ones(np.shape(X[halo[ii]][tag]))*weights[ii])
    x = np.append(x, np.log10(X[halo[ii]][tag]))
    y = np.append(y, np.log10(Y[halo[ii]][tag]))


x += 10 # units are 1E10 M_sol
y += 10 # units are 1E10 M_sol

# --- simply print the ranges of the quantities

print(np.min(x),np.median(x),np.max(x))
print(np.min(y),np.median(y),np.max(y))


# -- this will calculate the weighted quantiles of the distribution
quantiles = [0.84,0.50,0.16] # quantiles for range
bins = np.arange(9, 11.5, 0.2) # x-coordinate bins, in this case stellar mass
bincen = (bins[:-1]+bins[1:])/2.
out = flares.binned_weighted_quantile(x,y,ws,bins,quantiles)

# NOTE: this appears to break in this case presumably because many black holes masses are zero (thus -inf in log space).


# -- simpy make a scatter plot

fig = plt.figure(figsize=(3,3))
left  = 0.2
bottom = 0.2
width = 0.75
height = 0.75
ax = fig.add_axes((left, bottom, width, height))
# ax.legend()

# --- plot the median and quantiles for bins with >10 galaxies

N, bin_edges = np.histogram(x, bins=bins)
Ns = N>10
ax.plot(bincen, out[:,1], c=cmap(norm(z)), ls= ':')
ax.plot(bincen[Ns], out[:,1][Ns], c=cmap(norm(z)), label = rf'$\rm z={int(z)}$')
ax.fill_between(bincen[Ns], out[:,0][Ns], out[:,2][Ns], color=cmap(norm(z)), alpha = 0.2)

# --- scatter plot

ax.scatter(x,y,s=3,c='k',alpha=0.1)

ax.set_xlabel(r'$\rm log_{10}(M_{\star}/M_{\odot})$')
ax.set_ylabel(r'$\rm log_{10}(M_{BH}/M_{\odot})$')
fig.savefig(f'figures/MBH.png')
fig.clf()
