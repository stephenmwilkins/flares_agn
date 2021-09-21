import numpy as np
import pandas as pd
import flares
import _pickle as pickle

flares_dir = '../../../../data/simulations'

fl = flares.flares(f'{flares_dir}/flares_no_particlesed.hdf5', sim_type='FLARES') #_no_particlesed
df = pd.read_csv(f'{flares_dir}/weights_grid.txt')

lbol_frac = pickle.load(open('lbol_frac.p', 'rb'))

for halo in fl.halos:
    for tag in fl.tags:
        print(len(lbol_frac[halo][tag]))

