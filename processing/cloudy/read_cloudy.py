
import numpy as np


i = 1 # model index (should start at 1)

lam, incident, transmitted, nebular, total, linecont  = np.loadtxt(f'{output_dir}/{i}.cont'), delimiter='\t', usecols = (0,1,2,3,4,8)).T

# lam = wavelength
# incident = should be the input spectrum
# transmitted = the spectrum which is transmitted through the nebular
# nebular = the emission from the nebular
# total = transmitted + nebular
# linecont = **possibly** the fraction of emission from lines?

# Ross, try this and plot a few against each other?
