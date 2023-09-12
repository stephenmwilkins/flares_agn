import numpy as np

output_dir = 'output/linear'

AGN_T = np.linspace(10000, 1.5e6, 100) # range of AGN temperatures to model

for i in range(len(AGN_T)):
    lam, incident, transmitted, nebular, total, linecont  = np.loadtxt(f'{output_dir}/{i}.cont', delimiter='\t', usecols = (0,1,2,3,4,8)).T


# lam = wavelength
# incident = should be the input spectrum
# transmitted = the spectrum which is transmitted through the nebular
# nebular = the emission from the nebular
# total = transmitted + nebular
# linecont = **possibly** the fraction of emission from lines?
