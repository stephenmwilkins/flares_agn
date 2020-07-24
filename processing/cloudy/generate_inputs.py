

import numpy as np



AGN_T_ = np.arange() # range of AGN temperatures to model

output_dir = 'output/'


for i, AGN_T in enumerate(AGN_T_):

    cinput = [] # cloudy instructions file

    # --- insert cloudy commands here






    # --- define output filename

    output_file = f'{i+1}' # label outputs by index

    cinput.append(f'save last continuum "{output_dir}/{output_file}.cont" units Angstroms no clobber\n') # save the continuum emission
    cinput.append(f'save last lines, array "{output_dir}/{output_file}.lines" units Angstroms no clobber\n') # save the line emission
    cinput.append(f'save overview "{output_dir}/{output_file}.ovr" last\n') # save the overview file

    # --- write cloudy input file

    f = open('cinputs/'+str(i+1)+'.in','w')
    f.writelines(cinput)
    f.close()
