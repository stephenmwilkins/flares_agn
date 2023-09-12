import numpy as np



AGN_T_ = np.linspace(10000, 1.5e6, 100) # range of AGN temperatures to model

output_dir = 'output/linear'


for i, AGN_T in enumerate(AGN_T_):

    AGN_T = round(int(AGN_T), 3 - len(str(int(AGN_T))))

    cinput = ["AGN T = "+str(AGN_T)+" k  a(ox) = -1.4    a(uv) = -0.5  a(x) = -1\n"
    "luminosity 43\n"
    "radius 18\n"
    "hden 10\n"
    'stop column density 22\n'
    'iterate to convergence\n'
    ] # cloudy instructions file

    # --- define output filename

    output_file = f'{i+1}' # label outputs by index

    cinput.append(f'save last continuum "{output_dir}/{output_file}.cont" units Angstroms no clobber\n') # save the continuum emission
    cinput.append(f'save last lines, array "{output_dir}/{output_file}.lines" units Angstroms no clobber\n') # save the line emission
    cinput.append(f'save overview "{output_dir}/{output_file}.ovr" last\n') # save the overview file

    # --- write cloudy input file

    f = open('input/linear/'+str(i+1)+'.in','w')
    f.writelines(cinput)
    f.close()
