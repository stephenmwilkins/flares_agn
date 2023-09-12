import numpy as np
import pandas as pd

import astropy.constants as constants
import astropy.units as units

import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt

import flares
import flare.plt as fplt

# ----------------------------------------------------------------------
# --- open data

deltas = np.array([0.969639,0.918132,0.851838,0.849271,0.845644,0.842128,0.841291,0.83945,0.838891,0.832753,0.830465,0.829349,0.827842,0.824159,0.821425,0.820476,0.616236,0.616012,0.430745,0.430689,0.266515,0.266571,0.121315,0.121147,-0.007368,-0.007424,-0.121207,-0.121319,-0.222044,-0.222156,-0.311441,-0.311329,-0.066017,-0.066185,-0.00748,-0.007424,0.055076,0.054909,-0.47874,-0.433818])

fl = flares.flares('/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5', sim_type='FLARES')
halo = fl.halos



# ----------------------------------------------------------------------
# --- define parameters

tag = fl.tags[-3]  # --- select tag -3 = z=7
# spec_type = 'DustModelI' # --- select photometry type
log10Mstar_limit = 9.




# converting MBHacc units to M_sol/yr
h = 0.6777  # Hubble parameter
BH_Mdot_scaling = h * 6.445909132449984E23  # g/s
BH_Mdot_scaling /= constants.M_sun.to('g').value  # convert to M_sol/s
BH_Mdot_scaling *= units.yr.to('s')  # convert to M_sol/yr



# ----------------------------------------------------------------------
# --- define quantities to read in [not those for the corner plot, that's done later]

quantities = []
quantities.append({'path': 'Galaxy', 'dataset': 'SFR_inst_30', 'name': None, 'scaling': None})
quantities.append({'path': 'Galaxy', 'dataset': 'Mstar_30', 'name': None, 'scaling': 1E10})
quantities.append({'path': 'Galaxy', 'dataset': 'BH_Mass', 'name': None, 'scaling': 1E10})
quantities.append({'path': 'Galaxy', 'dataset': 'BH_Mdot', 'name': None, 'scaling': BH_Mdot_scaling})




# ----------------------------------------------------------------------
# --- get all the required quantities, including the weights and delta

qs = []
d = {}
D = {}
scalings = {}
for Q in quantities:

    if not Q['name']:
        qname = Q['dataset']
    else:
        qname = Q['name']

    d[qname] = fl.load_dataset(Q['dataset'], arr_type=Q['path'])
    D[qname] = np.array([])
    qs.append(qname)
    scalings[qname] = Q['scaling']

# --- read in weights
df = pd.read_csv('/cosma/home/dp004/dc-wilk2/data/flare/modules/flares/weight_files/weights_grid.txt')
weights = np.array(df['weights'])
ws = np.array([])
delta = np.array([])

for ii in range(len(halo)):
    for q in qs:
        D[q] = np.append(D[q], d[q][halo[ii]][tag])
    ws = np.append(ws, np.ones(np.shape(d[q][halo[ii]][tag]))*weights[ii])
    delta = np.append(delta, np.ones(np.shape(d[q][halo[ii]][tag]))*deltas[ii])


for q in qs:
    if scalings[q]:
        D[q] *= scalings[q]





# ----------------------------------------------
# define new quantities
D['log10Mstar_30'] = np.log10(D['Mstar_30'])
D['log10BH_Mdot'] = np.log10(D['BH_Mdot'])
D['log10BH_Mass'] = np.log10(D['BH_Mass'])
D['log10sSFR'] = np.log10(D['SFR_inst_30'])-np.log10(D['Mstar_30'])+9

# ----------------------------------------------
# define selection
s = D['log10Mstar_30']>log10Mstar_limit

# ----------------------------------------------
# Print info
print(f'Total number of galaxies: {len(ws[s])}')







# ----------------------------------------------
# define quantities for including in the corner plot

properties = ['log10Mstar_30', 'log10sSFR', 'log10BH_Mass', 'log10BH_Mdot']

labels = {}
labels['log10Mstar_30'] = r'\log_{10}({\rm M_{\star}}/{\rm M_{\odot})}'
labels['log10sSFR'] = r'\log_{10}({\rm sSFR}/{\rm Gyr^{-1})}'
labels['beta'] = r'\beta'
labels['log10HbetaEW'] = r'\log_{10}(H\beta\ EW/\AA)'
labels['log10FUV'] = r'\log_{10}(L_{FUV}/erg\ s^{-1}\ Hz^{-1})'
labels['AFUV'] = r'A_{FUV}'
labels['log10BH_Mass'] = r'\log_{10}({\rm M_{\bullet}}/{\rm M_{\odot})}'
labels['log10BH_Mdot'] = r'\log_{10}({\rm M_{\bullet}}/{\rm M_{\odot}\ yr^{-1})}'

limits = {}
limits['log10Mstar_30'] = [log10Mstar_limit,11]
limits['log10BH_Mass'] = [5.1,9.9]
limits['log10BH_Mdot'] = [-2.9,1.9]
limits['beta'] = [-2.9,-1.1]
limits['log10sSFR'] = [-0.9,1.9]
limits['log10HbetaEW'] = [0.01,2.49]
limits['AFUV'] = [0,3.9]
limits['log10FUV'] = [28.1,29.9]

# ----------------------------------------------
# ----------------------------------------------
# Corner Plot


norm = mpl.colors.Normalize(vmin=8.5, vmax=10.5)
cmap = cm.viridis

N = len(properties)

fig, axes = plt.subplots(N, N, figsize = (7,7))
plt.subplots_adjust(left=0.1, top=0.9, bottom=0.1, right=0.9, wspace=0.02, hspace=0.02)

for i in np.arange(N):
    for j in np.arange(N):
        axes[i, j].set_axis_off()

for i,x in enumerate(properties):
    for j,y in enumerate(properties[1:][::-1]):

        jj = N-1-j
        ii = i

        ax = axes[jj, ii]

        if j+i<(N-1):
            ax.set_axis_on()

            # --- scatter plot here
            ax.scatter(D[x][s],D[y][s], s=1, alpha=0.5, c = cmap(norm(D['log10Mstar_30'][s])))

            # --- weighted median Lines

            bins = np.linspace(*limits[x], 20)
            bincen = (bins[:-1]+bins[1:])/2.
            out = flares.binned_weighted_quantile(D[x][s],D[y][s], ws[s],bins,[0.84,0.50,0.16])

            ax.plot(bincen, out[:,1], c='k', ls = '-')
            # ax.fill_between(bincen[Ns], out[:,0][Ns], out[:,2][Ns], color='k', alpha = 0.2)

            ax.set_xlim(limits[x])
            ax.set_ylim(limits[y])

        if i == 0: # first column
            ax.set_ylabel(rf'$\rm {labels[y]}$', fontsize = 7)
        else:
            ax.yaxis.set_ticklabels([])

        if j == 0: # first row
            ax.set_xlabel(rf'$\rm {labels[x]}$', fontsize = 7)
        else:
            ax.xaxis.set_ticklabels([])

        # ax.text(0.5, 0.5, f'x{i}-y{j}', transform = ax.transAxes)


    # --- histograms

    bins = 50

    ax = axes[ii, ii]
    ax.set_axis_on()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    X = D[x][s]

    H, bin_edges = np.histogram(X, bins = bins, range = limits[x])
    Hw, bin_edges = np.histogram(X, bins = bins, range = limits[x], weights = ws[s])

    Hw *= np.max(H)/np.max(Hw)

    bin_centres = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])*0.5


    ax.fill_between(bin_centres, H*0.0, H, color='0.9')
    ax.plot(bin_centres, Hw, c='0.7', lw=1)

    ax.set_ylim([0.0,np.max(H)*1.2])





# --- add colourbar

cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
cmapper.set_array([])

cax = fig.add_axes([0.25, 0.87, 0.5, 0.015])
fig.colorbar(cmapper, cax=cax, orientation='horizontal')
cax.set_xlabel(r'$\rm\log_{10}({\rm M_{\star}}/{\rm M_{\odot})}$')



fig.savefig(f'figures/corner.pdf')
