import numpy as np
import pandas as pd

import astropy.constants as constants
import astropy.units as units

import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt

import flares
import FLARE.plt as fplt

import matplotlib.gridspec as gsc

cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=5., vmax=10.)
# ----------------------------------------------------------------------
# --- open data

deltas = np.array([0.969639,0.918132,0.851838,0.849271,0.845644,0.842128,0.841291,0.83945,0.838891,0.832753,0.830465,0.829349,0.827842,0.824159,0.821425,0.820476,0.616236,0.616012,0.430745,0.430689,0.266515,0.266571,0.121315,0.121147,-0.007368,-0.007424,-0.121207,-0.121319,-0.222044,-0.222156,-0.311441,-0.311329,-0.066017,-0.066185,-0.00748,-0.007424,0.055076,0.054909,-0.47874,-0.433818])

flares_dir = '../../../data/simulations'

fl = flares.flares(f'{flares_dir}/flares_no_particlesed.hdf5', sim_type='FLARES')

halo = fl.halos



# ----------------------------------------------------------------------
# --- define parameters

tags = [fl.tags[-1], fl.tags[-2], fl.tags[-3]]  # --- select tag -3 = z=7
zeds = [fl.zeds[-1], fl.zeds[-2], fl.zeds[-3]]
# spec_type = 'DustModelI' # --- select photometry type
log10Mstar_limit = 9.




# converting MBHacc units to M_sol/yr
h = 0.6777  # Hubble parameter
BH_Mdot_scaling = h #* 6.445909132449984E23  # g/s
#BH_Mdot_scaling /= constants.M_sun.to('g').value  # convert to M_sol/s
#BH_Mdot_scaling *= units.yr.to('s')  # convert to M_sol/yr



# ----------------------------------------------------------------------
# --- define quantities to read in [not those for the corner plot, that's done later]

quantities = []
quantities.append({'path': 'Galaxy', 'dataset': 'SFR_aperture/SFR_30/SFR_inst', 'name': 'SFR_inst_30', 'scaling': None})
quantities.append({'path': 'Galaxy', 'dataset': 'Mstar_aperture/Mstar_30', 'name': 'Mstar_30', 'scaling': 1E10})
quantities.append({'path': 'Galaxy', 'dataset': 'BH_Mass', 'name': None, 'scaling': 1E10})
quantities.append({'path': 'Galaxy', 'dataset': 'BH_Mdot', 'name': None, 'scaling': BH_Mdot_scaling})




# ----------------------------------------------------------------------
# --- get all the required quantities, including the weights and delta

qs = {}
d = {}
D = {}
scalings = {}
for tag in tags:

    d[tag] = {}
    D[tag] = {}
    qs[tag] = []
    scalings[tag] = {}

    for Q in quantities:

        if not Q['name']:
            qname = Q['dataset']
        else:
            qname = Q['name']

        d[tag][qname] = fl.load_dataset(Q['dataset'], arr_type=Q['path'])
        D[tag][qname] = np.array([])
        qs[tag].append(qname)
        scalings[tag][qname] = Q['scaling']

# --- read in weights
df = pd.read_csv(f'{flares_dir}/weights_grid.txt')
weights = np.array(df['weights'])
ws = {}
delta = {}

for tag in tags:

    ws[tag] = np.array([])
    delta[tag] = np.array([])

    for ii in range(len(halo)):

        for q in qs[tag]:
            D[tag][q] = np.append(D[tag][q], d[tag][q][halo[ii]][tag])

        ws[tag] = np.append(ws[tag], np.ones(np.shape(d[tag][q][halo[ii]][tag]))*weights[ii])
        delta[tag] = np.append(delta[tag], np.ones(np.shape(d[tag][q][halo[ii]][tag]))*deltas[ii])


for tag in tags:
    for q in qs[tag]:
        if scalings[tag][q]:
            D[tag][q] *= scalings[tag][q]

# ----------------------------------------------
# define new quantities
for i, tag in enumerate(tags):
    D[tag]['log10Mstar_30'] = np.log10(D[tag]['Mstar_30'])
    D[tag]['log10BH_Mdot'] = np.log10(D[tag]['BH_Mdot'])
    D[tag]['log10BH_Mass'] = np.log10(D[tag]['BH_Mass'])
    D[tag]['log10sSFR'] = np.log10(D[tag]['SFR_inst_30'])-np.log10(D[tag]['Mstar_30'])+9

    # ----------------------------------------------
    # define selection
    D[tag]['selection'] = (D[tag]['log10Mstar_30']>log10Mstar_limit)&(D[tag]['log10BH_Mass']>5.5)&\
                          (D[tag]['log10sSFR']>-1)&(D[tag]['log10sSFR']>-1)&(D[tag]['log10BH_Mdot']>-3.5)

    # ----------------------------------------------
    # Print info
    print(f"Total number of galaxies at z={zeds[i]}: {len(ws[tag][D[tag]['selection']])}")

# ----------------------------------------------
# define quantities for including in the plot

properties = ['log10Mstar_30', 'log10BH_Mass', 'log10sSFR', 'log10BH_Mdot']

labels = {}
labels['log10Mstar_30'] = r'\log_{10}({\rm M_{\star}}/{\rm M_{\odot})}'
labels['log10sSFR'] = r'\log_{10}({\rm sSFR}/{\rm Gyr^{-1})}'
labels['beta'] = r'\beta'
labels['log10HbetaEW'] = r'\log_{10}(H\beta\ EW/\AA)'
labels['log10FUV'] = r'\log_{10}(L_{FUV}/erg\ s^{-1}\ Hz^{-1})'
labels['AFUV'] = r'A_{FUV}'
labels['log10BH_Mass'] = r'\log_{10}({\rm M_{SMBH}}/{\rm M_{\odot})}'
labels['log10BH_Mdot'] = r'\log_{10}({\rm \dot{M}_{SMBH}}/{\rm M_{\odot}\ yr^{-1})}'

limits = {}
limits['log10Mstar_30'] = [9.5,11.4]
limits['log10BH_Mass'] = [5.5,9.5]
limits['log10BH_Mdot'] = [-3.5,1.5]
limits['beta'] = [-2.9,-1.1]
limits['log10sSFR'] = [-0.9,1.45]
limits['log10HbetaEW'] = [0.01,2.49]
limits['AFUV'] = [0,3.9]
limits['log10FUV'] = [28.1,29.9]

# ----------------------------------------------
# ----------------------------------------------
# Corner Plot

N = len(properties)

fig = plt.figure(figsize=(12, 5), constrained_layout=False, linewidth=0)

outer = fig.add_gridspec(1, 3, width_ratios=[4, 4, 4], height_ratios=[2])
gs1 = gsc.GridSpecFromSubplotSpec(3,2, subplot_spec=outer[0, 0], wspace=0, hspace=0)
gs2 = gsc.GridSpecFromSubplotSpec(3,2, subplot_spec=outer[0, 1], wspace=0, hspace=0)
gs3 = gsc.GridSpecFromSubplotSpec(3,2, subplot_spec=outer[0, 2], wspace=0, hspace=0)
outer.update(hspace=0.1)
# Populating the axes individually in order to better control

def plot_cell(ax, tag_index, x_name, y_name):
    s = D[tags[tag_index]]['selection']
    x = D[tags[tag_index]][x_name][D[tags[tag_index]]['selection']]
    y = D[tags[tag_index]][y_name][D[tags[tag_index]]['selection']]
    ax.scatter(x, y, s=1, alpha=0.2, color=cmap(norm(zeds[tag_index])))
    # --- weighted median Lines

    bins = np.linspace(*limits[x_name], 10)
    bincen = (bins[:-1] + bins[1:]) / 2.
    out = flares.binned_weighted_quantile(x, y, ws[tags[tag_index]][s], bins, [0.84, 0.50, 0.16])

    N, bins_ = np.histogram(x, bins=bins)
    Ns = (N >= 5)

    ax.plot(bincen, out[:, 1], c='k', ls='-')
    ax.fill_between(bincen[Ns], out[:,0][Ns], out[:,2][Ns], color='k', alpha = 0.3)

    ax.set_xlim(limits[x_name])
    ax.set_ylim(limits[y_name])

labelpad1 = 16
labelpad2 = 11
# leftmost grid
ax101 = plt.subplot(gs1[0])
plot_cell(ax101, 0, properties[0], properties[1])
ax101.get_yaxis().set_visible(True)
ax101.get_xaxis().set_visible(False)
ax101.set_ylabel(rf'$\rm {labels[properties[1]]}$', labelpad=labelpad1)
# empty
ax100 = plt.subplot(gs1[1])
ax100.set_axis_off()
ax100.text(0.5, 0.5, r'$\rm z={0:.0f}$'.format(zeds[0]), fontsize=10, color=cmap(norm(zeds[0])), ha='center')

# top left
ax111 = plt.subplot(gs1[2])
plot_cell(ax111, 0, properties[0], properties[-2])
ax111.get_yaxis().set_visible(True)
ax111.get_xaxis().set_visible(False)
ax111.set_ylabel(rf'$\rm {labels[properties[-2]]}$')
# top right
ax112 = plt.subplot(gs1[3])
plot_cell(ax112, 0, properties[1], properties[-2])
ax112.get_xaxis().set_visible(False)
ax112.get_yaxis().set_visible(False)
# bottom left
ax121 = plt.subplot(gs1[4])
plot_cell(ax121, 0, properties[0], properties[-1])
ax121.get_yaxis().set_visible(True)
ax121.get_yaxis().set_visible(True)
ax121.set_ylabel(rf'$\rm {labels[properties[-1]]}$', labelpad=labelpad2)
ax121.set_xlabel(rf'$\rm {labels[properties[0]]}$')
# bottom right
ax122 = plt.subplot(gs1[5])
plot_cell(ax122, 0, properties[1], properties[-1])
ax122.get_yaxis().set_visible(False)
ax122.get_xaxis().set_visible(True)
ax122.set_xlabel(rf'$\rm {labels[properties[1]]}$')

# middle grid
# leftmost grid
ax201 = plt.subplot(gs2[0])
plot_cell(ax201, 1, properties[0], properties[1])
ax201.get_yaxis().set_visible(True)
ax201.get_xaxis().set_visible(False)
ax201.set_ylabel(rf'$\rm {labels[properties[1]]}$', labelpad=labelpad1)
# empty
ax200 = plt.subplot(gs2[1])
ax200.set_axis_off()
ax200.text(0.5, 0.5, r'$\rm z={0:.0f}$'.format(zeds[1]), fontsize=10, color=cmap(norm(zeds[1])), ha='center')


# top left
ax211 = plt.subplot(gs2[2])
plot_cell(ax211, 1, properties[0], properties[-2])
ax211.get_yaxis().set_visible(True)
ax211.get_xaxis().set_visible(False)
ax211.set_ylabel(rf'$\rm {labels[properties[-2]]}$')
# top right
ax212 = plt.subplot(gs2[3])
plot_cell(ax212, 1, properties[1], properties[-2])
ax212.get_xaxis().set_visible(False)
ax212.get_yaxis().set_visible(False)
# bottom left
ax221 = plt.subplot(gs2[4])
plot_cell(ax221, 1, properties[0], properties[-1])
ax221.get_yaxis().set_visible(True)
ax221.get_yaxis().set_visible(True)
ax221.set_ylabel(rf'$\rm {labels[properties[-1]]}$', labelpad=labelpad2)
ax221.set_xlabel(rf'$\rm {labels[properties[0]]}$')
# bottom right
ax222 = plt.subplot(gs2[5])
plot_cell(ax222, 1, properties[1], properties[-1])
ax222.get_yaxis().set_visible(False)
ax222.get_xaxis().set_visible(True)
ax222.set_xlabel(rf'$\rm {labels[properties[1]]}$')

# rightmost grid
# leftmost grid
ax301 = plt.subplot(gs3[0])
plot_cell(ax301, 2, properties[0], properties[1])
ax301.get_yaxis().set_visible(True)
ax301.get_xaxis().set_visible(False)
ax301.set_ylabel(rf'$\rm {labels[properties[1]]}$', labelpad=labelpad1)
# empty
ax300 = plt.subplot(gs3[1])
ax300.set_axis_off()
ax300.text(0.5, 0.5, r'$\rm z={0:.0f}$'.format(zeds[2]), fontsize=10, color=cmap(norm(zeds[2])), ha='center')


# top left
ax311 = plt.subplot(gs3[2])
plot_cell(ax311, 2, properties[0], properties[-2])
ax311.get_yaxis().set_visible(True)
ax311.get_xaxis().set_visible(False)
ax311.set_ylabel(rf'$\rm {labels[properties[-2]]}$')
#top right
ax312 = plt.subplot(gs3[3])
plot_cell(ax312, 2, properties[1], properties[-2])
ax312.get_xaxis().set_visible(False)
ax312.get_yaxis().set_visible(False)
# bottom left
ax321 = plt.subplot(gs3[4])
plot_cell(ax321, 2, properties[0], properties[-1])
ax321.get_yaxis().set_visible(True)
ax321.get_yaxis().set_visible(True)
ax321.set_ylabel(rf'$\rm {labels[properties[-1]]}$', labelpad=labelpad2)
ax321.set_xlabel(rf'$\rm {labels[properties[0]]}$')
#bottom right
ax322 = plt.subplot(gs3[5])
plot_cell(ax322, 2, properties[1], properties[-1])
ax322.get_yaxis().set_visible(False)
ax322.get_xaxis().set_visible(True)
ax322.set_xlabel(rf'$\rm {labels[properties[1]]}$')

"""
# --- add colourbar

cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
cmapper.set_array([])

cax = fig.add_axes([0.25, 0.87, 0.5, 0.015])
fig.colorbar(cmapper, cax=cax, orientation='horizontal')
cax.set_xlabel(r'$\rm\log_{10}({\rm M_{\star}}/{\rm M_{\odot})}$')
"""


fig.savefig(f'figures/host_gridspec_v2.pdf', bbox_inches='tight')
