import numpy as np
import pandas as pd

import astropy.constants as const
import astropy.units as u

import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')
import matplotlib.pyplot as plt


from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import flares
import FLARE.plt as fplt

import matplotlib.gridspec as gsc

cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=42, vmax=47)

def l_agn(m_dot, etta=0.1):
    m_dot = (m_dot*u.M_sun/u.yr).to(u.kg / u.s) # accretion rate in SI
    c = const.c #speed of light
    etta = etta
    l = (etta*m_dot*c**2).to(u.erg/u.s)
    return np.log10(l.value) # output in log10(erg/s)

# ----------------------------------------------------------------------
# --- open data

deltas = np.array([0.969639,0.918132,0.851838,0.849271,0.845644,0.842128,0.841291,0.83945,0.838891,0.832753,0.830465,0.829349,0.827842,0.824159,0.821425,0.820476,0.616236,0.616012,0.430745,0.430689,0.266515,0.266571,0.121315,0.121147,-0.007368,-0.007424,-0.121207,-0.121319,-0.222044,-0.222156,-0.311441,-0.311329,-0.066017,-0.066185,-0.00748,-0.007424,0.055076,0.054909,-0.47874,-0.433818])

flares_dir = '../../../data/simulations'

fl = flares.flares(f'{flares_dir}/flares_no_particlesed.hdf5', sim_type='FLARES')

halo = fl.halos



# ----------------------------------------------------------------------
# --- define parameters

tags = fl.tags  # --- select tag -3 = z=7
zeds = fl.zeds
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
    D[tag]['log10L_agn'] = l_agn(D[tag]['BH_Mdot'])

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

fig = plt.figure(figsize=(4, 5), constrained_layout=False, linewidth=0)

outer = fig.add_gridspec(1, 1, width_ratios=[1], height_ratios=[1])
gs1 = gsc.GridSpecFromSubplotSpec(3,2, subplot_spec=outer[0, 0], wspace=0, hspace=0)
# Populating the axes individually in order to better control

def plot_cell(ax, tag_index, x_name, y_name):
    s = D[tags[tag_index]]['selection']
    x = D[tags[tag_index]][x_name][D[tags[tag_index]]['selection']]
    y = D[tags[tag_index]][y_name][D[tags[tag_index]]['selection']]
    c = D[tags[tag_index]]['log10L_agn'][D[tags[tag_index]]['selection']]

    ax.scatter(x, y, s=1, alpha=0.3, color=cmap(norm(c)))
    # --- weighted median Lines

    bins = np.linspace(*limits[x_name], 10)
    bincen = (bins[:-1] + bins[1:]) / 2.
    out = flares.binned_weighted_quantile(x, y, ws[tags[tag_index]][s], bins, [0.84, 0.50, 0.16])

    N, bins_ = np.histogram(x, bins=bins)
    Ns = (N >= 5)

    ax.plot(bincen, out[:, 1], c='k', ls='-')
    ax.fill_between(bincen[Ns], out[:,0][Ns], out[:,2][Ns], color='k', alpha = 0.2)

    ax.set_xlim(limits[x_name])
    ax.set_ylim(limits[y_name])

labelpad1 = 16
labelpad2 = 11
# leftmost grid
ax101 = plt.subplot(gs1[0])
plot_cell(ax101, -1, properties[0], properties[1])
ax101.get_yaxis().set_visible(True)
ax101.get_xaxis().set_visible(False)
ax101.set_ylabel(rf'$\rm {labels[properties[1]]}$', labelpad=labelpad1)
# empty
ax100 = plt.subplot(gs1[1])
ax100.set_axis_off()
ax100.text(0.5, 0.5, r'$\rm z={0:.0f}$'.format(zeds[-1]), fontsize=10, color='k', ha='center')

# top left
ax111 = plt.subplot(gs1[2])
plot_cell(ax111, -1, properties[0], properties[-2])
ax111.get_yaxis().set_visible(True)
ax111.get_xaxis().set_visible(False)
ax111.set_ylabel(rf'$\rm {labels[properties[-2]]}$')
# top right
ax112 = plt.subplot(gs1[3])
plot_cell(ax112, -1, properties[1], properties[-2])
ax112.get_xaxis().set_visible(False)
ax112.get_yaxis().set_visible(False)
# bottom left
ax121 = plt.subplot(gs1[4])
plot_cell(ax121, -1, properties[0], properties[-1])
ax121.get_yaxis().set_visible(True)
ax121.get_yaxis().set_visible(True)
ax121.set_ylabel(rf'$\rm {labels[properties[-1]]}$', labelpad=labelpad2)
ax121.set_xlabel(rf'$\rm {labels[properties[0]]}$')
# bottom right
ax122 = plt.subplot(gs1[5])
plot_cell(ax122, -1, properties[1], properties[-1])
ax122.get_yaxis().set_visible(False)
ax122.get_xaxis().set_visible(True)
ax122.set_xlabel(rf'$\rm {labels[properties[1]]}$')

y_loc = ax122.get_yaxis()
x_loc = ax122.get_xaxis()

# middle grid
# leftmost grid

cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
cmapper.set_array([])

cax = inset_axes(ax122, # here using axis of the lowest plot
               width="10%",  # width = 5% of parent_bbox width
               height="300%",  # height : 340% good for a (4x4) Grid
               loc='lower left',
               bbox_to_anchor=(1.05, 0., 1, 1),
               bbox_transform=ax122.transAxes,
               borderpad=0,
               )
#cax = fig.add_axes([0.7, 0.1, 0.1, .9])

fig.colorbar(cmapper, cax=cax, orientation='vertical')
cax.set_ylabel(r'$\rm\log_{10}[{\rm L_{AGN, bol}}\,/\,{\rm erg \, s^{-1}]}$')



fig.savefig(f'figures/host_gridspec_v3.pdf', bbox_inches='tight')
