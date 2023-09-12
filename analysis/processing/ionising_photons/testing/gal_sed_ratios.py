import numpy as np

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import _pickle as pickle

selection = -1

snapshots = ['005_z010p000', '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000']
snapshot = snapshots[selection]
zeds = [10, 9, 8, 7, 6, 5]


spec_types = ['Pure_Stellar', 'No_ISM', 'DustModelI', 'Intrinsic']
line_types = [':', '-.', '--', '-']

ratios = ['No_ISM', 'Intrinsic']

flares_dir = '../../../../../data/simulations'

for i, snapshot in enumerate(snapshots):

    zed = zeds[i]

    print(zed)

    data = pickle.load(open(f'{flares_dir}/sed_sample_{zed}.p', 'rb'))


    for halo in data.keys():
        fig, axes = plt.subplots(2, 5, figsize=(10, 4), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.07, bottom=0.15, top=1.0, right=0.85, wspace=0.0, hspace=0.0)

        for ii in range(10):
            axes.flatten()[ii].plot(np.log10(data[halo][snapshot]['lam']), np.log10((data[halo][snapshot]['Pure_Stellar'][ii]/data[halo][snapshot]['Intrinsic'][ii])), c='k', ls='-', alpha=0.8)
            axes.flatten()[ii].set_ylim(-5, 5)

        fig.text(0.01, 0.55, r'$\rm log_{10}[F_{\nu}^{Pure\_Stellar} \; / \; F_{\nu}^{Intrinsic}]$', ha='left', va='center',
                 rotation='vertical', fontsize=10)
        fig.text(0.45, 0.05, r'$\rm log_{10}[\lambda\;/\;\AA]$', ha='center', va='bottom', fontsize=10)

        fig.savefig(f'figures/ratios/{snapshot}/pure_stellar__intrinsic/{halo}.pdf', bbox_inches='tight')
        fig.clf()
