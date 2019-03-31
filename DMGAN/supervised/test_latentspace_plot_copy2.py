"""
Author : Nianzu Ethan Zheng
Datetime : 2018-3-26
Place: Shenyang China
Copyright
For Mutal information Metrics
"""
import os, sys
from model import config
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.gridspec as gridspec

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
from data_providers.data_provider import SuvsDataProvider
from plot import Visualizer
from sequential.utils import pickle_load

vis = Visualizer()
dp = SuvsDataProvider(num_validation=config.num_vad, shuffle='every_epoch')
config.is_train = False
config.batch_size = dp.valid.num_sample
test_times = 60

# Gaussian mixture
description1 = 'logs/hybrid_GAN_lin_res-Dz=0.01-R=1-Lat=1.5-Tv=0.01-d-gm-bc-gs-2018-02-04-metric_plus.pkl'
[q_errors1, r_adjs1, z_adjs1, z_true1], name1 = pickle_load(description1)

# Swiss Roll
description2 = 'logs/hybrid_GAN_lin_res-Dz=0.01-R=1-Lat=1.5-Tv=0.01-d-sr-bc-gs-2018-02-04-metric_plus.pkl'
[q_errors2, r_adjs2, z_adjs2, z_true2], name2 = pickle_load(description2)

# Uniform Desk
description3 = 'logs/hybrid_GAN_lin_res-Dz=0.01-R=1-Lat=1.5-Tv=0.01-d-ud-bc-gs-2018-02-04-metric_plus.pkl'
[q_errors3, r_adjs3, z_adjs3, z_true3], name3 = pickle_load(description3)

# Uniform Square
description4 = 'logs/hybrid_GAN_lin_res-Dz=0.01-R=1-Lat=1.5-Tv=0.01-d-us-bc-gs-2018-02-04-metric_plus.pkl'
[q_errors4, r_adjs4, z_adjs4, z_true4], name4 = pickle_load(description4)

# Gaussian
description5 = 'logs/hybrid_GAN_lin_res-Dz=0.01-R=1-Lat=1.5-Tv=0.01-d-gs-bc-gs-2018-02-04-metric_plus.pkl'
[q_errors5, r_adjs5, z_adjs5, z_true5], name5 = pickle_load(description5)

# Load data
# latent space
from cycler import cycler

plt.style.use('classic-znz')
mpl.rcParams['lines.linewidth'] = 1
mpl.rc('legend', labelspacing=0.05, fontsize='medium')
mpl.rcParams['legend.labelspacing'] = 0.05
mpl.rc('axes', prop_cycle=cycler(
    'color', ['#8EBA42', '#988ED5', '#e5ae38', '#6d904f', '#8b8b8b', '#810f7c']
))
                                                                            # E24A33 : red
                                                                            # 348ABD : blue
                                                                            # 988ED5 : purple
                                                                            # 777777 : gray
                                                                            # FBC15E : yellow
                                                                            # 8EBA42 : green
                                                                            # FFB5B8 : pink
mpl.rc('xtick', labelsize='small')


# small, medium, large, x-large, xx-large, larger, or smaller

# FOR TITLE
def animate_name(axs, tt):
    axs.set_title(label=tt, fontsize='medium', color='black', weight='semibold', loc='center')


# FOR SUB-INDEX
def animate(axs, tt):
    ylim = axs.get_ylim()
    xlim = axs.get_xlim()
    axs.text(xlim[0] + 0.5 * (xlim[1] - xlim[0]), ylim[0] - 0.15 * (ylim[1] - ylim[0]), "({})".format(tt),
             ha='center', va='bottom', weight='normal', color='black', fontsize=15)


# SUBPLOTS UTILIZATION
fig = plt.figure(figsize=(18, 12), dpi=500)
gs = gridspec.GridSpec(2, 6)
                                                       # gs.update(wspace=0.4, hspace=0.3)
ax1 = plt.subplot(gs[0, 0:2])
ax2 = plt.subplot(gs[0, 2:4])
ax3 = plt.subplot(gs[0, 4:6])
ax4 = plt.subplot(gs[1, 1:3])
ax5 = plt.subplot(gs[1, 3:5])


params = {"c": '#348ABD', "s": 30, "alpha": 0.9, 'marker': 'o', 'edgecolors': 'none'}
params_another = {"color": '#8EBA42', 'alpha': 0.9, 'linestyle': '--', "lw": 0.9,
                  'marker': 's', 'mec': 'k', "markersize": 6, "mew": 0.6,
                  }

# BELOW SEVEN OPERATIONAL INDICES
ax1.scatter(z_adjs1[:, 0], z_adjs1[:, 1], **params)
ax1.plot(z_true1[:, 0], z_true1[:, 1], **params_another)
ax1.set(xlabel="Latent Variable $Z_1$", ylabel="Latent Variable $Z_2$", xlim=[-4, 4], ylim=[-4, 4])
animate(ax1, 'a')
animate_name(ax1, 'Gaussian Mixture')

ax2.scatter(z_adjs2[:, 0], z_adjs2[:, 1], **params)
ax2.plot(z_true2[:, 0], z_true2[:, 1], **params_another)
ax2.set(xlabel="Latent Variable $Z_1$", ylabel="Latent Variable $Z_2$", xlim=[-4, 4], ylim=[-4, 4])
animate(ax2, 'b')
animate_name(ax2, 'Swiss Roll')

ax3.scatter(z_adjs3[:, 0], z_adjs3[:, 1], **params)
ax3.plot(z_true3[:, 0], z_true3[:, 1], **params_another)
ax3.set(xlabel="Latent Variable $Z_1$", ylabel="Latent Variable $Z_2$", xlim=[-3, 3], ylim=[-3, 3])
animate(ax3, 'c')
animate_name(ax3, 'Circle Uniform')

ax4.scatter(z_adjs4[:, 0], z_adjs4[:, 1], **params)
ax4.plot(z_true4[:, 0], z_true4[:, 1], **params_another)
ax4.set(xlabel="Latent Variable $Z_1$", ylabel="Latent Variable $Z_2$", xlim=[-3, 3], ylim=[-3, 3])
animate(ax4, 'd')
animate_name(ax4, 'Standard Uniform')

ax5.scatter(z_adjs5[:, 0], z_adjs5[:, 1], label='Latent space operating point', **params, )
ax5.plot(z_true5[:, 0], z_true5[:, 1], label='Actual Mapping point', **params_another)
ax5.set(xlabel="Latent Variable $Z_1$", ylabel="Latent Variable $Z_2$", xlim=[-3, 3], ylim=[-3, 3])
animate(ax5, 'e')
animate_name(ax5, 'Gaussian')
ax5.legend(loc='lower left', bbox_to_anchor=(1.04, 0))

gs.tight_layout(fig=fig, pad=2, w_pad=2, h_pad=2.5)
# C:\Users\CYD\Desktop\IAI_Conv\supervised
plt.savefig('{}/{}.svg'.format('./historical_records', "Latent_Space_track_center"))
# plt.show()
