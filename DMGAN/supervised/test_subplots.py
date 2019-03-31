"""
Author : Nianzu Ethan Zheng
Datetime : 2018-2-2
Place: Shenyang China
Copyright
For Visualization of training process and for Comparision of the different parameters
"""
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
from plot import Visualizer
import numpy as np
from sequential.utils import pickle_load
from base_options import BaseConfig
import matplotlib.pyplot as plt
import matplotlib as mpl

vis = Visualizer()
opt = BaseConfig()

# load data
# path = os.path.join(opt.logs_path, opt.description + '-test.pkl')
path = 'logs/hybrid_GAN_cc_res-Dz=0.01-R=1-Lat=1.5-Tv=0.01-d-gs-bc-gs-2018-02-04-test.pkl'
# path = 'logs/hybrid_GAN_cc_vgg-Dz=0.01-R=10-Lat=0.5-Tv=0.01-x-gs-bc-gs-2018-02-04-test.pkl'
# path = 'logs/hybrid_GAN_cc_res-Dz=0.01-R=10-Lat=0.5-Tv=0.01-x-gs-bc-gs-2018-02-04-test.pkl'
d, _ = pickle_load(path)
t = np.arange(0, len(d["actual_Q"][:, 0])) + 1

"""
Plot 3x3 Sub-figures
"""
# GLOBAL SETTING
# FILE PATH: C:\Program Files\Anaconda3\Lib\site-packages\matplotlib\mpl-data\stylelib
plt.style.use('classic-znz')
# mpl.rcParams['lines.linewidth'] = 1
mpl.rc('lines', linewidth=1, markersize=4)
mpl.rc('font', family='Times New Roman')
mpl.rcParams['legend.labelspacing'] = 0.05


def animate(axs, tt):
    ylim = axs.get_ylim()
    axs.text(-10, ylim[-1] + 0.01 * (ylim[1] - ylim[0]), tt,
             ha='center', va='bottom', weight='normal', color='black', fontsize=17)


def animate_name(axs, tt):
    ylim = axs.get_ylim()
    axs.text(3, ylim[0] + 0.9 * (ylim[1] - ylim[0]), tt,
             ha='left', va='bottom', weight='bold', color='black', fontsize="medium", zorder=30)


# SUBPLOTS UTILIZATION
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(4*5, 2*5), dpi=80)
clm_ac = '-*'
clm_rn = '-s'

## BELOW SEVEN OPERATIONAL INDICES
axes[0, 0].plot(t, d["actual_r"][:, 0], clm_ac, t, d["result_r"][:, 0], clm_rn)
axes[0, 0].set(xlabel="Sample number", ylabel='Magnetic tube recovery rate (%)')
axes[0, 0].legend(["Primitive target", "Renewed target"], loc='lower right')
animate(axes[0, 0], 'a')
animate_name(axes[0, 0], 'MTRR')

axes[0, 1].plot(t, d["actual_r"][:, 1], clm_ac, t, d["result_r"][:, 1], clm_rn)
axes[0, 1].set(xlabel="Sample number", ylabel="Particle size of LMPL (%)")
axes[0, 1].legend(["Primitive target", "Renewed target"], loc='lower right')
animate(axes[0, 1], 'b')
animate_name(axes[0, 1], 'PSLMPL')

axes[0, 2].plot(t, d["actual_r"][:, 2], clm_ac, t, d["result_r"][:, 2], clm_rn)
axes[0, 2].set(xlabel="Sample number", ylabel="Concentrate grade of LMPL (%)")
axes[0, 2].legend(["Primitive target", "Renewed target"], loc='lower right')
animate(axes[0, 2], 'c')
animate_name(axes[0, 2], 'CGLMPL')

axes[1, 0].plot(t, d["actual_r"][:, 3], clm_ac, t, d["result_r"][:, 3], clm_rn)
axes[1, 0].set(xlabel="Sample number", ylabel="Tailing grade of LMPL (%)")
axes[1, 0].legend(["Primitive target", "Renewed target"], loc='lower right')
animate(axes[1, 0], 'd')
animate_name(axes[1, 0], 'TGLMPL')

axes[1, 1].plot(t, d["actual_r"][:, 4], clm_ac, t, d["result_r"][:, 4], clm_rn)
axes[1, 1].set(xlabel="Sample number", ylabel="Particle size of HMPL (%)")
axes[1, 1].legend(["Primitive target", "Renewed target"], loc='lower right')
animate(axes[1, 1], 'e')
animate_name(axes[1, 1], 'PSHMPL')

axes[1, 2].plot(t, d["actual_r"][:, 5], clm_ac, t, d["result_r"][:, 5], clm_rn)
axes[1, 2].set(xlabel="Sample number", ylabel="Tailing grade of HMPL (%)")
axes[1, 2].legend(["Primitive target", "Renewed target"], loc='lower right')
animate(axes[1, 2], 'f')
animate_name(axes[1, 2], 'TGHMPL')

axes[2, 0].plot(t, d["actual_r"][:, 6], clm_ac, t, d["result_r"][:, 6], clm_rn)
axes[2, 0].set(xlabel="Sample number", ylabel="Concentrate grade of HMPL (%)")
axes[2, 0].legend(["Primitive target", "Renewed target"], loc='lower right')
animate(axes[2, 0], 'g')
animate_name(axes[2, 0], 'CGHMPL')

## BELOW TWO PRODUCTION INDICES
axes[2, 1].plot(t, d["actual_Q"][:, 1], clm_ac, t, d["result_Q"][:, 1], clm_rn)
axes[2, 1].axhline(y=52.6, lw=1, c='r', ls='--')
axes[2, 1].set(xlabel="Sample number", ylabel="The mixed concentrate grade (%)")
axes[2, 1].legend(["The existing system", "The proposed method", 'Target grade'], loc='lower right')
animate(axes[2, 1], 'h')
animate_name(axes[2, 1], 'Grade')
axes[2, 1].annotate(s='52.6%', xy=(36, 52.6), xycoords='data',
                    xytext=(30, 53.3), textcoords='data', weight='semibold', color='red', size="medium",
                    arrowprops=dict(arrowstyle='fancy',
                                    connectionstyle='arc3',
                                    color='red'))

axes[2, 2].plot(t, d["actual_Q"][:, 0], clm_ac, t, d["result_Q"][:, 0], clm_rn)
axes[2, 2].axhline(y=6868.5, lw=1, c='r', ls='--')
axes[2, 2].set(xlabel="Sample number", ylabel="Yield of concentrated ore (t/d)")
axes[2, 2].legend(["The existing system", "The proposed method", 'Target yield'], loc='lower right')
animate(axes[2, 2], 'i')
animate_name(axes[2, 2], 'Yield')
axes[2, 2].annotate(s='6868.5t/d', xy=(50, 6868.5), xycoords='data',
                    xytext=(50, 5500), textcoords='data', weight='semibold', color='red', size="medium",
                    arrowprops=dict(arrowstyle='fancy',
                                    connectionstyle='arc3',
                                    color='red'))

fig.tight_layout(pad=2, w_pad=1.3, h_pad=1.5)

plt.savefig('{}/{}.png'.format('./historical_records', "nine-grid"))
#plt.show()