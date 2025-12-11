"""
This file should Campbell diagram plots for structural and aeroelastic
modal shapes.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from myteampack import MyHTC
from lacbox.io import load_ind, load_pwr, load_cmb, load_amp
from lacbox.vis import plot_amp

#%% plot commands
#size
mpl.rcParams['figure.figsize'] = (16,8)

#font size of label, title, and legend
mpl.rcParams['font.size'] = 25
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['axes.titlesize'] = 25
mpl.rcParams['legend.fontsize'] = 22

#Lines and markers
mpl.rcParams['lines.linewidth'] = 1.6
mpl.rcParams['lines.markersize'] = 9
mpl.rcParams['scatter.marker'] = "+"
plt_marker = "d"

#Latex font
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

#Export
mpl.rcParams['savefig.bbox'] = "tight"

plasma_custom = [
    "#0D0887",  # indigo
    "#7E03A8",  # purple
    "#CB4679",  # pink
    "#F89441",  # orange
    "#FFB300",  # strong orange (L* ≈ 80)
]
cmap_custom = LinearSegmentedColormap.from_list("plasma_custom", plasma_custom)
colors = cmap_custom(np.linspace(0, 1, 4))
#%% Paths
SAVEFIG = True

ROOT = Path(__file__).parent.parent  # define root as folder where this script is located

DESIGN_NAME = "IEC_Ya_Later"

PATH_STR = ROOT / 'hawc_files' / 'res_hawc2s' / 'campbell_diagram_nat_freq_struct'
PATH_STR_AMPL = ROOT / 'hawc_files' / 'res_hawc2s' / 'modal_ampl_nat_freq_struct'
PATH_STR_TOW_RIGID = ROOT / 'hawc_files' / 'res_hawc2s' / 'campbell_diagram_nat_freq_struct_tower_rigid'

PATH_AERO_CMB = ROOT / 'hawc_files' / 'res_hawc2s' / 'IEC_Ya_Later_hawc2s_modal_aeroelastic.cmb'
PATH_AERO_AMP = ROOT / 'hawc_files' / 'res_hawc2s' / 'IEC_Ya_Later_hawc2s_modal_aeroelastic.amp'
PLOT_FLDR_AMPL = Path(ROOT, 'plots', 'campbell_diagrams')
if not PLOT_FLDR_AMPL.is_dir():
    PLOT_FLDR_AMPL.mkdir()

#%% Open data
# unpack structural modes and amplitudes
wsp_str, freq_str, zeta_str = load_cmb(PATH_STR, cmb_type='structural')
wsp_str, freq_str_tow_rigid, zeta_st_tow_rigidr  = load_cmb(PATH_STR_TOW_RIGID,
                                                            cmb_type='structural')
amplitudes = load_amp(PATH_STR_AMPL)
rot_speed = wsp_str*(30/np.pi) # rotor speed [RPM]

# unpack aeroelastic modes and amplitudes
ws_des_flex, pitch_des_flex, omega_des_flex, power_des_flex,\
    thrust_des_flex = np.loadtxt(ROOT / 'hawc_files' / 'data' / f'{DESIGN_NAME}_flex.opt',
                                 skiprows=1, unpack=True)

wsp_aero, freq_aero, zeta_aero = load_cmb(PATH_AERO_CMB,
                           'aeroelastic')
amp = load_amp(PATH_AERO_AMP)

#%% standstill frequencies and theoretical values for structural nat. frequencies

# standistill nat. freq. for tower and blade (tower, 1st flap, 1st edge, 2nd flap, 2nd edge)
standstill_freq = [0.25, 0.60, 0.90, 1.72, 2.70]

# simple method for natural frequencies (tower, 1st flap, 1st edge, 2nd flap, 2nd edge)
simple_method = np.zeros_like(freq_str)
simple_method[:,0] = standstill_freq[0] * np.ones_like(freq_str[:,0])
simple_method[:,1] = standstill_freq[0] * np.ones_like(freq_str[:,0])
simple_method[:,2] = standstill_freq[1] * np.ones_like(freq_str[:,0]) - rot_speed/60
simple_method[:,3] = standstill_freq[1] * np.ones_like(freq_str[:,0])
simple_method[:,4] = standstill_freq[1] * np.ones_like(freq_str[:,0]) + rot_speed/60
simple_method[:,5] = standstill_freq[2] * np.ones_like(freq_str[:,0]) - rot_speed/60
simple_method[:,6] = standstill_freq[2] * np.ones_like(freq_str[:,0]) + rot_speed/60
simple_method[:,7] = standstill_freq[2] * np.ones_like(freq_str[:,0])
simple_method[:,8] = standstill_freq[3] * np.ones_like(freq_str[:,0]) + rot_speed/60
simple_method[:,9] = standstill_freq[3] * np.ones_like(freq_str[:,0])
simple_method[:,10] = standstill_freq[3] * np.ones_like(freq_str[:,0]) - rot_speed/60

#%% swap of modal frequencies that overlap each other

# 1st flap sym and 1st flap fw
i_FW = 3      # 1st FW flap
i_SYM = 4     # 1st sym. flap
idx = 6       # punto in cui si scambiano (7° punto)

# swap for frequencies that change one in the other
temp = freq_str[idx:, i_FW].copy()
freq_str[idx:, i_FW] = freq_str[idx:, i_SYM]
freq_str[idx:, i_SYM] = temp

# 1st flap fw and 1st edge bw
i_FW = 3      # 1st FW flap
i_BW = 5     # 1st sym. flap
idx = -1       # punto in cui si scambiano (7° punto)

# swap for frequencies that change one in the other
temp = freq_str[idx, i_FW].copy()
freq_str[idx, i_FW] = freq_str[idx, i_BW]
freq_str[idx, i_BW] = temp

temp = zeta_str[idx, i_FW].copy()
zeta_str[idx, i_FW] = zeta_str[idx, i_BW]
zeta_str[idx, i_BW] = temp
#%% mode names and labels
nmodes = np.size(freq_aero,axis=1)
modenames = ['1st FA tower', '1st STS tower', '1st BW flap', '1st FW flap', '1st sym. flap',
              '1st BW edge', '1st FW edge', '2nd BW flap', '2nd FW flap', '2nd sym. flap',
              '1st sym. edge']
modenames_aero = ['1st FA tower', '1st STS tower', '1st BW flap', '1st sym. flap', '1st FW flap',
              '1st BW edge', '1st FW edge', '2nd BW flap', '2nd FW flap', '2nd sym. flap',
              '1st sym. edge']

omega = np.vstack([omega_des_flex * i / 60 for i in [1, 3, 6]]).T

#colors for simple method
colors_simple = [colors[0],colors[0],colors[1],colors[1],colors[1],colors[2],
                 colors[2],colors[2],colors[3],colors[3],colors[3]]

def modecolor(modename):
    if 'tower' in modename:
        return colors[0]
    elif 'flap' in modename and '1st' in modename:
        return colors[1]
    elif 'edge' in modename:
        return colors[2]
    elif 'flap' in modename and '2nd' in modename:
        return colors[3]
    else:
        return 'k'

def marker(modename):
    if '1st FA tower' in modename:
        return 'o'
    elif '1st STS tower' in modename:
        return 's'
    elif 'BW' in modename:
        return '<'
    elif 'FW' in modename:
        return '>'
    elif 'sym' in modename:
        return 'o'
    else:
        return '+'

#%% structural modes plot
fig, axs = plt.subplots(1,2,figsize=(20,8))

for i in range(nmodes):
    axs[0].plot(rot_speed, freq_str[:, i], marker=marker(modenames[i]),
                color=modecolor(modenames[i]), label=modenames[i])
    
    axs[0].plot(rot_speed,simple_method[:, i], marker=None,linestyle='--',
                color=colors_simple[i],linewidth=1.5,label='_nolegend_')
    
    axs[1].plot(rot_speed, zeta_str[:, i], marker=marker(modenames[i]),
                color=modecolor(modenames[i]))

axs[0].set(xlabel='Rotational speed [RPM]', ylabel='Damped nat. frequencies [Hz]')
axs[0].grid(True, which='major', zorder=3, alpha=0.5)
axs[0].grid(True, which='minor', zorder=3, alpha=0.3) 
axs[0].set_xlim([rot_speed[0],rot_speed[-1]])
axs[0].minorticks_on()
axs[0].tick_params(direction='in',right=True,top =True)
axs[0].tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
axs[0].tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
axs[0].tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)

axs[1].set(xlabel='Rotational speed [RPM]', ylabel='Modal damping [% critical]')
axs[1].grid(True, which='major', zorder=3, alpha=0.5)
axs[1].grid(True, which='minor', zorder=3, alpha=0.3)  
axs[1].set_xlim([rot_speed[0],rot_speed[-1]])
axs[1].minorticks_on()
axs[1].tick_params(direction='in',right=True,top =True)
axs[1].tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
axs[1].tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
axs[1].tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)

fig.legend(loc='upper center', ncols=6, framealpha=0,fontsize=20,
           labels=[*modenames],
           bbox_to_anchor=(0.5, 1))
plt.subplots_adjust(top=0.7)
plt.tight_layout(rect=[0, 0, 1, 0.9])
if SAVEFIG:
    plt.savefig(PLOT_FLDR_AMPL / 'structural_modes.pdf')

#%% aeroealstic modes plot
fig, axs = plt.subplots(1,2,figsize=(20,8))

for i in range(nmodes):
    axs[0].plot(wsp_aero, freq_aero[:, i], marker=marker(modenames_aero[i]),
                color=modecolor(modenames_aero[i]), label=modenames_aero[i])
    
    axs[1].plot(wsp_aero, zeta_aero[:, i], marker=marker(modenames_aero[i]),
                color=modecolor(modenames_aero[i]))

axs[0].plot(wsp_aero, omega, color='k')
axs[0].set(xlabel='Wind speed [m/s]', ylabel='Damped nat. frequencies [Hz]')
axs[0].grid(True, which='major', zorder=3, alpha=0.5)
axs[0].grid(True, which='minor', zorder=3, alpha=0.3) 
axs[0].set_xlim([wsp_aero[0],wsp_aero[-1]])
axs[0].minorticks_on()
axs[0].tick_params(direction='in',right=True,top =True)
axs[0].tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
axs[0].tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
axs[0].tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)

axs[1].set(xlabel='Wind speed [m/s]', ylabel='Modal damping [% critical]')
axs[1].grid(True, which='major', zorder=3, alpha=0.5)
axs[1].grid(True, which='minor', zorder=3, alpha=0.3)
axs[1].set_xlim([wsp_aero[0],wsp_aero[-1]])
axs[1].minorticks_on()
axs[1].tick_params(direction='in',right=True,top =True)
axs[1].tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
axs[1].tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
axs[1].tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)

fig.legend(loc='upper center', ncols=6, framealpha=0,fontsize=20,
           labels=[*modenames, "1P, 3P, 6P"],
           bbox_to_anchor=(0.5, 1))
plt.subplots_adjust(top=0.7)
plt.tight_layout(rect=[0, 0, 1, 0.9])
if SAVEFIG:
    plt.savefig(PLOT_FLDR_AMPL / 'aeroelastic_modes.pdf')

