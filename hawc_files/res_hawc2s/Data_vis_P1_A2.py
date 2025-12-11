"""
    This file plots the results of Part 1 of Assignment 2 of LAC
"""

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from lacbox.io import load_cmb

filepath = Path.cwd()

cmb_struct_wn = load_cmb(filepath / 'campbell_diagram_nat_freq_struct.cmb',
                         cmb_type='structural')

rot_speed = cmb_struct_wn[0]*(30/np.pi)  # In RPM
labels = ["Tower FA", "Tower StS", "1st BW flap", "1st Sym flap", "1st FW flap",
          "1st BW edge", "1st FW edge", "2nd BW flap", "2nd FW flap", "1st Sym edge",
          "2nd Sym flap"]
markers = ["x", "x", "<", "o", ">", "<", ">", "<", ">", "o", "o"]

fig, axs = plt.subplots(1,2,figsize=(20,8))
for i in range(len(cmb_struct_wn[1][0,:])):
    for j in range(2):
        ax = axs[j]
        ax.plot(rot_speed,cmb_struct_wn[j+1][:, i], label=labels[i], marker=markers[i])
        ax.grid()
    ax.set_xlabel('Rotational speed [RPM]')
    if j==0:
        ax.set_ylabel('Damped nat. frequencies [Hz]')
    else:
        ax.set_ylabel('Modal damping [% critical]')

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=5)
