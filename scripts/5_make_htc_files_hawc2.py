"""
Use control tuning files to make step HAWC2 files.

This file should create the following htc files:
 * _fX_dY_C[T/P].htc (X files)
"""
from collections import namedtuple
from pathlib import Path
from itertools import product

import numpy as np

from lacbox.io import load_ctrl_txt
from myteampack import MyHTC

#%% file names and pathsROOT = Path(__file__).parent.parent

DESIGN_NAME = "IEC_Ya_Later"

ROOT = Path(__file__).parent.parent  # define root as folder where this script is located

MASTER_FILE = ROOT / 'hawc_files' / '_master' / f'{DESIGN_NAME}.htc'  # your master htc file

RES_DIR = ROOT / 'hawc_files' / 'res_hawc2s_ctrl'  # location of _ctrl_tunint.txt files

HAWC2_HTC_DIR = ROOT / 'hawc_files' / 'htc_hawc2' / 'htc_step'  # where to save the step-wind files

#%% data

# We have to keep the rotational speed to the opt(TSR)*rated_ws/tip_radius
OPT_TSR= 7.2    # To be changed if another one is desired
GEN_RATIO = 50
V_RTD = 11.256  # This is valid for the current design. If R is changed, this needs to change as well
R = 90.879      # Same as V_RTD

max_gen_speed = OPT_TSR*V_RTD*GEN_RATIO*30/(R*np.pi)
omega_rtd_LSS = OPT_TSR*V_RTD / R
min_omega_rtd_LSS= 6*np.pi/30

# step-wind settings
CUTIN, CUTOUT = 4, 25
DT, TSTART = 100, 100

# get a list of all _ctrl_tuning.txt files...
ctrl_tuning_file = namedtuple("ctrl_tuning_file", ["omega", "zeta", "params"])
omega = np.arange(0.03, 0.07, 0.01)   # [0.03 0.04 0.05 0.06, 0.07]
zeta = np.arange(0.5, 0.81, 0.1)     # [0.5 0.6 0.7 0.8]
CPCT  = ["CP", "CT"]                  # 2 values

combinations = list(product(CPCT, omega, zeta))

ctrl_basepath = DESIGN_NAME + \
    "_hawc2s_flex_ctrltune_{CPCT}_f{omega:g}_Z{zeta:g}_ctrl_tuning.txt"

ctrl_tuning_files = [
    ctrl_tuning_file(o, z,
        load_ctrl_txt(Path(RES_DIR, ctrl_basepath.format(CPCT=cpct,
                                                         omega=o,
                                                         zeta=z)))
    )
    for cpct, o, z in combinations
]

#%% Create HTC files
wsp_steps = np.arange(CUTIN+1, CUTOUT+1, 1)
step_times = np.arange(1, len(wsp_steps)+1, 1)*DT + TSTART
t_end = step_times[-1] + DT

for i, ctf in enumerate(ctrl_tuning_files):

    htc = MyHTC(MASTER_FILE)

    htc._update_ctrl_params(ctf.params,
                            rated_rot_speed=omega_rtd_LSS,
                            min_rot_speed=min_omega_rtd_LSS)

    # Nome file customizzato
    append_name = f"_hawc2s_flex_ctrleval_{ ctf[2]['CP/CT']}_f{ctf[0]:.3}_Z{ctf[1]:.2}"

    htc.make_step(save_dir=HAWC2_HTC_DIR,
                  wsp=4,
                  wsp_steps=wsp_steps,
                  step_times=step_times,
                  last_step_len=DT,
                  start_record_time=TSTART,
                  append=append_name)