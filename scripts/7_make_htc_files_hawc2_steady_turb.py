"""Make steady or turbulent htc files for cluster.

Steady wind settings:
    * No shear, tower shadow method is 0, turb_format is 0

Turbulence settings:
    * Power-law shear with 0.2 exponent, tower shadow method is 3,
      turb_format is 1, etc.
"""
from pathlib import Path
import random

import numpy as np

from lacbox.htc import _clean_directory
from lacbox.io import load_ctrl_txt
from myteampack import MyHTC

# Select what to create
create_steady = False
create_turb = True
model = "IEC_Ya_Later"  # "dtu_10mw" or "IEC_Ya_Later"

# define folders
ROOT = Path(__file__).parent.parent
DESIGN_NAME = model
MASTER_FILE = ROOT / 'hawc_files' / '_master' / f'{DESIGN_NAME}.htc'  # your master htc file
OPT_PATH = ROOT / 'hawc_files' / 'data' / f'{DESIGN_NAME}_flex.opt'
DEL_HTC_DIR = True  # delete htc directory if it already exists?

# settings for both steady and turbulence
WSPS = range(5, 25)

# ---------- STEADY WIND ----------
if create_steady:
    STEADY_HTC_DIR =  ROOT / 'hawc_files' / 'htc_steady' / f'{DESIGN_NAME}'  # where to save the step-wind files
    RES_DIR = ROOT / 'hawc_files' / 'res_steady' / f'{DESIGN_NAME}'
    CASES = ['tilt', 'notilt', 'notiltrigid', 'notiltnodragrigid']  # ['tilt', 'notilt', 'notiltrigid', 'notiltnodragrigid']
    TIME_START = 200
    TIME_STOP = 400

    # clean the top-level htc directory if requested
    _clean_directory(STEADY_HTC_DIR, DEL_HTC_DIR)

    # make the steady wind files
    for case in CASES:
        TILT = None  # default: don't change tilt
        RIGID = False  # default: flexible blades and tower
        WITHDRAG = True
        if 'notilt' in case:
            TILT = 0
        if 'rigid' in case:
            RIGID = True
        if 'nodrag' in case:
            WITHDRAG = False
        # generate the files
        for wsp in WSPS:
            append = f'_steady_{case}_{wsp:04.1f}'  # fstring black magic! zero-fill with 1 decimal: e.g., '_steady_05.0'
            htc = MyHTC(MASTER_FILE)
            htc.make_steady(STEADY_HTC_DIR, wsp, append, opt_path=OPT_PATH, resdir=RES_DIR,
                            tilt=TILT, subfolder=case, rigid=RIGID, withdrag=WITHDRAG,
                            time_start=TIME_START, time_stop=TIME_STOP)

# ---------- TURBULENT WIND ----------

if create_turb:
    TURB_HTC_DIR = ROOT / 'hawc_files' / 'htc_turb' / f'{DESIGN_NAME}'  # where to save the step-wind files
    RES_DIR = ROOT / 'hawc_files' / 'res_turb' / f'{DESIGN_NAME}'
    CASES = ['tca', 'tcb']
    TI_REF = {"tca": .16, "tcb": .14}  # Turbulence intensities for IEC A & B
    TIME_START = 100
    TIME_STOP = 700
    START_SEED = 42  # initialize the random-number generator for reproducability
    NSEEDS = 6

    if model == "IEC_Ya_Later":
        # Rated speed
        rpm_rtd_HSS = 425.81 # HSS rated rotational speed [rpm]
        rpm_min_HSS = 300  # HSS minimum rotational speed [rpm]
        n_gear = 50  # Gear ratio
        omega_rtd_LSS = rpm_rtd_HSS/n_gear*2*np.pi/60
        omega_min_LSS = rpm_min_HSS/n_gear*2*np.pi/60

        # Load final controller tuning (C1 - omega=0.06, zeta=0.8, constant power)
        ctrl_path = Path(ROOT,'hawc_files', "res_hawc2s_ctrl", DESIGN_NAME +
                         "_hawc2s_flex_ctrltune_CT_f0.05_Z0.8_ctrl_tuning.txt")
        ctrl_tuning_file = load_ctrl_txt(ctrl_path)

    random.seed(START_SEED)

    # clean the top-level htc directory if requested
    _clean_directory(TURB_HTC_DIR, DEL_HTC_DIR)

    # Make the turbulent wind files
    for idx_seed in range(6):
        for wsp in WSPS:
            sim_seed = random.randrange(int(2**16))
            for tc in CASES:
                htc = MyHTC(MASTER_FILE)

                if model == "IEC_Ya_Later":
                    # update controller block in htc file
                    htc._update_ctrl_params(ctrl_tuning_file,
                                            min_rot_speed=omega_min_LSS,
                                            rated_rot_speed=omega_rtd_LSS)

                htc.make_turb(wsp=wsp, ti=TI_REF[tc]*(0.75*wsp+5.6)/wsp,
                              seed=sim_seed,
                              append=f"_turb_{tc}_{wsp:04.1f}_{sim_seed:d}",
                              save_dir=TURB_HTC_DIR, subfolder=tc,
                              opt_path=OPT_PATH, resdir=RES_DIR,
                              dy=190, dz=190,
                              rigid=False, withdrag=True, tilt=None,
                              time_start=TIME_START, time_stop=TIME_STOP)
