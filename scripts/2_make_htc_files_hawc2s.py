"""Make HAWC2S files from a master htc file.

This file should create the following htc files:
 * _hawc2s_1wsp.htc
 * _hawc2s_multitsr.htc
 * _hawc2s_rigid.htc
 * _hawc2s_flex.htc
 * _hawc2s_ctrltune_fX_dY_C[T/P].htc (7 files)
 * ...and more?

We recommend saving the HAWC2S files in a dedicated subfolder. If you
do this, note that you will need to move the htc before running HAWCStab2.

Requires myteampack (which requires lacbox).
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from myteampack import MyHTC
from lacbox.io import load_ind, load_pwr, load_st

#%% Booleans

MAKE_HTC = True
MAKE_HTC_CTRL = False

save_modal_amp = False

#%% file names and paths
# Paths
DESIGN_NAME = "IEC_Ya_Later"
DTU_NAME = 'dtu_10mw'

ROOT = Path(__file__).parent.parent  # define root as folder where this script is located

MASTER_FILE = ROOT / 'hawc_files' / '_master' / f'{DESIGN_NAME}.htc'  # your master htc file
MASTER_FILE_DTU = ROOT / 'hawc_files' / '_master' / f'{DTU_NAME}.htc'  # your master htc file

TARGET_DIR = ROOT / 'hawc_files' / 'htc_hawc2s'  # where to save the htc files this script will make
TARGET_DIR_CTRL = ROOT / 'hawc_files' / 'htc_hawc2s_ctrl'  # where to save the htc files this script will make

#%% data

# We have to keep the rotational speed to the opt(TSR)*rated_ws/tip_radius
OPT_TSR= 7.2    # To be changed if another one is desired
GEN_RATIO = 50
V_RTD = 11.256  # This is valid for the current design. If R is changed, this needs to change as well
R = 90.879      # Same as V_RTD

max_gen_speed = OPT_TSR*V_RTD*GEN_RATIO*30/(R*np.pi)
GENSPEED = (300, max_gen_speed)  # minimum and maximum generator speed [rpm]

# controller parameters
freq = np.arange(0.03,0.07,0.01)
damp = np.arange(0.5,0.81,0.1)
const_power = np.array([1,0])
#%% htc file generation
if MAKE_HTC:
    # make rigid hawc2s file for single-wsp opt file for redesign rotor
    htc = MyHTC(MASTER_FILE)
    htc.make_hawc2s(TARGET_DIR,
                    rigid=True,
                    append='_hawc2s_1wsp',
                    opt_path= f'./data/{DESIGN_NAME}_1wsp.opt',
                    compute_steady_states=True,
                    save_power=False,
                    save_induction=True,
                    minipitch=0,
                    opt_lambda=OPT_TSR,
                    genspeed=GENSPEED)
    
    # make rigid hawc2s file for multi-tsr opt file for redesign rotor
    htc = MyHTC(MASTER_FILE)
    htc.make_hawc2s(TARGET_DIR,
                    rigid=True,
                    append='_hawc2s_multitsr',
                    opt_path=f'./data/{DESIGN_NAME}_multitsr.opt',
                    compute_steady_states=True,
                    save_power=True,
                    minipitch=0,
                    opt_lambda=OPT_TSR,
                    genspeed=GENSPEED)
    
    # rigid hawc2s file for redesign rotor
    htc = MyHTC(MASTER_FILE)
    htc.make_hawc2s(TARGET_DIR,
                    rigid=True,
                    append='_hawc2s_rigid',
                    opt_path=f'./data/{DESIGN_NAME}_rigid.opt',
                    compute_steady_states=True,
                    save_power=True,
                    save_induction=False,
                    compute_optimal_pitch_angle=False,
                    minipitch=0,
                    opt_lambda=OPT_TSR,
                    genspeed=GENSPEED)
    
    # rigid  hawc2s file for DTU 10MW rotor
    htc = MyHTC(MASTER_FILE_DTU)
    htc.make_hawc2s(TARGET_DIR,
                    rigid=True,
                    append='_hawc2s_rigid',
                    opt_path=f'./data/{DTU_NAME}_rigid.opt',
                    compute_steady_states=True,
                    save_power=True,
                    save_induction=False,
                    compute_optimal_pitch_angle=False)
    
    # flexible hawc2s file for redesign rotor 
    htc = MyHTC(MASTER_FILE)
    htc.make_hawc2s(TARGET_DIR,
                    rigid=False,
                    append='_hawc2s_flex',
                    opt_path=f'./data/{DESIGN_NAME}_flex.opt',
                    compute_steady_states=True,
                    save_power=True,
                    save_induction=False,
                    minipitch=0,
                    opt_lambda=OPT_TSR,
                    genspeed=GENSPEED)
    
    # flexible hawc2s file for dtu rotor
    htc = MyHTC(MASTER_FILE_DTU)
    htc.make_hawc2s(TARGET_DIR,
                    rigid=False,
                    append='_hawc2s_flex',
                    opt_path=f'./data/{DTU_NAME}_flex_minrotspd.opt',
                    compute_steady_states=True,
                    save_power=True,
                    save_induction=False)

    # htc file for structural modal analysis
    htc = MyHTC(MASTER_FILE)
    htc.make_hawc2s(TARGET_DIR,
                    rigid=False,
                    append='_hawc2s_modal_structural',
                    opt_path=f'./data/{DESIGN_NAME}_flex.opt',
                    compute_structural_modal_analysis=True,
                    save_modal_amplitude=save_modal_amp,
                    minipitch=0,
                    opt_lambda=OPT_TSR,
                    genspeed=GENSPEED)

    # htc file for aeroelastic modal analysis
    htc = MyHTC(MASTER_FILE)
    htc.make_hawc2s(TARGET_DIR,
                    rigid=False,
                    append='_hawc2s_modal_aeroelastic',
                    opt_path=f'./data/{DESIGN_NAME}_flex.opt',
                    compute_steady_states=True,
                    compute_stability_analysis=True,
                    save_modal_amplitude=save_modal_amp,
                    minipitch=0,
                    opt_lambda=OPT_TSR,
                    genspeed=GENSPEED)

if MAKE_HTC_CTRL:
    for f_ in freq:
        for d_ in damp:
            for p_ in const_power:
                if p_==1:
                    cp ='CP'
                else:
                    cp = 'CT'
                OPT_TSR = 7.2
                htc = MyHTC(MASTER_FILE)
                htc.make_hawc2s_ctrltune(TARGET_DIR,
                                        rigid=False,
                                        append=f'_hawc2s_flex_ctrltune_{cp}_f{f_:.3}_Z{d_:.3}',
                                        opt_path=f'./data/{DESIGN_NAME}_flex.opt',
                                        compute_steady_states=True,
                                        compute_controller_input = True,
                                        save_power=True,
                                        save_induction=False,
                                        minipitch=0,
                                        opt_lambda=OPT_TSR,
                                        genspeed=GENSPEED,
                                        partial_load=(0.05, 0.7),
                                        full_load=(f_, d_),
                                        gain_scheduling=2, 
                                        constant_power=p_ # 0 for cost. torque, 1 for const. power
                                        )