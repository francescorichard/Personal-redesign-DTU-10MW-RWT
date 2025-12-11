"""
This file plots operational data and rotor parameters obtained from rigid and
flexible .ht files
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from myteampack import MyHTC
from lacbox.io import load_ind, load_pwr, load_st

#%% plot commands
#size
mpl.rcParams['figure.figsize'] = (16,8)

#font size of label, title, and legend
mpl.rcParams['font.size'] = 25
mpl.rcParams['xtick.labelsize'] = 35
mpl.rcParams['ytick.labelsize'] = 35
mpl.rcParams['axes.labelsize'] = 50
mpl.rcParams['axes.titlesize'] = 30
mpl.rcParams['legend.fontsize'] = 35

#Lines and markers
mpl.rcParams['lines.linewidth'] = 1.5
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
#%% control booleans

# save data to then plot them
SAVE_IND = True
SAVE_PWR = False
SAVE_RIGID = False
SAVE_FLEXIBLE = False
SAVEFIG = True

# These cannot be True if the respective SAVE_ is not True
MAKE_PLOTS_IND = True # This can be true only if the SAVE_IND is true
MAKE_PLOTS_PWR = False # This can be true only if the SAVE_PWR is true
MAKE_PLOTS_RIGID = False # This can be true only if the _rigid.htc file exists
MAKE_PLOTS_FLEXIBLE = False # This can be true only if the _flexible.htc file

save_modal_amp = False

#%% file names and paths
# Paths
DESIGN_NAME = "IEC_Ya_Later"
DTU_NAME = 'dtu_10mw'

ROOT = Path(__file__).parent.parent  # define root as folder where this script is located

PWR_PATH = ROOT / 'hawc_files' / 'res_hawc2s' / f'{DESIGN_NAME}_hawc2s_multitsr.pwr'
FLEX_PWR_PATH = ROOT / 'hawc_files' / 'res_hawc2s' / f'{DESIGN_NAME}_hawc2s_flex.pwr'
PWR_PATH_DTU = ROOT / 'hawc_files' / 'res_hawc2s' / f'{DTU_NAME}_hawc2s_rigid.pwr'

IND_PATH = ROOT / 'hawc_files' / 'res_hawc2s'
PLOT_FLDR_1WSP = Path(ROOT, 'plots', '1wsp_results')
PLOT_FLDR_MULTITSR = Path(ROOT, 'plots', 'multitsr_results')
PLOT_FLDR_RIGIDWSP = Path(ROOT, 'plots', 'rigid_op_data_results')
PLOT_FLDR_FLEXWSP = Path(ROOT, 'plots', 'flex_op_data_results')

if not PLOT_FLDR_1WSP.is_dir():
    PLOT_FLDR_1WSP.mkdir()
if not PLOT_FLDR_MULTITSR.is_dir():
    PLOT_FLDR_MULTITSR.mkdir()
if not PLOT_FLDR_RIGIDWSP.is_dir():
    PLOT_FLDR_RIGIDWSP.mkdir()
if not PLOT_FLDR_FLEXWSP.is_dir():
    PLOT_FLDR_FLEXWSP.mkdir()

#%% Extra data
# save dtu 10 mw data
pwr_data_dtu = load_pwr(PWR_PATH_DTU)

OPT_TSR= 7.2    # To be changed if another one is desired

#%% Save redesign and DTU data
if SAVE_IND:
    # saving the ind file
    ind_data = {}
    complete_name = IND_PATH /f"{DESIGN_NAME}_hawc2s_1wsp_u8000.ind"
    ind_data[OPT_TSR]= load_ind(complete_name)
    # extracting information from the .ind files for different TSR
    r = ind_data[OPT_TSR]['s_m']
    a= ind_data[OPT_TSR]['a']
    alpha = ind_data[OPT_TSR]['aoa_rad']
    cl = ind_data[OPT_TSR]['Cl']
    glide = ind_data[OPT_TSR]['Cl']/ind_data[OPT_TSR]['Cd']
    cp = ind_data[OPT_TSR]['CP']
    ct = ind_data[OPT_TSR]['CT']
    alpha_loaded = np.load(Path(ROOT, 'hawc_files', 'saved_data', "array_aoa.npy"))
    cl_loaded =np.load(Path(ROOT, 'hawc_files', 'saved_data', "array_cl.npy"))
    cd_loaded = np.load(Path(ROOT, 'hawc_files', 'saved_data', "array_cd.npy"))
    r_loaded = np.load(Path(ROOT, 'hawc_files', 'saved_data', "array_r.npy"))
    ct_group = np.load(Path(ROOT, 'hawc_files', "array_local_ct.npy"))
    r_group = np.load(Path(ROOT, 'hawc_files', "array_r_group.npy"))

if SAVE_PWR:
    # saving the pwr data in a dict
    pwr_data = load_pwr(PWR_PATH)
    TSR = np.arange(5.5, 10.01, .1)
    cp_pwr = pwr_data['Cp']
    ct_pwr = pwr_data['Ct']

if SAVE_RIGID:
    path_file_redesign = ROOT / 'hawc_files' / 'data' / f'{DESIGN_NAME}_rigid.opt'
    path_file_dtu = ROOT / 'hawc_files' / 'data' / f'{DTU_NAME}_rigid.opt'
    # unpack data
    ws_des, pitch_des, omega_des, power_des, thrust_des = np.loadtxt(path_file_redesign,
                                                         skiprows=1,unpack=True)
    ws, pitch, omega, power, thrust = np.loadtxt(path_file_dtu,
                                                 skiprows=1,unpack=True)
    # saving the pwr data in a dict
    path_pwr_rigid = ROOT / 'hawc_files' / 'res_hawc2s' / f'{DESIGN_NAME}_hawc2s_rigid.pwr'
    pwr_data = load_pwr(path_pwr_rigid)
    Cp_des = pwr_data['Cp']
    Cp = power*1e3/(0.5*1.225*np.pi*89.17**2*ws**3)
    CT_des = pwr_data['Ct']
    CT = thrust*1e3/(0.5*1.225*np.pi*89.17**2*ws**2)

if SAVE_FLEXIBLE:
    path_file_redesign_rigid = ROOT / 'hawc_files' / 'data' / f'{DESIGN_NAME}_rigid.opt'
    path_file_dtu_rigid = ROOT / 'hawc_files' / 'data' / f'{DTU_NAME}_rigid.opt'
    path_file_redesign_flex = ROOT / 'hawc_files' / 'data' / f'{DESIGN_NAME}_flex.opt'
    path_file_dtu_flex = ROOT / 'hawc_files' / 'data' / f'{DTU_NAME}_flex_minrotspd.opt'
    
    # unpack data rigid
    ws_des, pitch_des, omega_des, power_des, thrust_des = np.loadtxt(path_file_redesign_rigid,
                                                         skiprows=1,unpack=True)
    ws_dtu, pitch_dtu, omega_dtu, power_dtu, thrust_dtu = np.loadtxt(path_file_dtu_rigid,
                                                 skiprows=1,unpack=True)
    
    # unpack data flex
    ws_des_flex, pitch_des_flex, omega_des_flex, power_des_flex,\
        thrust_des_flex = np.loadtxt(path_file_redesign_flex, skiprows=1, unpack=True)
    ws_dtu_flex, pitch_dtu_flex, omega_dtu_flex, power_dtu_flex,\
        thrust_dtu_flex = np.loadtxt(path_file_dtu_flex, skiprows=1, unpack=True)
    
    # importing structural data
    Design_st_path = ROOT / 'hawc_files' / 'data' / f'{DESIGN_NAME}_Blade_st.dat'
    Design_st_data = load_st(Design_st_path)
    DTU_st_path = ROOT / 'hawc_files' / 'data' / 'DTU_10MW_RWT_Blade_st.dat'
    DTU_st_data = load_st(DTU_st_path)
    subset_Design = Design_st_data[0][0]
    subset_DTU = DTU_st_data[0][0]
    
    # importing pwr data flexible
    flex_pwr_data = load_pwr(FLEX_PWR_PATH)
    pwr_P_flex = flex_pwr_data['P_kW']
    pwr_ws_flex = flex_pwr_data['V_ms']
    CP_flex = flex_pwr_data['Cp']
    CT_flex = flex_pwr_data['Ct']
    
    # importing pwr data rigid
    rigid_pwr_data = load_pwr(ROOT / 'hawc_files' / 'res_hawc2s' / f'{DESIGN_NAME}_hawc2s_rigid.pwr')
    ws_rigid = rigid_pwr_data['V_ms']
    CP_rigid = rigid_pwr_data['Cp']
    CT_rigid = rigid_pwr_data['Ct']
    
    # importing pwr data rigid dtu 10 mw
    dtu_rigid_pwr_data = load_pwr(ROOT / 'hawc_files' / 'res_hawc2s' / f'{DTU_NAME}_hawc2s_rigid.pwr')
    ws_dtu_rigid = dtu_rigid_pwr_data['V_ms']
    CP_dtu_rigid = dtu_rigid_pwr_data['Cp']
    CT_dtu_rigid = dtu_rigid_pwr_data['Ct']
    
    # importing pwr data flexible dtu 10 mw
    dtu_flec_pwr_data = load_pwr(ROOT / 'hawc_files' / 'res_hawc2s' / f'{DTU_NAME}_hawc2s_flex.pwr')
    ws_dtu_flex = dtu_flec_pwr_data['V_ms']
    CP_dtu_flex = dtu_flec_pwr_data['Cp']
    CT_dtu_flex = dtu_flec_pwr_data['Ct']
    pitch_dtu_flex = dtu_flec_pwr_data['Pitch_deg']
   
#%% plotting function
def plot_lines(colors, labels, labels_axis, SAVEFIG, save_path,
             x1=None, x2=None, y1=None, y2=None,**kwargs):
    
    fig,ax = plt.subplots(1,1)

    if x1 is not None:
        ax.plot(x1, y1, color=colors[0],linestyle='-',label=labels[0],zorder=1)
        ax.set_xlim(min(x1), max(x1))

    if x2 is not None:
        ax.plot(x2, y2, color=colors[2],linestyle='-',label=labels[1],zorder=2)

    extra = kwargs.get("extra", [])
    for c in extra:
        ax.plot(
            c["x"],
            c["y"],
            color=c.get("color", "black"),
            linestyle=c.get("ls", "--"),
            label=c.get("label", None),
        )

    ax.set_xlabel(labels_axis[0])
    ax.set_ylabel(labels_axis[1])

    if x1 is not None and x2 is not None:
        ax.legend(loc='best',frameon=False)

    ax.grid(True, which='major', zorder=3, alpha=0.5)
    ax.grid(True, which='minor', zorder=3, alpha=0.3)        
    ax.minorticks_on()
    ax.tick_params(direction='in',right=True,top =True)
    ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
    ax.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)   
    ax.tick_params(direction='in',which='major',length=10,bottom=True,top=True,left=True,right=True)   
    if SAVEFIG:
        plt.savefig(save_path)

#%% make plots from 1 wsp .opt
if MAKE_PLOTS_IND:
    
    # plot Cl along the blade
    labels=['HAWC2S results','design rotor']
    labels_axis=[r'$s\: [m]$',r'$C_l$']
    save_path = PLOT_FLDR_1WSP / 'local_cl.pdf'
    plot_lines(colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path,x1=r, x2=r_loaded, y1=cl, y2=cl_loaded)

    
    # plot Cl/Cd along the blade
    labels=['HAWC2S results','design rotor']
    labels_axis=[r'$s\: [m]$',r'$C_l/C_d$']
    save_path = PLOT_FLDR_1WSP / 'local_glide_ratio.pdf'
    plot_lines(colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path,x1=r, x2=r_loaded, y1=glide, y2=cl_loaded/cd_loaded,)
    
    # plot AoA along the blade
    labels=['HAWC2S results','design rotor']
    labels_axis=[r'$s\: [m]$',r'$\alpha$']
    save_path = PLOT_FLDR_1WSP / 'local_aoa.pdf'
    plot_lines(colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path,x1=r, x2=r_loaded, y1=np.rad2deg(alpha), y2=alpha_loaded,)
    
    # plot axial induction factor along the blade
    labels=['Local a']
    labels_axis=[r'$s\: [m]$',r'$a\:[-]$']
    save_path = PLOT_FLDR_1WSP / 'local_axial_induction.pdf'
    plot_lines( colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path, x1=r, y1=a)
    
    # plot local power coefficient along the blade
    labels=[r'$Local\: C_p\: [-]$']
    labels_axis=[r'$s\: [m]$',r'$C_{p,\:local}\: [-]$']
    save_path = PLOT_FLDR_1WSP / 'local_Cp.pdf'
    plot_lines( colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path, x1=r, y1=cp)
    
    # plot local thrust coefficient along the blade
    labels=[r'$Local\: C_T\: [-]$']
    labels_axis=[r'$s\: [m]$',r'$C_{T,\:local}\: [-]$']
    save_path = PLOT_FLDR_1WSP / 'local_CT.pdf'
    plot_lines( colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path, x1=r, y1=ct)
    
    # plot local thrust coefficient along the blade for group design
    labels=[r'$Personal\: C_T\: [-]$', r'$Group\: C_T\: [-]$']
    labels_axis=[r'$s\: [m]$',r'$C_{T,\:local}\: [-]$']
    save_path = PLOT_FLDR_1WSP / 'local_CT_group_personal.pdf'
    plot_lines( colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path, x1=r, x2=r_group, y1=ct, y2=ct_group)

#%% make plots from multitsr.opt
if MAKE_PLOTS_PWR:    
    # plot CP and CT versus TSR
    fig,ax1 = plt.subplots(1,1, figsize=(15,10))
    ax2 = ax1.twinx()
    ax1.plot(TSR, ct_pwr, color=colors[0], label='Ct', marker='o',linestyle='-',zorder=2)
    ax2.plot(TSR, cp_pwr, color=colors[2], label='Cp', marker='x',linestyle='--',zorder=1)
    ax1.set_xlabel('TSR')
    ax1.set_ylabel(r'$C_T$', color=colors[0])
    ax2.set_ylabel(r'$C_p$', color=colors[2])
    ax1.yaxis.label.set_color(colors[0])
    ax1.spines['left'].set_color(colors[0])
    ax2.yaxis.label.set_color(colors[2])
    ax2.spines['right'].set_color(colors[2])
    ax1.tick_params(axis='y', which = "both", colors=colors[0])
    ax2.tick_params(axis='y', which = "both", colors=colors[2])
    ax1.set_xlim(min(TSR), max(TSR))
    ax2.grid(True, which='major', axis='y', zorder=3,alpha=0.5)
    ax1.grid(True, which='major', axis='x', zorder=3,alpha=0.5)
    ax1.minorticks_on()
    ax1.tick_params(direction='in',right=False,top =True,left=True)
    ax2.tick_params(direction='in',right=True,top =False,left=False)
    ax1.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
    ax1.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=False)   
    ax1.tick_params(direction='in',which='major',length=10,bottom=True,top=True,left=True,right=False)   
    ax2.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=False,right=True)   
    ax2.tick_params(direction='in',which='major',length=10,bottom=True,top=True,left=False,right=True)   
    if SAVEFIG:
        plt.savefig(PLOT_FLDR_MULTITSR / 'cp_ct_vs_tsr.pdf')

#%% make plots of rigid rotor
if MAKE_PLOTS_RIGID:
    
    # aerodynamic power
    labels=['Redesign rotor','DTU 10MW']
    labels_axis=[r'$V_0\: [m/s]$',r'$P\: [MW]$']
    save_path = PLOT_FLDR_RIGIDWSP / 'design_power.pdf'
    plot_lines(colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path,x1=ws_des, x2=ws, y1=power_des/1e3, y2=power/1e3)

    # thrust
    labels=['Redesign rotor','DTU 10MW']
    labels_axis=[r'$V_0\: [m/s]$',r'$T\: [MN]$']
    save_path = PLOT_FLDR_RIGIDWSP / 'design_thrust.pdf'
    plot_lines(colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path,x1=ws_des, x2=ws, y1=thrust_des/1e3, y2=thrust/1e3)
    
    # plot Cp
    labels=['Redesign rotor','DTU 10MW']
    labels_axis=[r'$V_0\: [m/s]$',r'$C_p\: [-]$']
    save_path = PLOT_FLDR_RIGIDWSP / 'design_cp.pdf'
    plot_lines(colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path,x1=ws_des, x2=ws, y1=Cp_des, y2=Cp)
    
    # plot CT
    labels=['Redesign rotor','DTU 10MW']
    labels_axis=[r'$V_0\: [m/s]$',r'$C_T\: [-]$']
    save_path = PLOT_FLDR_RIGIDWSP / 'design_ct.pdf'
    plot_lines(colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path,x1=ws_des, x2=ws, y1=CT_des, y2=CT)
    # plot Cp
    labels=['Redesign rotor','DTU 10MW']
    labels_axis=[r'$V_0\: [m/s]$',r'$C_T\: [-]$']
    save_path = PLOT_FLDR_RIGIDWSP / 'design_ct.pdf'
    plot_lines(colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path,x1=ws_des, x2=ws, y1=CT_des, y2=CT)
    
    # pitch
    labels=['Redesign rotor','DTU 10MW']
    labels_axis=[r'$V_0\: [m/s]$',r'$\theta_p\: [deg]$']
    save_path = PLOT_FLDR_RIGIDWSP / 'design_pitch.pdf'
    plot_lines(colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path,x1=ws_des, x2=ws, y1=pitch_des, y2=pitch)
    
    #rot speed
    labels=['Redesign rotor','DTU 10MW']
    labels_axis=[r'$V_0\: [m/s]$',r'$\omega\: [rpm]$']
    save_path = PLOT_FLDR_RIGIDWSP / 'design_omega.pdf'
    plot_lines(colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path,x1=ws_des, x2=ws, y1=omega_des, y2=omega)
    
    #tip speed
    labels=['Redesign rotor','DTU 10MW']
    labels_axis=[r'$V_0\: [m/s]$',r'$Tip\:speed\: [m/s]$']
    save_path = PLOT_FLDR_RIGIDWSP / 'design_tip_speed.pdf'
    y1 = omega_des*pwr_data['Tip_z_m'][0]*np.pi/30
    y2 = omega*pwr_data_dtu['Tip_z_m'][0]*np.pi/30
    plot_lines(colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path,x1=ws_des, x2=ws, y1=y1, y2=y2)

#%% make plots of flexible rotor
if MAKE_PLOTS_FLEXIBLE:
    # mass distribution
    labels=['Redesign rotor','DTU 10MW']
    labels_axis=[r'$s\:[m]$',r'$m \: [kg/m]$']
    save_path = PLOT_FLDR_FLEXWSP / 'mass_distribution_flex.pdf'
    plot_lines(colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path,x1=subset_Design['s'], x2=subset_DTU['s'],
                       y1=subset_Design['m'], y2=subset_DTU['m'])
        
    labels=[r'Redesign rotor $I_x$',r'Redesign rotor $I_y$']
    labels_axis=[r'$s\:[m]$',r'$I \: [m^4]$']
    save_path = PLOT_FLDR_FLEXWSP / 'inertia_flex.pdf'
    plot_lines(colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path,x1=subset_Design['s'], x2=subset_Design['s'],
                       y1=subset_Design['I_x'], y2=subset_Design['I_y'],
                       extra=[
                           {"x": subset_DTU['s'], "y":  subset_DTU['I_x'],
                            "color": colors[0], "label": r'DTU 10MW $I_x$'},
                           {"x": subset_DTU['s'], "y":  subset_DTU['I_y'],
                            "color": colors[2], "label": r'DTU 10MW $I_y$'}
               ])

    labels=['Rigid redesign','Rigid DTU 10MW']
    labels_axis=[r'$V_0\: [m/s]$',r'$\theta_P\: [deg]$']
    save_path = PLOT_FLDR_FLEXWSP / 'pitch_flex_rigid.pdf'
    plot_lines(colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path,x1=ws_des, x2=ws_dtu,
                       y1=pitch_des, y2=pitch_dtu,
                       extra=[
                           {"x": ws_des, "y":  pitch_des_flex,
                            "color": colors[0], "label": 'Flexible redesign'},
                           {"x": ws_dtu, "y":  pitch_dtu_flex,
                            "color": colors[2], "label": 'Flexible DTU 10MW'}
               ])

    labels=['Rigid redesign','Rigid DTU 10MW']
    labels_axis=[r'$V_0\: [m/s]$',r'$\omega\: [RPM]$']
    save_path = PLOT_FLDR_FLEXWSP / 'omega_flex_rigid.pdf'
    plot_lines(colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path,x1=ws_des, x2=ws_dtu,
                       y1=omega_des, y2=omega_dtu,
                       extra=[
                           {"x": ws_des, "y":  omega_des_flex,
                            "color": colors[0], "label": 'Flexible redesign'},
                           {"x": ws_dtu, "y":  omega_dtu_flex,
                            "color": colors[2], "label": 'Flexible DTU 10MW'}
               ])

    labels=['Rigid redesign','Rigid DTU 10MW']
    labels_axis=[r'$V_0\: [m/s]$',r'$CP \: [-]$']
    save_path = PLOT_FLDR_FLEXWSP / 'cp_flex_rigid.pdf'
    plot_lines(colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path,x1=ws_des, x2=ws_dtu,
                       y1=CP_rigid, y2=CP_dtu_rigid,
                       extra=[
                           {"x": ws_des, "y":  CP_flex,
                            "color": colors[0], "label": 'Flexible redesign'},
                           {"x": ws_dtu, "y":  CP_dtu_flex,
                            "color": colors[2], "label": 'Flexible DTU 10MW'}
               ])

    labels=['Rigid redesign','Rigid DTU 10MW']
    labels_axis=[r'$V_0\: [m/s]$',r'$CT \: [-]$']
    save_path = PLOT_FLDR_FLEXWSP / 'ct_flex_rigid.pdf'
    plot_lines(colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path,x1=ws_des, x2=ws_dtu,
                       y1=CT_rigid, y2=CT_dtu_rigid,
                       extra=[
                           {"x": ws_des, "y":  CT_flex,
                            "color": colors[0], "label": 'Flexible redesign'},
                           {"x": ws_dtu, "y":  CT_dtu_flex,
                            "color": colors[2], "label": 'Flexible DTU 10MW'}
               ])

    labels=['Rigid redesign','Rigid DTU 10MW']
    labels_axis=[r'$V_0\: [m/s]$',r'$P\: [MW]$']
    save_path = PLOT_FLDR_FLEXWSP / 'power_flex_rigid.pdf'
    plot_lines(colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path,x1=ws_des, x2=ws_dtu,
                       y1=power_des/1e3, y2=power_dtu/1e3,
                       extra=[
                           {"x": ws_des, "y":  power_des_flex/1e3,
                            "color": colors[0], "label": 'Flexible redesign'},
                           {"x": ws_dtu, "y":  power_dtu_flex/1e3,
                            "color": colors[2], "label": 'Flexible DTU 10MW'}
               ])

    labels=['Rigid redesign','Rigid DTU 10MW']
    labels_axis=[r'$V_0\: [m/s]$',r'$T\: [MN]$']
    save_path = PLOT_FLDR_FLEXWSP / 'thrust_flex_rigid.pdf'
    plot_lines(colors,labels=labels,labels_axis=labels_axis, SAVEFIG=SAVEFIG,
                       save_path=save_path,x1=ws_des, x2=ws_dtu,
                       y1=thrust_des/1e3, y2=thrust_dtu/1e3,
                       extra=[
                           {"x": ws_des, "y":  thrust_des_flex/1e3,
                            "color": colors[0], "label": 'Flexible redesign'},
                           {"x": ws_dtu, "y":  thrust_dtu_flex/1e3,
                            "color": colors[2], "label": 'Flexible DTU 10MW'}
               ])  

