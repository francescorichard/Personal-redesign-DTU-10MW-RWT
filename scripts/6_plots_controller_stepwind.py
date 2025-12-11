"""
This file generates plots for hawc2 step wind simualtion for different controller
parameters.
It also chooses the best controller parameters based on a weighting function that
weights the overshoot and the settling time of the rotational speed, the pitch,
and the electric power
"""

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy import signal
from itertools import product
from lacbox import io
import scivis

#%% booleans

SAVEFIG = True

#%% Paths and name folders

ROOT = Path(__file__).parent.parent

RES_DIR = ROOT / 'hawc_files' / 'res_steady'
OPT_DIR = ROOT / 'hawc_files' / 'data' / 'IEC_Ya_Later_flex.opt'

PLOT_FLDR = Path(ROOT, 'plots', 'step_wind')
if not PLOT_FLDR.is_dir():
    PLOT_FLDR.mkdir()
# %% Load results

# Find all hdf5 files
hdf5_files = sorted(RES_DIR.glob("IEC_Ya_Later_hawc2s_flex_ctrleval_*.hdf5"))

# initialize arrays for cpct,omega and damping used in the simulation
cpct_arr  = np.zeros(len(hdf5_files), dtype=int)   # 0=CT, 1=CP
omega_arr = np.zeros(len(hdf5_files))
zeta_arr  = np.zeros(len(hdf5_files))

for i, res_file in enumerate(hdf5_files):

    stem = res_file.stem                                     # without .hdf5
    parts = stem.split('_')                                  # split by "_"

    cpct_str = parts[-3]                                     # "CP" or "CT"
    omega_str = parts[-2][1:]                                # remove leading "f"
    zeta_str  = parts[-1][1:]                                # remove leading "Z"

    # Cinvert and save
    cpct_arr[i]  = int(cpct_str)
    omega_arr[i] = float(omega_str)
    zeta_arr[i]  = float(zeta_str)

    res = io.ReadHAWC2(res_file, loaddata=True)
    
    # look for a few channels only
    chan_names = np.array(res.chaninfo[2])
    out_cols = ["Time",
                "Rotor speed",
                "pitch1 angle",
                "DLL :  2 inpvec :   2  pelec [w]",
                "Free wind speed Abs_vhor, gl. coo, of gl. pos    0.00,   0.00,-119.00"]

    out_col_idx = [np.argwhere(chan_names==col).item() for col in out_cols]
    out_col_units = np.array(res.chaninfo[1])[out_col_idx]

    data_tmp = res.data[:, out_col_idx]
    if i == 0:
        data = np.empty((*data_tmp.shape, len(hdf5_files))) # create the output data array
    data[..., i] = data_tmp

names_short = ["t", "omega", "theta", "P_el", "V_0"]
data_dict = {names_short[i]: data[:, i, :] for i in range(len(names_short))}
data_dict["P_el"] /= 1e6  # Conversion from W to MW

#%% calculating overshoot and stabilization time for all variables
# overshoot is the highest value over the value that one should get in a steady
# simulation.
# settling time is the time needed to reduce to the 2% of the mean value in the 
# simulation.

dt = data_dict["t"][1,0] - data_dict["t"][0,0]    # time step
Tstep = 100                                       # length step wind
Nstep = int(Tstep / dt)                           # n° of steps

n_cases = data_dict["omega"].shape[1]
n_steps = data_dict["omega"].shape[0] // Nstep

tol = 0.02     # ±2% settling band

variables = ["theta", "omega","P_el"]

# Calculate reference values
ws, pitch, rot_speed, aero_power, aero_thrust = np.loadtxt(OPT_DIR,
                                                           skiprows=1,
                                                           unpack=True)

# transform aero power to electric power
aero_power[aero_power > 100] = 10000

# eliminate values from 11.1 to 11.9 m/s to have 1 m/s step
indices_to_remove = np.arange(8,17,1)
ws= np.delete(ws,indices_to_remove)
ref_known = {
    "theta": np.delete(pitch,indices_to_remove),
    "omega": np.delete(rot_speed*np.pi/30,indices_to_remove),
    "P_el": np.delete(aero_power/1e3,indices_to_remove)
}

# initialize results for overshoot and settling time
results = {}

for var in variables:
    y_all = data_dict[var]     # picking the simulation of a certain variable
    overshoot_results = np.zeros((n_steps-1, n_cases))
    settling_results  = np.zeros((n_steps-1, n_cases))
    
    # getting the reference values for that variable
    ref_vals = ref_known.get(var, None)
    
    # picking the simulation of a certain controller setting 
    for c_ in range(n_cases):
        y = y_all[:,  c_]
        
        # looping on the n° of step winds
        for k in range(8, n_steps): # starting from 8 since below rated it does not matter
            if ref_vals is not None:
                # Use the reference value
                ref = ref_vals[k]
            else:
                # Use the mean of the last 20% values. Not good for pitch
                y_prev = y[(k-1)*Nstep : k*Nstep]
                ref = np.mean(y_prev[-int(0.2*Nstep):])
            
            # values of the variable at that step
            y_post = y[k*Nstep : (k+1)*Nstep]
            
            # Mean value of second half of oscillations
            mid = len(y_post) // 2
            ref_settle = np.mean(y_post[mid:])

            # Percentage overshoot
            peak = np.max(y_post)
            overshoot_results[k-1,  c_] = (peak - ref) / ref * 100

            #Area of settling
            lower = ref_settle * (1 - tol)
            upper = ref_settle * (1 + tol)
            
            # location (if exists) of settling time
            idx = np.where((y_post >= lower) & (y_post <= upper))[0]

            if len(idx) == 0:
                settling_results[k-1,  c_] = np.nan
                continue

            for m in idx:
                if np.all((y_post[m:] >= lower) & (y_post[m:] <= upper)):
                    settling_results[k-1,  c_] = m * dt
                    break
            else:
                settling_results[k-1,  c_] = np.nan
    
    results[var] = {
        "overshoot": overshoot_results,
        "settling": settling_results
    }

#%% Overshoot and settling time plot

combinations = [
    ("CP" if cpct_arr[i] == 1 else "CT", omega_arr[i], zeta_arr[i])
    for i in range(len(hdf5_files))
]

CPCT = [c[0] for c in combinations]
omega = np.array([c[1] for c in combinations])
zeta  = np.array([c[2] for c in combinations])

omega_unique = np.unique(omega)
zeta_unique  = np.unique(zeta)

n_omega = len(omega_unique)
n_zeta  = len(zeta_unique)

# use V=13 m/s since it's a bit more stable than 12 m/s (first wind speed after rated)
step_to_plot = 8  # step da plottare

cases_CP = [i for i, (c,_,_) in enumerate(combinations) if c=="CP"]
cases_CT = [i for i, (c,_,_) in enumerate(combinations) if c=="CT"]

#  reshape to n_omega x n_zeta ---
def reshape_to_grid(vec):
    if vec.ndim == 1:
        return vec.reshape(n_omega, n_zeta)
    else:
        return vec.reshape(vec.shape[0], n_omega, n_zeta)

# Function to plot heatmap
def plot_imshow(ax, grid, title, cbar_label=None):
    
    # centering the squares on the values used for frequency and damping
    d_omega = omega_unique[1] - omega_unique[0]
    d_zeta  = zeta_unique[1] - zeta_unique[0]
    extent = [zeta_unique[0]-d_zeta/2, zeta_unique[-1]+d_zeta/2,
              omega_unique[0]-d_omega/2, omega_unique[-1]+d_omega/2]

    im = ax.pcolormesh(zeta_unique, omega_unique, grid, cmap='viridis',
                   edgecolors='white', linewidth=0.5)
    cbar = plt.colorbar(im, ax=ax)
    if cbar_label is not None:
        cbar.set_label(cbar_label,fontsize=16)
    
    ax.set_xticks(zeta_unique)
    ax.set_yticks(omega_unique)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    ax.set_xlabel('ζ [-]',fontsize=16)
    ax.set_ylabel('ω [Hz]',fontsize=16)
    ax.set_title(title,fontsize=16)


variables = ["omega", "theta", "P_el"]
overshoot_CP = {}
overshoot_CT = {}
settling_CP = {}
settling_CT = {}
for var_to_plot in variables:
    name = '\\' + str(var_to_plot.strip('"'))
    if  var_to_plot == "P_el":
        name = 'P_{el}' 
    overshoot_results = results[var_to_plot]['overshoot']
    settling_results  = results[var_to_plot]['settling']

    overshoot_CP[var_to_plot] = overshoot_results[step_to_plot, cases_CP]
    settling_CP[var_to_plot]  = settling_results[step_to_plot,  cases_CP]
    overshoot_CT[var_to_plot] = overshoot_results[step_to_plot, cases_CT]
    settling_CT[var_to_plot]  = settling_results[step_to_plot,  cases_CT]
    
    # normalize settling time based
    if var_to_plot == 'P_el':
        pass
    else:
        settling_CP[var_to_plot] = (settling_CP[var_to_plot]-np.nanmin(settling_CP[var_to_plot]))/\
            (np.nanmax(settling_CP[var_to_plot])-np.nanmin(settling_CP[var_to_plot]))
    settling_CT[var_to_plot] = (settling_CT[var_to_plot]-np.nanmin(settling_CT[var_to_plot]))/\
        (np.nanmax(settling_CT[var_to_plot])-np.nanmin(settling_CT[var_to_plot]))
                            
    # normalize overshoot based 
    overshoot_CP[var_to_plot] = (overshoot_CP[var_to_plot]-np.nanmin(overshoot_CP[var_to_plot]))/\
        (np.nanmax(overshoot_CP[var_to_plot])-np.nanmin(overshoot_CP[var_to_plot]))
    overshoot_CT[var_to_plot] = (overshoot_CT[var_to_plot]-np.nanmin(overshoot_CT[var_to_plot]))/\
        (np.nanmax(overshoot_CT[var_to_plot])-np.nanmin(overshoot_CT[var_to_plot]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Overshoot CP
    grid_over_CP = reshape_to_grid(overshoot_CP[var_to_plot])
    plot_imshow(axes[0,0], grid_over_CP,
                fr"Overshoot ${name}$ (CP) - V={ws[step_to_plot+1]} m/s")
    
    # Settling CP
    grid_set_CP = reshape_to_grid(settling_CP[var_to_plot])
    plot_imshow(axes[0,1], grid_set_CP,
                fr"Settling Time ${name}$ (CP) - V={ws[step_to_plot+1]} m/s")
    
    # Overshoot CT
    grid_over_CT = reshape_to_grid(overshoot_CT[var_to_plot])
    plot_imshow(axes[1,0], grid_over_CT,
                fr"Overshoot ${name}$ (CT) - V={ws[step_to_plot+1]} m/s")
    
    # Settling CT
    grid_set_CT = reshape_to_grid(settling_CT[var_to_plot])
    plot_imshow(axes[1,1], grid_set_CT,
                fr"Settling Time ${name}$ (CT) - V={ws[step_to_plot+1]} m/s")
    
    plt.tight_layout()
    if SAVEFIG:
        plt.savefig(PLOT_FLDR / f'overshoot_settling_{var_to_plot}.pdf')
        
#%% Weighted metric afterr choosing constant torque

over_pitch_CT   = overshoot_CT["theta"]
over_omega_CT   = overshoot_CT["omega"]
over_power_CT   = overshoot_CT["P_el"]
sett_pitch_CT   = settling_CT["theta"]
sett_omega_CT   = settling_CT["omega"]
sett_power_CT   = settling_CT["P_el"]

weighted_CT = (
      1/6 * over_pitch_CT
    + 1/6 * over_omega_CT
    + 1/6 * sett_pitch_CT
    + 1/6 * sett_omega_CT
    + 1/6 * over_power_CT
    + 1/6 * sett_power_CT
)


# Reshape al grid 2D (omega × zeta)
grid_weighted_CT = reshape_to_grid(weighted_CT)

fig, ax = plt.subplots(figsize=(7, 6))

plot_imshow(ax, grid_weighted_CT,
            f"Weighted metric (CT only) – V={ws[step_to_plot+1]} m/s",
            "Weighted metric")

plt.tight_layout()

if SAVEFIG:
    plt.savefig(PLOT_FLDR / "weighted_CT_metric.pdf")

# %% Plot step wind results over time

# Plot settings
latex = False
plot_stp = 30  # step width for the plots (to speed up plotting)
line_colors = plt.get_cmap("tab20").colors
right_axis_color = "#a80922"
note_color, note_alpha = "#20174D", .7
#line_colors = ["#100102", "#5f73eb", "#fca600"]

# Rated speed
OPT_TSR= 7.2    # To be changed if another one is desired
GEN_RATIO = 50
V_RTD = 11.256  # This is valid for the current design. If R is changed, this needs to change as well
R = 90.879      # Same as V_RTD

rpm_HSS = OPT_TSR*V_RTD*GEN_RATIO*30/(R*np.pi)
omega_rtd_LSS = OPT_TSR*V_RTD / R

wind_step = 2
CUTIN, CUTOUT = 4, 25
DT, TSTART = 100, 100
wsp_steps = np.arange(CUTIN, CUTOUT+1, 1)
step_times = np.arange(0, len(wsp_steps), 1)*DT + TSTART
yticks_V0 = np.arange(CUTIN, np.ceil(CUTOUT/wind_step)*wind_step+ 1,
                      wind_step)
ylabel_mapping = {"omega": r"\omega", "theta": r"\theta", "P_el": "P_{el}"}
yunits_mapping = {"omega": "rad/s", "theta": "deg", "P_el": "MW"}

arrowstyle_lin = dict(arrowstyle="-", connectionstyle="angle,angleA=90")
rc_profile = scivis.rcparams._prepare_rcparams(latex=latex)

# Plotting
with mpl.rc_context(rc_profile):
    for name in ["omega", "theta", "P_el"]:
        for i in range(2):
            # Plot parameter
            labels = [f"{CPCT[(i*20) + j]}_{omega_arr[(i*20) + j]:.3}"+\
                      f"_{zeta_arr[(i*20) + j]:.2}"  for j in range(20)]
            fig, ax_l, _ = scivis.plot_line(data_dict["t"][0:-1:plot_stp, 0],
                                            data_dict[name][0:-1:plot_stp,
                                                            i*20:(i*20)+20].T,
                                            plt_labels=labels,
                                            ax_labels=["t", ylabel_mapping[name]],
                                            ax_units=["s", yunits_mapping[name]],
                                            ax_show_grid_minor=True,
                                            colors=line_colors, linestyles="-",
                                            latex=latex)
            ax_l.zorder = 2
            ax_l.grid(zorder=2)
            zorders = tuple(range(25, 5, -1))
            for idx, line in enumerate(ax_l.get_lines()):
                line.zorder = zorders[idx]

            # Plot wind speed on second axis
            ax_r = ax_l.twinx()
            ax_r.zorder = 1
            fig, ax_r, _ = scivis.plot_line(data_dict["t"][0:-1:plot_stp, 0],
                                            data_dict["V_0"][0:-1:plot_stp, 0],
                                            ax=ax_r, colors=right_axis_color,
                                            ax_labels=["t", "V_0"],
                                            ax_units=["s", "m/s"],
                                            ax_show_grid=False,
                                            show_legend=False,
                                            latex=latex)
            ax_r.get_lines()[0].zorder = 1

            # Change color or right axis
            ax_r.tick_params(axis='y', which = "both", colors=right_axis_color)
            ax_r.yaxis.label.set_color(right_axis_color)
            ax_r.spines['right'].set_color(right_axis_color)

            #Adjust y-ticks
            ax_r.set_yticks(yticks_V0)
            ax_r.set_ylim((CUTIN-.5, CUTOUT+.5))

            # Add dash-dotted lines for the wind speed steps
            x_max = ax_l.get_xlim()[1]
            ax_r.hlines(wsp_steps, xmin=step_times + DT, xmax=x_max,
                         linestyles="-.", colors=right_axis_color, alpha=.6,
                         zorder=1)

            # Force legend to upper left corner
            ax_l.legend(loc="lower center",bbox_to_anchor=(0.5,1.09),
                                      ncols=7,fontsize=14, frameon=False)
            # ax_l.get_legend().set_loc("lower center",bbox_to_anchor=(0.5,1.09),
            #                           ncols=5)

            # Add line for rated wind speed
            if name == "omega":
                ax_l.axhline(omega_rtd_LSS, ls="-.",
                             c=note_color, alpha=note_alpha)
                arrowstyle = dict(arrowstyle="-", alpha=0)
                bbox=dict(facecolor='w', alpha=0.4, ls="none")
                ax_l.annotate(
                    text="$\omega_{rtd}$", xy=(500, omega_rtd_LSS),
                    xytext=(0, 7),
                    xycoords="data", textcoords="offset points",
                    rotation="horizontal", ha="center", va="bottom",
                    arrowprops=arrowstyle, bbox=bbox,
                    c=note_color, alpha=note_alpha)

            if SAVEFIG:
                name = name.strip("")
                name_fig = "step_wind_" +  f"{CPCT[(i*20)]}_" + name + ".pdf"
                fig.savefig(PLOT_FLDR / name_fig)

# %% Plotting PSD

# dt = data_dict["t"][1, 0] - data_dict["t"][0, 0]
# fs=1/dt

# domega = data_dict["omega"] - np.mean(data_dict["omega"], axis=0)
# f = []
# spectrum = []
# for i in range(domega.shape[1]):
#     f_i, spectrum_i = signal.welch(domega[:,i], fs, nperseg=1024)

#     f.append(f_i)
#     spectrum.append(spectrum_i)

# f = np.stack(f)
# spectrum = np.stack(spectrum)

# with mpl.rc_context(rc_profile):
#     for i in range(2):
#         labels = [f"$C{((i*20) + (j+1))}$" for j in range(20)]
#         fig, ax, _ = scivis.plot_line(f[i*3:(i*3)+3, :],
#                                         spectrum[i*3:(i*3)+3, :],
#                                         plt_labels=labels,
#                                         ax_labels=["f", "PSD"],
#                                         ax_units=["Hz", ""],
#                                         ax_lims=[[0,10], [1e-14, 5e-3]],
#                                         autoscale_y=False,
#                                         ax_show_grid_minor=True,
#                                         colors=line_colors, linestyles="-",
#                                         latex=latex)

#         ax.set_yscale("log")
#         ax.set_ylim([1e-15, 5e-3])
#         ax.zorder = 2
#         ax.grid(zorder=2)
#         zorders = (6, 5, 4)
#         for idx, line in enumerate(ax.get_lines()):
#             line.zorder = zorders[idx]

#         scivis.axvline(ax, x=rpm_HSS/n_gear/60, text=r"$\omega_{rtd}$")

#         # Save the figure
#         fig.savefig(f"p3_PSD_{i+1}.svg")








