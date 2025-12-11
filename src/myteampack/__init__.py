# pylint:disable=line-too-long
"""Main classes and functions.
"""
from pathlib import Path

from lacbox.htc import HTCFile
from lacbox.io import load_oper, load_ctrl_txt
import numpy as np


class MyHTC(HTCFile):
    """Your team's class to auto-generate htc files.

    Instantiate a MyHTC object by passing in path to an htc file. E.g.:
        >> htc = MyHTC('./test.htc')
    """

    def _add_ctrltune_block(self, partial_load=(0.05, 0.7),  # pylint:disable=unused-argument
                            full_load=(0.06, 0.7),
                            gain_scheduling=2, constant_power=1, **kwargs):
        """Add a controller-tuning block to HAWC2S file. See HAWCStab2
        manual for explanation of input parameters.
        """

        ctrltune = self.hawcstab2.add_section('controller_tuning',
                                              pre_comments='\n; Need this block to call compute_controller_input')
        ctrltune.add_line(name='partial_load', values=partial_load, comments='fn [hz], zeta [-]')
        ctrltune.add_line(name='full_load', values=full_load, comments='fn [hz], zeta [-]')
        ctrltune.add_line(name='gain_scheduling', values=[gain_scheduling], comments='1 linear, 2 quadratic')
        ctrltune.add_line(name='constant_power', values=[constant_power], comments='0 constant torque, 1 constant power at full load')
        #ctrltune.add_line(name='regions', values=regions, comments='Index of opt point (starting from 1) where new ctrl region starts')

    def _add_hawc2s_commands(self, rigid, tipcorr=True, induction=True,
                             num_modes=12, **kwargs):
        """Add commands for HAWC2S to end of hawcstab2 block.

        Comment/uncomment a HAWC2S command by passing the command as a
        keyword argument equal to True. E.g.:
            >> htc._add_hawc2s_commands(rigid=True, compute_steady_states=True, save_power=True)
        """
        defl_key = ['', 'no'][rigid]  # blade deformation or no?
        tipcorr_key = ['no', ''][tipcorr]  # tip correction or no?
        ind_key = ['no', ''][induction]  # induction or no?
        # add the pre-amble and output folder
        self.hawcstab2.add_line(name='', values=[''], comments='\n; HAWC2S commands (uncomment as needed)')
        self.hawcstab2.add_line(name='output_folder', values=['res_hawc2s'], comments='define the folder where generated files should be saved')
        # commands, options, and comments for hawc2s commands
        hawc2s_commands = {'compute_optimal_pitch_angle': [['use_operational_data'],
                                                           're-calculate and save opt file (pitch/rotor speed curve)'],
                           'compute_steady_states': [[f'{defl_key}bladedeform', f'{tipcorr_key}tipcorrect',
                                                      f'{ind_key}induction', 'nogradients'],
                                                     'compute steady states -- needed for aeroelastic calculations'],
                           'save_power': [[''], 'save steady-state values to .pwr'],
                           'save_induction': [[''], 'save steady-state spanwise values to .ind files, 3 for each wind speed'],
                           'compute_stability_analysis': [[f'windturbine {num_modes}'],
                                                          'compute/save aeroelastic campbell diagram (.cmb), XX modes'],
                           'save_modal_amplitude': [[''], 'save modal amplitudes and phrases to .amb file'],
                           'compute_controller_input': [[''], 'calculate/save controller parameters (reqs. steady_states)'],
                           }
        # iterate over possible hawc2s commands
        for command, (values_, comments_) in hawc2s_commands.items():
            # if the command was passed in as a keyword argument, use the value passed in
            if command in kwargs:
                uncomment_command = kwargs[command]
            # if not given as keyword argument, default is for command to be commented
            else:
                uncomment_command = False
            name_ = [';', ''][uncomment_command] + command
            self.hawcstab2.add_line(name=name_, values=values_, comments=comments_)

    def _check_hawcstab2(self):
        """Verify HAWCStab2 block exists."""
        try:  # try to access the block
            self.hawcstab2
        except KeyError as exc:  # if we get a KeyError, block missing
            print('HAWCStab2 block not present in base file! Halting.')
            raise exc

    def _del_not_h2s_blocks(self):
        """Delete blocks in the htc file that HAWC2S doesn't use."""
        del self.simulation
        del self.dll
        del self.output
        del self.wind  # remove entire wind block for simplicity
        del self.aerodrag  # hawcstab2 dies with aerodrag can't handle this either

    def _del_wind_ramp_abs(self):
        """Remove all wind_ramp_abs calls (used in step-winds) in wind block."""
        for k in self.wind.keys():
            if 'wind_ramp_abs' in k:
                del self.wind.contents[k]

    def _set_initial_rotor_speed(self, omega0, bodyname='shaft', rotvec=(0., 0., -1.)):
        """Set the initial rotor speed on the shaft [rad/s]"""
        body = self.new_htc_structure.orientation.get_section(bodyname, field='mbdy2')
        body.mbdy2_ini_rotvec_d1 = rotvec + (omega0,)

    def _update_name_and_save(self, htcdir, append, name=None,
                              subfolder='', resdir='res'):  # TODO Make append a kwarg next year. Also make it a public method
        """Update filename and save the file."""
        save_dir = Path(htcdir) / subfolder  # sanitize inputs and add subfolder
        # get new name (excl extension) from HTCFile attribute "filename"
        if name is None:
            name = Path(self.filename).name.replace('.htc', append)
        # set filename using HTCFile method
        self.set_name(name, subfolder=subfolder, resdir=resdir)
        # save the file
        self.save((save_dir / (name + '.htc')).as_posix())
        return name + '.htc'

    def _update_ctrl_params(self, ctrl_dict,
                            min_rot_speed=None, rated_rot_speed=None):
        """This function updates the parameters used for the control tuning

        Args:
            ctrl_dict (dict): Dictionary containing the PI controller parameters
            min_rot_speed (int, optional): Minimum rotor speed, in rad/s. Defaults to None.
            rated_rot_speed (int, optional): Rated rotor speed, in rad/s. Defaults to None.
        """

        if min_rot_speed is not None:
            self.dll.type2_dll__1.init.constant__2 = [2, min_rot_speed]

        if rated_rot_speed is not None:
            self.dll.type2_dll__1.init.constant__3 = [3, rated_rot_speed]

        match ctrl_dict["CP/CT"]:
            case "CP" | 1:
                ctrl_dict["CP/CT"] = 1
            case "CT" | 0:
                ctrl_dict["CP/CT"] = 0
            case _:
                raise ValueError("Full load torque control type must be "
                                 "'CP' or 'CT'")

        self.dll.type2_dll__1.init.constant__11 = [11, ctrl_dict["K_Nm/(rad/s)^2"]]
        self.dll.type2_dll__1.init.constant__12 = [12, ctrl_dict["KpTrq_Nm/(rad/s)"]]
        self.dll.type2_dll__1.init.constant__13 = [13, ctrl_dict["KiTrq_Nm/rad"]]
        self.dll.type2_dll__1.init.constant__15 = [15, ctrl_dict["CP/CT"]]
        self.dll.type2_dll__1.init.constant__16 = [16, ctrl_dict["KpPit_rad/(rad/s)"]]
        self.dll.type2_dll__1.init.constant__17 = [17, ctrl_dict["KiPit_rad/rad"]]
        self.dll.type2_dll__1.init.constant__21 = [21, ctrl_dict["K1_deg"]]
        self.dll.type2_dll__1.init.constant__22 = [22, ctrl_dict["K2_deg^2"]]
    def get_main_body_c2_def_axis(self, mbdy_name="blade1"):
        """Get c2-def axis for a main body unpacked as arrays

        Parameters
        ----------
        mbdy_name : str, optional
            Name of the main body to extract C2-def axis for, by default "blade1"

        Returns
        -------
        tuple
            0. x-axis values
            1. y-axis values
            2. z-axis values
            3. twist values
        """
        # Getting original axis and curve length
        blade1_mbdy = self.new_htc_structure.main_body(name=mbdy_name) # Extracting main_body `blade1`
        nsec = blade1_mbdy.c2_def.nsec.values[0] # Get the number of nsec
        # Initialize axis and twist arrays
        x=np.zeros(nsec) # [m]
        y=np.zeros(nsec) # [m]
        z=np.zeros(nsec) # [m]
        twist=np.zeros(nsec) # [deg]
        # Extract C2-def axis
        sec_name = "sec"
        for isec in range(nsec):
            x[isec], y[isec], z[isec], twist[isec] = blade1_mbdy.c2_def[sec_name].values[1:]
            sec_name = f"sec__{isec+2}"
        return x, y, z, twist

    def make_hawc2s(self, save_dir, rigid, append, opt_path,
                     genspeed=None, minipitch=0, opt_lambda=8, **kwargs):
        """Make a HAWC2S file with specific settings.

        Args:
            save_dir (str/pathlib.Path): Path to folder where the htc file
                should be saved.
            rigid (boolean): Whether HAWC2S analysis should be a rigid or flexible
                structure.
            append (str): Text to append to the name of the master file.
            opt_path (str): Relative path from the saved htc file to the opt_file.
            minipitch (float): Minimum pitch angle [deg]. Default to 10MW DTU Report
            opt_lambda (float): Optimum tip-speed ratio. Default to 10MW DTU Report
            genspeed (tuple, optional): 2-element tuple of minimum and maximum generator
                speed. Defaults to None -> not adding/overwriting.
        """
        # verify the file has hawcstab2 block
        self._check_hawcstab2()
        # delete blocks in master htc file that HAWC2S doesn't use
        self._del_not_h2s_blocks()
        # update the flexibility parameter in operational_data subblock
        defl_flag = [1, 0][rigid]  # 0 if rigid=True, else 1
        self.hawcstab2.operational_data.include_torsiondeform = defl_flag
        # correct the path to the opt file
        self.hawcstab2.operational_data_filename = opt_path
        # update the minimum generator speed
        if not genspeed is None:
            self.hawcstab2.operational_data.genspeed = genspeed
        if minipitch !=0:
            self.hawcstab2.operational_data.minipitch = minipitch
        if opt_lambda != 8:
            self.hawcstab2.operational_data.opt_lambda = opt_lambda
        # add hawc2s commands
        self._add_hawc2s_commands(rigid=rigid, **kwargs)
        # update filename and save the file
        name = self._update_name_and_save(save_dir, append)
        print(f'File "{name}" saved.')

    def make_hawc2s_ctrltune(self, save_dir, rigid, append, opt_path,
                     genspeed=None, minipitch=0, opt_lambda=7.5, **kwargs):
        """Make a HAWC2S file with specific settings.

        Args:
            save_dir (str/pathlib.Path): Path to folder where the htc file
                should be saved.
            rigid (boolean): Whether HAWC2S analysis should be a rigid or flexible
                structure.
            append (str): Text to append to the name of the master file.
            opt_path (str): Relative path from the saved htc file to the opt_file.
            minipitch (float): Minimum pitch angle [deg]. Default to 10MW DTU Report
            opt_lambda (float): Optimum tip-speed ratio. Default to 10MW DTU Report
            genspeed (tuple, optional): 2-element tuple of minimum and maximum generator
                speed. Defaults to None -> not adding/overwriting.
        """
        # verify the file has hawcstab2 block
        self._check_hawcstab2()
        # delete blocks in master htc file that HAWC2S doesn't use
        self._del_not_h2s_blocks()
        # update the flexibility parameter in operational_data subblock
        defl_flag = [1, 0][rigid]  # 0 if rigid=True, else 1
        self.hawcstab2.operational_data.include_torsiondeform = defl_flag
        # correct the path to the opt file
        self.hawcstab2.operational_data_filename = opt_path
        # update the minimum generator speed
        if not genspeed is None:
            self.hawcstab2.operational_data.genspeed = genspeed
        if minipitch !=0:
            self.hawcstab2.operational_data.minipitch = minipitch
        if opt_lambda != 7.5:
            self.hawcstab2.operational_data.opt_lambda = opt_lambda
        # add hawc2s commands
        self._add_hawc2s_commands(rigid=rigid, **kwargs)
        # Add controller tuning
        self._add_ctrltune_block(**kwargs)
        # update filename and save the file
        name = self._update_name_and_save(save_dir, append)
        print(f'File "{name}" saved.')

    def make_turb(self, save_dir, wsp, append, ti, seed, dy=190, dz=190,
                  resdir='res_turb/', opt_path=None, tilt=None, subfolder='',
                  rigid=False, withdrag=True, time_start=100, time_stop=700):
        """Make a turbulent wind htc file."""
        # delete hawcstab2 block for a cleaner file
        del self.hawcstab2

        # delete any steps
        self._del_wind_ramp_abs()

        # set initial rotor speed if opt file given
        if opt_path is not None:
            omega0 = get_initial_rotor_speed(wsp, opt_path)
            self._set_initial_rotor_speed(omega0)

        # set the start and stop time
        self.set_time(start=time_start, stop=time_stop)  # simulation times

        # update tilt if a number is passed in
        if type(tilt) in [int, float]:
            shaft_ori = self.new_htc_structure.orientation.relative__2
            shaft_ori.mbdy2_eulerang__2 = [tilt, 0, 0]
        elif tilt is not None:
            raise ValueError('Keyword argument "tilt" must be None, int or float!')

        # rigid tower/blades if requested
        if rigid:
            self.new_htc_structure.main_body.timoschenko_input.set = [1, 2]
            self.new_htc_structure.main_body__7.timoschenko_input.set = [1, 2]

        # delete aerodynamic drag if requested
        if not withdrag:
            del self.aerodrag

        # wind-speed values
        self.wind.tint = ti  # TI from input
        self.wind.turb_format = 1  # Mann's model
        self.wind.tower_shadow_method = 3  # Advanced potential flow model
        self.wind.wsp = wsp  # mean wind speed
        self.wind.shear_format = [3, .2]  # power law wsp profile with height

        # Mann's turbulence block
        # (with default values from HAWC2 docs and example file from DTU Learn)
        fname_base = Path(self.filename).name.replace('.htc', append)
        fnames_turb = [r"./turb/" + fname_base + "_" + c + ".bin"
                       for c in ("u", "v", "w")]
        NY, NZ = 32, 32
        self.add_mann_turbulence(L=33.6, ae23=1, Gamma=3.9, seed=seed,
                                 high_frq_compensation=True,
                                 filenames=fnames_turb,
                                 no_grid_points=(1024, 32, 32),
                                 box_dimension=(wsp*(time_stop-time_start),
                                                NY*dy/(NY-1),
                                                NZ*dz/(NZ-1)),
                                 dont_scale=False, std_scaling=None)

        # update the filename and save
        name = self._update_name_and_save(save_dir, append,
                                          subfolder=subfolder, resdir=resdir)
        print(f'Turbulent wind file saved: {name}')



    def make_step(self, save_dir, wsp, wsp_steps, step_times, start_record_time, last_step_len,
                  append, ramp_dt=1, resdir='res_steady/',
                  opt_path=None, tilt=None, subfolder='',
                  rigid=False, withdrag=True):
        """Make a step-wind htc file."""
        # delete hawcstab2 block for a cleaner file
        del self.hawcstab2

        # delete any steps
        self._del_wind_ramp_abs()

        # set initial rotor speed if opt file given
        if opt_path is not None:
            omega0 = get_initial_rotor_speed(wsp, opt_path)
            self._set_initial_rotor_speed(omega0)

        # set the start and stop time
        self.set_time(start=start_record_time, stop=step_times[-1]+last_step_len)  # simulation times

        # update tilt if a number is passed in
        if type(tilt) in [int, float]:
            shaft_ori = self.new_htc_structure.orientation.relative__2
            shaft_ori.mbdy2_eulerang__2 = [tilt, 0, 0]
        elif tilt is not None:
            raise ValueError('Keyword argument "tilt" must be None, int or float!')

        # rigid tower/blades if requested
        if rigid:
            self.new_htc_structure.main_body.timoschenko_input.set = [1, 2]
            self.new_htc_structure.main_body__7.timoschenko_input.set = [1, 2]

        # delete aerodynamic drag if requested
        if not withdrag:
            del self.aerodrag

        # wind-speed  values
        self.wind.tint = 0  # no TI
        self.wind.turb_format = 0  # no turbulence
        self.wind.tower_shadow_method = 0  # no tower shadow
        self.wind.wsp = wsp  # mean wind speed
        self.wind.shear_format = [3, 0]  # power law with alpha=0 (constant wind)

        if ramp_dt <0: ramp_dt = 0

        # Initial steady period (from 0 to first step time)
        step_pattern = r"wind_ramp_abs  {t0:8.2f}  {t1:8.2f}  "\
            "{v0:6.2f}  {v1:6.2f}; Step from {vstart:.2f} to  {vend:.2f}\n"

        wind_steps = [step_pattern.format(t0=step_times[0], t1=step_times[0] + ramp_dt,
                                        v0=0, v1=wsp_steps[0]-wsp,
                                        vstart=wsp, vend=wsp_steps[0])]
        wind_steps[0] = wind_steps[0].replace("wind_ramp_abs  ", "")

        # Add other wind steps
        for i in range(1, len(wsp_steps)):
            v_start = wsp_steps[i-1]
            t_start = step_times[i]
            v_step = wsp_steps[i] - v_start

            # Ramp to next speed
            wind_steps.append("\t" +
                step_pattern.format(t0=t_start, t1=t_start + ramp_dt,
                                    v0=0, v1=v_step,
                                    vstart=v_start, vend=v_start+v_step))

        self.wind.wind_ramp_abs = wind_steps

        # update the filename and save
        name = self._update_name_and_save(save_dir, append,
                                          subfolder=subfolder, resdir=resdir)
        print(f'Steady-wind file saved: {name}')

    def make_steady(self, save_dir, wsp, append, resdir='res_steady/',
                    opt_path=None, tilt=None, subfolder='',
                    rigid=False, withdrag=True, time_start=200, time_stop=400):
        """Make a steady-wind htc file."""
        # delete hawcstab2 block for a cleaner file
        del self.hawcstab2

        # delete any steps
        self._del_wind_ramp_abs()

        # set initial rotor speed if opt file given
        if opt_path is not None:
            omega0 = get_initial_rotor_speed(wsp, opt_path)
            self._set_initial_rotor_speed(omega0)

        # set the start and stop time
        self.set_time(start=time_start, stop=time_stop)  # simulation times

        # update tilt if a number is passed in
        if type(tilt) in [int, float]:
            shaft_ori = self.new_htc_structure.orientation.relative__2
            shaft_ori.mbdy2_eulerang__2 = [tilt, 0, 0]
        elif tilt is not None:
            raise ValueError('Keyword argument "tilt" must be None, int or float!')

        # rigid tower/blades if requested
        if rigid:
            self.new_htc_structure.main_body.timoschenko_input.set = [1, 2]
            self.new_htc_structure.main_body__7.timoschenko_input.set = [1, 2]

        # delete aerodynamic drag if requested
        if not withdrag:
            del self.aerodrag

        # wind-speed  values
        self.wind.tint = 0  # no TI
        self.wind.turb_format = 0  # no turbulence
        self.wind.tower_shadow_method = 0  # no tower shadow
        self.wind.wsp = wsp  # mean wind speed
        self.wind.shear_format = [1, wsp]  # constant wsp profile with height

        # update the filename and save
        name = self._update_name_and_save(save_dir, append, subfolder=subfolder, resdir=resdir)
        print(f'Steady-wind file saved: {name}')

    def set_main_body_c2_def_axis(self, x, y, z, twist, mbdy_name="blade1"):
        """Set the C2-def axis for a main-body from a set of arrays.

        Parameters
        ----------
        x : np.ndarray
            x-axis values
        y : np.ndarray
            y-axis values
        z : np.ndarray
            z-axis values
        twist : np.ndarray
            twist values
        mbdy_name : str, optional
            Name of the main body to extract C2-def axis for, by default "blade1"
        """
        # Data validation
        assert (abs(x[0]) < 1e-8) & (abs(y[0]) < 1e-4) & (abs(z[0]) < 1e-8), f"First value of the axis should be x,y,z=0,0,0 (given: {x[0]=:1.3e}, {y[0]=:1.3e}, {z[0]=:1.3e})"
        assert twist[0] < 0, f"The first twist value should be negative in HAWC2 (given: {twist[0]=:2.3f})"
        assert (len(x) == len(y) & len(x) == len(z) & len(x) == len(twist)), f"All input arrays need to be of equal length (given: {len(x)=}, {len(y)=}, {len(z)=}, {len(twist)=})"
        blade1_mbdy = self.new_htc_structure.main_body(name=mbdy_name) # Extracting main_body `blade1`
        # Delete C2-def section
        blade1_mbdy.c2_def.delete()
        # Added new C2-def section
        blade1_mbdy.add_section(section_name="c2_def")
        nsec = len(z)
        blade1_mbdy.c2_def.nsec = nsec
        sec_name = "sec"
        for isec in range(nsec):
            blade1_mbdy.c2_def[sec_name] = [isec+1, x[isec], y[isec], z[isec], twist[isec]]
            sec_name = f"sec__{isec+2}"

def get_initial_rotor_speed(wsp, opt_path):
    """Given a wind speed and path to opt file, find initial rotor speed.

    Args:
        wsp (int, float): Wind speed [m/s].
        opt_path (str, pathlib.Path): Path to opt file.

    Returns:
        int, float: Initial rotor speed interpolated from opt file [rad/s].
    """
    opt_dict = load_oper(opt_path)
    opt_wsps = opt_dict['ws_ms']
    opt_rpm = opt_dict['rotor_speed_rpm']
    omega_rpm = np.interp(wsp, opt_wsps, opt_rpm)
    omega0 = omega_rpm * np.pi / 30  # rpm to rad/s
    return omega0
