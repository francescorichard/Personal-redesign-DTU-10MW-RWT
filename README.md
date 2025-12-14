# Design code for final redesign of DTU 10MW RWT

HAWC files and Python code used in the design exercise in
46320: Loads, Aerodynamics and Control of Wind Turbines.

## Re-install myteampack

These steps assume (1) you have installed Python, VS Code, HAWC2/HAWCStab2, etc.,
per the instructions given in the course and (2) that you are familiar with
conda environments, installing Python packages, etc.

**If you have previously installed another version of myteampack**

1. Uninstall `myteampack`: `pip uninstall myteampack`

**Proceed...**

1. (Recommended) Create and activate a conda environment called `lac`
1. Install the new `myteampack` editably. In the main directory, `pip install -e .`

## Install scivis

Scivis is a plotting package created by Davis Stower as an alternative to the default `matplotlib`. 
In order to install it,l in the conda environment just created, run `pip install -e <path_to_where_scivis_is>`.

## Structure of the redesign
The structure of the DTU 10MW RWT redesign is as follows:
1. All the files needed by HAWC2 are inside the folder `hawc_files`, which is then subdivided as follows:

```text
hawc_files/
├── master/
│   └── RWT .htc file + general redesign .htc
├── control/
│   └── Files not to be modified (except wpdata.100)
├── data/
│   └── Structural _st files, _ae files and .opt files of both turbines
├── ht_hawc2/
│   └── HTC files with different natural frequencies,
│       damping and Region 3 torque/power strategies
├── htc_hawc2s/
│   └── Rigid and flexible .htc files,
│       Campbell diagram and .multitsr cases
├── htc_steady/
│   └── HTC files for steady-state simulations
│       with different cases
├── htc_turb/
│   └── HTC files with two turbulence classes,
│       6 seeds for each wind speed
├── res_hawc2s/
│   └── Natural frequency evaluation files,
│       .ind and .pwr
├── saved_data/
│   └── .npy arrays for plotting and results comparison
├── stats/
│   └── .csv files from post-processing
│       for loads evaluation
└── cpostprocess_hawc2.py
    └── Post-processing script to generate
        .csv files with loads and operational statistics

```
  Finally, running the scripts will generate ulterior subfolders res_.

2. All the scripts needed tu create the .htc files and get the results are in the folder `scripts`. The order to follow is given by the inital number in the name of each script.
3. You might need to create a `plot` folder if it's not generated automatically after the first script. 









