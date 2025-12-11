# Design code for Assignments 1 through 4

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

## Where to put code

* Code to generate your design files:
    * In `hawc_files`
* Code to evaluate your design, make plots, etc.:
    * In `scripts`

## Overview of design code

![Overview of LAC code](diagram_lac_code.png)