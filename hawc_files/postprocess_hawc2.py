"""Post-process a collection of time series into statistics.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd

from lacbox.io import ReadHAWC2
from lacbox.postprocess import eq_load


def process_statistics(res_dir, save_path, calc_del=True, m_vals=(3, 4, 5, 8, 10, 12)):
    """Make stats dataframe from HAWC2 time series, save as CSV.

    Parameters
    ----------
    res_dir : str or pathlib.Path
        Path to top-level folder with HAWC2-generated HDF5 time series. Can
        contain nested subfolders.
    save_path : str or pathlib.Path
        Path to the stats file to save as CSV.
    calc_del : boolean, optional
        Whether to calculate the 10-minute damage equivalent loads, which takes more
        time. Default is True.
    m_vals : iterable, optional
        The values of the Wöhler exponent for which to calculate DELs. Only used if
        `calc_del` is True. Default is `(3, 4, 5, 8, 10, 12)`.

    Returns
    -------
    res_df : pandas.DataFrame
        Dataframe with statistics.
    """
    # check save_path
    if not save_path.suffix:  # no extension given
        save_path = save_path.parent / (save_path.name + '.csv')
    if save_path.suffix.lower() != '.csv':
        raise ValueError('save_path must end with .csv!')

    # sanitize input path objects
    res_dir = Path(res_dir)
    save_path = Path(save_path)

    # get list of hdf5 time-series files in the folder(s)
    res_paths = list(res_dir.rglob('*.hdf5'))
    print(f'\n{len(res_paths)} HAWC2 HDF5 time series to process.\n')

    # define constants: statstics to calculate and columns of output dataframe
    base_stats = ['mean', 'max', 'min', 'std', '1%', '50%', '99%']  # base statistics
    stats = base_stats
    if calc_del is True:  # include extra if calculating DELs
        del_stats = ['del' + str(m) for m in m_vals]
        stats = base_stats + del_stats
    else:
        stats = base_stats
    cols = ['path', 'filename', 'subfolder', 'ichan', 'names',
            'units', 'desc'] + stats  # columns of stats_df

    # initialize output dataframe and begin loop over files
    for i, fpath in enumerate(res_paths):

        # get the filename and name of subfolder
        fpath = res_paths[i]
        filename = fpath.name
        subfolder = fpath.relative_to(res_dir).parent

        print(f'Processing {i+1}/{len(res_paths)}...')

        # load the data, turn to dataframe so we can use .describe()
        h2res = ReadHAWC2(fpath)
        time_df = pd.DataFrame(h2res.data)
        ichans = np.arange(time_df.shape[1]) + 1  # index from 1 to match HAWC2
        names, units, desc = h2res.chaninfo

        # initialize our stats dataframe for this file and set filenames, paths, etc.
        new_df = pd.DataFrame(columns=cols, index=ichans-1)
        new_df['path'] = str(fpath)
        new_df['filename'] = filename
        new_df['subfolder'] = str(subfolder)
        new_df['ichan'] = ichans
        new_df['names'] = names
        new_df['units'] = units
        new_df['desc'] = desc

        # calculate base states
        percentiles = []
        for stat in base_stats:
            match_res = re.search(r"(\d+)(?:%)", stat)
            if match_res is not None:
                percentiles.append(float(match_res.group(1)))
        new_df.loc[:, base_stats] = calc_stats(time_df, percentiles)

        # if DELs are requested to be calculated
        if calc_del is True:
            targets = ("Mx coo", "My coo", "Mz coo")
            chans_del = [(s, n) for s, n in zip(names, ichans)
                         if any(t in s for t in targets)]

            t = time_df[names.index("Time")]
            sim_time = int(np.ceil(t.max()-t.min()))

            for chan_name, chan_idx in chans_del:
                chan_vals = time_df.iloc[:, chan_idx-1].values
                new_df.loc[chan_idx-1, del_stats] = eq_load(chan_vals,
                                                            m=m_vals,
                                                            neq=sim_time)[0]

        # append to large df
        if i == 0:
            stats_df = new_df
        else:
            stats_df = pd.concat((stats_df, new_df), ignore_index=True)

    # fix all datatypes before saving
    stats_df = stats_df.astype({"ichan": int})
    for stat in stats:
        stats_df = stats_df.astype({stat: float})

    # save as csv
    stats_df.to_csv(save_path)

    print(f'\nResults saved to file "{str(save_path)}".\n')

    return stats_df

def calc_stats(df: pd.DataFrame, percentiles: list = (1, 50, 99)) -> np.ndarray:
    """Calculate column-wise statistics of a pandas DataFrame.

    Calculate mean, max, min, std deviation and specified percentiles for each
    column of a pandas DataFrame.


    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame to calculate the statistics for.
    percentiles : list, optional
        Percentiles to calculate. The default is [1, 50, 99].

    Returns
    -------
    stats : np.ndarray
        Numpy array with the calculated statistics with the metrics on axis 1
        and the columns of the dataframe on axis 0.

    """
    data = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    n_stats = 4+len(percentiles)
    stats = np.empty((data.shape[1], n_stats))

    stats[:, 0] = np.nanmean(data, axis=0)
    stats[:, 1] = np.nanmax(data, axis=0)
    stats[:, 2] = np.nanmin(data, axis=0)
    stats[:, 3] = np.nanstd(data, axis=0, ddof=1)
    stats[:, 4:n_stats] = np.nanpercentile(data, percentiles, axis=0).T

    return stats

# execute this block only if this file is run as script
if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parent

    # inputs
    model = "IEC_Ya_Later"
    RES_DIR = ROOT / 'res_turb' / model  # !!! TOP-LEVEL !!! directory with res files to process, NOT the subfolder
    CALC_DEL = True  # calculate DELs in the statistics? It takes longer.
    SAVE_PATH = ROOT / f'{model}_turb_stats.csv'  # where should I save the stats file?


    # call the function
    stats_df = process_statistics(RES_DIR, SAVE_PATH, calc_del=CALC_DEL)
