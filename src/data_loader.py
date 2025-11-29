"""
Data Loader Module
Simple functions for loading DAS data files
"""

import numpy as np
import pandas as pd
import datetime
import glob
import os


# Global constants
DATA_PATH = '../data/'
DX = 5.106500953873407  # Spatial resolution [m]
DT = 0.0016             # Temporal resolution [s]


def get_available_files(data_path=DATA_PATH):

    files = glob.glob(os.path.join(data_path, '*.npy'))
    files.sort()
    return files
def load_das_segment(start_time, end_time, data_path=DATA_PATH, 
                     dx=DX, dt=DT, verbose=True):
    """Load DAS data for a specific time segment."""
    # Get all available files
    all_files = glob.glob(os.path.join(data_path, '*.npy'))
    all_files.sort()

    if len(all_files) == 0:
        raise ValueError(f"No .npy files found in {data_path}")

    # Filter files in time range
    files_to_load = []
    for fpath in all_files:
        fname = os.path.basename(fpath).replace('.npy', '')
        if start_time <= fname <= end_time:
            files_to_load.append(fpath)

    if len(files_to_load) == 0:
        raise ValueError(f"No files found between {start_time} and {end_time}")

    # Load and stack all files
    data_list = []
    for fpath in files_to_load:
        fname = os.path.basename(fpath)
        try:
            data_chunk = np.load(fpath)
            data_list.append(data_chunk)
            if verbose:
                print(f"✓ {fname} - Shape: {data_chunk.shape}")
        except Exception as e:
            if verbose:
                print(f"✗ {fname} - Error: {e}")

    # Stack vertically (concatenate in time)
    data = np.vstack(data_list)

    if verbose:
        print("\n" + "="*60)
        print("LOADING COMPLETE")
        print("="*60)
        print(f"Files loaded: {len(files_to_load)}")
        print(f"Data shape: {data.shape}")
        print(f"Duration: {data.shape[0] * dt:.2f} seconds")
        print(f"Spatial extent: {data.shape[1] * dx:.1f} meters")
        print("="*60 + "\n")

    # Create DataFrame with time and position labels
    time_start = datetime.datetime.strptime(f'20240507 {start_time}', '%Y%m%d %H%M%S')
    time_index = pd.date_range(start=time_start, periods=len(data), freq=f'{dt}s')
    spatial_columns = np.arange(data.shape[1]) * dx

    df = pd.DataFrame(data=data, index=time_index, columns=spatial_columns)

    return df