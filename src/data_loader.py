"""
Data Loader Module
Handles loading and concatenating DAS data files
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import glob
import os


class DASDataLoader:
    """Class for loading and managing DAS data"""

    def __init__(self, data_path='../data/', dx=5.106500953873407, dt=0.0016):
        self.data_path = data_path
        self.dx = dx
        self.dt = dt
        self.sampling_rate = 1 / dt
        self.nyquist_freq = self.sampling_rate / 2

    def find_files_in_range(self, start_time, end_time, date='20240507'):
        # Parse start and end times
        start_dt = datetime.strptime(f'{date} {start_time}', '%Y%m%d %H%M%S')
        end_dt = datetime.strptime(f'{date} {end_time}', '%Y%m%d %H%M%S')

        # Calculate duration and number of files
        duration_seconds = (end_dt - start_dt).total_seconds()
        n_files = int(duration_seconds / 10) + 1  # Each file is 10 seconds

        # Generate expected filenames
        filenames = []
        current_time = start_dt

        for i in range(n_files):
            time_str = current_time.strftime('%H%M%S')
            filename = os.path.join(self.data_path, f'{time_str}.npy')
            filenames.append(filename)
            current_time += timedelta(seconds=10)

        return filenames

    def load_segment(self, start_time, end_time, date='20240507', verbose=True):
        filenames = self.find_files_in_range(start_time, end_time, date)

        data_list = []
        loaded_files = []
        missing_files = []

        if verbose:
            print("=" * 60)
            print(f"Loading data from {start_time} to {end_time}")
            print("=" * 60)

        for fname in filenames:
            try:
                data_chunk = np.load(fname)
                data_list.append(data_chunk)
                loaded_files.append(fname)
            except FileNotFoundError:
                missing_files.append(fname)

        if not data_list:
            raise ValueError("No files loaded! Check your data path and file names.")

        # Concatenate along time axis
        data = np.vstack(data_list)

        # Create metadata
        metadata = {
            'start_time': start_time,
            'end_time': end_time,
            'date': date,
            'n_files_expected': len(filenames),
            'n_files_loaded': len(loaded_files),
            'n_files_missing': len(missing_files),
            'missing_files': missing_files,
            'shape': data.shape,
            'duration_seconds': data.shape[0] * self.dt,
            'n_channels': data.shape[1],
            'spatial_extent_m': data.shape[1] * self.dx,
            'dx': self.dx,
            'dt': self.dt,
            'sampling_rate': self.sampling_rate
        }

        if verbose:
            print("\n" + "=" * 60)
            print("LOADING SUMMARY")
            print("=" * 60)
            print(f"Files loaded: {len(loaded_files)}/{len(filenames)}")
            print(f"Final shape: {data.shape}")
            print(f"Duration: {metadata['duration_seconds']:.2f} seconds")
            print(f"Spatial extent: {metadata['spatial_extent_m']:.1f} meters")
            print(f"Sampling rate: {self.sampling_rate:.1f} Hz")
            print("=" * 60 + "\n")

        return data, metadata

    def create_dataframe(self, data, start_time, date='20240507'):
        # Create time index
        time_start = datetime.strptime(f'{date} {start_time}', '%Y%m%d %H%M%S')
        time_index = pd.date_range(start=time_start, periods=len(data), freq=f'{self.dt}s')

        # Create spatial columns (in meters)
        spatial_columns = np.arange(data.shape[1]) * self.dx

        # Create DataFrame
        df = pd.DataFrame(data=data, index=time_index, columns=spatial_columns)

        return df

    def load_and_prepare(self, start_time, end_time, date='20240507', verbose=True):
        data, metadata = self.load_segment(start_time, end_time, date, verbose)
        df = self.create_dataframe(data, start_time, date)

        if verbose:
            print("DataFrame Info:")
            print(f"  Time range: {df.index[0]} to {df.index[-1]}")
            print(f"  Spatial range: {df.columns[0]:.1f} m to {df.columns[-1]:.1f} m")
            print(f"  Shape: {df.shape}")
            print()

        return data, df, metadata

def get_available_files(data_path='../data/'):
    files = glob.glob(os.path.join(data_path, '*.npy'))
    files.sort()
    return files
