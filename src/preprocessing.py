import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.colors import Normalize
from skimage.filters import threshold_li
from skimage.morphology import binary_closing, disk, remove_small_objects


def squeeze_dataframe(df, axis, factor):
    """Reduce size by averaging."""
    int_factor = int(1 / factor)
    data = df.values
    
    if axis == 0:  # Time
        new_h = data.shape[0] // int_factor
        squeezed = data[:new_h * int_factor].reshape(new_h, int_factor, -1).mean(axis=1)
        return pd.DataFrame(squeezed, index=df.index[::int_factor][:new_h], columns=df.columns)
    else:  # Space
        new_w = data.shape[1] // int_factor
        squeezed = data[:, :new_w * int_factor].reshape(data.shape[0], new_w, int_factor).mean(axis=2)
        return pd.DataFrame(squeezed, index=df.index, columns=df.columns[::int_factor][:new_w])


def stretch_dataframe(df, axis, factor):
    """Increase size by repeating values."""
    int_factor = int(factor)
    stretched = np.repeat(df.values, int_factor, axis=axis)
    
    if axis == 0:
        return pd.DataFrame(stretched, index=df.index.repeat(int_factor), columns=df.columns)
    else:
        return pd.DataFrame(stretched, index=df.index, columns=df.columns.repeat(int_factor))


def resize_to_square(df, target_size=750):
    """Resize to square by squeezing time and stretching space."""
    df = squeeze_dataframe(df, axis=0, factor=target_size / df.shape[0])
    df = stretch_dataframe(df, axis=1, factor=int(target_size / df.shape[1]))
    return df


def set_axis(x, no_labels=7):
    """Get evenly spaced axis positions and labels."""
    step = int(x.shape[0] / (no_labels - 1))
    positions = np.arange(0, x.shape[0], step)
    labels = x[::step]
    return positions, labels


def visualize_das(df, title="DAS Data", figsize=(12, 16), save_path=None, normalize=True):
    """Plot DAS data."""
    fig, ax = plt.subplots(figsize=figsize)
    
    if normalize:
        low, high = np.percentile(df, [3, 99])
        norm = Normalize(vmin=low, vmax=high, clip=True)
        im = ax.imshow(df, interpolation='none', aspect='auto', norm=norm)
    else:
        im = ax.imshow(df, interpolation='none', aspect='auto')
    
    ax.set_ylabel('Time')
    ax.set_xlabel('Space [m]')
    ax.set_title(title)
    
    # Axis labels
    x_pos, x_labels = set_axis(df.columns.values)
    ax.set_xticks(x_pos, np.round(x_labels))
    
    y_pos, y_labels = set_axis(df.index.time)
    ax.set_yticks(y_pos, y_labels)
    
    plt.colorbar(im, ax=ax)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def preprocess_das_data(df_raw, target_size=750, threshold_multiplier=0.8, output_dir="."):
    """Preprocessing pipeline: center → abs → resize → threshold → morphology."""
    print(f"\n{'='*70}\nPREPROCESSING\n{'='*70}")
    
    # Center and absolute value
    df_abs = np.abs(df_raw - df_raw.mean())
    
    # Resize
    df_square = resize_to_square(df_abs, target_size)
    visualize_das(df_square, "Resized", 
                  save_path=os.path.join(output_dir, '01_resized_square.png'))
    
    # Percentile threshold
    low, high = np.percentile(df_square, [3, 99])
    data_thresh = np.clip(df_square.values, low, None)
    data_thresh = np.clip(data_thresh, 0, high)
    
    df_thresh = pd.DataFrame(data_thresh, index=df_square.index, columns=df_square.columns)
    visualize_das(df_thresh, "Percentile Thresholded", normalize=False,
                  save_path=os.path.join(output_dir, '02_percentile_thresholded.png'))
    
    # Normalize to [0, 1]
    image = df_thresh.values
    image_norm = (image - image.min()) / (image.max() - image.min())
    
    # Li threshold
    thresh = threshold_li(image_norm) * threshold_multiplier
    binary = image_norm > thresh
    
    binary_df = pd.DataFrame(binary, index=df_thresh.index, columns=df_thresh.columns)
    visualize_das(binary_df, "Binary (Li Threshold)", normalize=False,
                  save_path=os.path.join(output_dir, '03_binary_thresholded.png'))
    
    # Morphological closing
    binary = binary_closing(binary, disk(2))
    binary_df = pd.DataFrame(binary, index=df_thresh.index, columns=df_thresh.columns)
    visualize_das(binary_df, "Morphological Closing", normalize=False,
                  save_path=os.path.join(output_dir, '04_morphological_closing.png'))
    
    # Remove small objects
    binary = remove_small_objects(binary, min_size=100)
    binary_df = pd.DataFrame(binary, index=df_thresh.index, columns=df_thresh.columns)
    visualize_das(binary_df, "Cleaned Binary", normalize=False,
                  save_path=os.path.join(output_dir, '05_cleaned_binary.png'))
    
    print("✓ Preprocessing complete\n")
    
    return binary_df, binary, df_thresh, image

def frequency_filter_fft(df, dt=0.0016, lowcut=2.0, highcut=80.0):
    """Apply FFT bandpass filter to each channel."""

    n_samples = df.shape[0]
    n_channels = df.shape[1]
    filtered_data = np.zeros_like(df.values)

    print(f"FFT filtering {n_channels} channels: {lowcut}-{highcut} Hz")

    for i in range(n_channels):
        signal_data = df.iloc[:, i].values
        fft = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(n_samples, dt)
        mask = (np.abs(freqs) >= lowcut) & (np.abs(freqs) <= highcut)
        fft_filtered = fft * mask
        filtered_signal = np.fft.ifft(fft_filtered)
        filtered_data[:, i] = np.real(filtered_signal)
    
    return pd.DataFrame(data=filtered_data, index=df.index, columns=df.columns)