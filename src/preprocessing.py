import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import Normalize

import cv2

import os

fs = 625

def squeeze_dataframe(df, axis, factor):
    """
    Squeeze DataFrame along specified axis using averaging.
    
    Parameters:
    - df: input DataFrame
    - axis: 0 for time (rows), 1 for space (columns)
    - factor: squeeze factor (< 1). e.g., 0.5 = squeeze to half size
    """
    if not 0 < factor < 1:
        raise ValueError("Squeeze factor must be between 0 and 1")
    
    int_factor = int(1 / factor)
    data = df.values
    
    if axis == 0:  # Time squeeze
        new_height = data.shape[0] // int_factor
        cropped = data[:new_height * int_factor]
        squeezed = cropped.reshape(new_height, int_factor, data.shape[1]).mean(axis=1)
        new_index = df.index[::int_factor][:new_height]
        new_columns = df.columns
    else:  # Space squeeze
        new_width = data.shape[1] // int_factor
        cropped = data[:, :new_width * int_factor]
        squeezed = cropped.reshape(data.shape[0], new_width, int_factor).mean(axis=2)
        new_index = df.index
        new_columns = df.columns[::int_factor][:new_width]
    
    return pd.DataFrame(squeezed, index=new_index, columns=new_columns)


def stretch_dataframe(df, axis, factor):
    """
    Stretch DataFrame along specified axis using repetition.
    
    Parameters:
    - df: input DataFrame
    - axis: 0 for time (rows), 1 for space (columns)
    - factor: stretch factor (> 1). Must be integer.
    """
    if factor <= 1:
        raise ValueError("Stretch factor must be > 1")
    
    int_factor = int(factor)
    stretched = np.repeat(df.values, int_factor, axis=axis)
    
    if axis == 0:  # Time stretch
        new_index = df.index.repeat(int_factor)
        new_columns = df.columns
    else:  # Space stretch
        new_index = df.index
        new_columns = df.columns.repeat(int_factor)
    
    return pd.DataFrame(stretched, index=new_index, columns=new_columns)


def resize_to_square(df, target_size=1000):
    """Resize DataFrame to square dimensions using squeeze/stretch."""
    print(f"\nResizing to {target_size}×{target_size}...")
    
    initial_rows = df.shape[0]
    initial_cols = df.shape[1]
    
    print(f"Initial shape: {initial_rows} × {initial_cols}")
    
    # Calculate factors
    squeeze_factor_time = target_size / initial_rows
    stretch_factor_space = int(target_size / initial_cols)
    
    print(f"Time squeeze factor: {squeeze_factor_time:.4f}")
    print(f"Space stretch factor: {stretch_factor_space}")
    
    # Apply transformations
    df_resized = squeeze_dataframe(df, axis=0, factor=squeeze_factor_time)
    df_resized = stretch_dataframe(df_resized, axis=1, factor=stretch_factor_space)
    
    print(f"Final shape: {df_resized.shape[0]} × {df_resized.shape[1]}")
    print("✓ Resizing complete")
    
    return df_resized

def set_axis(x, no_labels=7):

    nx = x.shape[0]

    step_x = int(nx / (no_labels - 1))

    x_positions = np.arange(0, nx, step_x)

    x_labels = x[::step_x]

    return x_positions, x_labels

def visualize_das(df, title="DAS Data", figsize=(12, 16), save_path=None, normalize=True):
    """Visualize DAS data with proper scaling and labels."""
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    
    # Preprocessing: center and take absolute value
    
    if normalize:
        df_display = df.copy()
        # Normalize to percentiles for better visualization
        low, high = np.percentile(df_display, [3, 99]) 
        norm = Normalize(vmin=low, vmax=high, clip=True)
        im = ax.imshow(df_display, interpolation='none', aspect='auto', norm=norm)
    else:
        im = ax.imshow(df, interpolation='none', aspect='auto')
    plt.ylabel('Time')
    plt.xlabel('Space [m]')
    plt.title(title)
    
    # Add colorbar
    cax = fig.add_axes([ax.get_position().x1+0.06, ax.get_position().y0, 
                        0.02, ax.get_position().height])
    plt.colorbar(im, cax=cax)
    
    # Set axis ticks
    x_positions, x_labels = set_axis(df.columns.values)
    ax.set_xticks(x_positions, np.round(x_labels))
    
    y_positions, y_labels = set_axis(df.index.time)
    ax.set_yticks(y_positions, y_labels)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")


def squeeze_image(img, axis, factor):
    """
    Squeeze image along specified axis using averaging for sharp results.
    
    Parameters:
    - img: input image
    - axis: 0 for vertical squeeze, 1 for horizontal squeeze
    - factor: squeeze factor (< 1). e.g., 0.5 = squeeze to half size
    
    Returns:
    - squeezed image
    """
    if factor >= 1:
        raise ValueError("Squeeze factor must be < 1")
    
    int_factor = int(1 / factor)
    
    if axis == 0:  # Vertical squeeze
        new_height = img.shape[0] // int_factor
        return img[:new_height * int_factor].reshape(new_height, int_factor, img.shape[1]).mean(axis=1)
    else:  # Horizontal squeeze
        new_width = img.shape[1] // int_factor
        return img[:, :new_width * int_factor].reshape(img.shape[0], new_width, int_factor).mean(axis=2)


def stretch_image(img, axis, factor):
    """
    Stretch image along specified axis using repetition for sharp results.
    
    Parameters:
    - img: input image
    - axis: 0 for vertical stretch, 1 for horizontal stretch
    - factor: stretch factor (> 1). e.g., 2.0 = stretch to double size
    
    Returns:
    - stretched image
    """
    if factor <= 1:
        raise ValueError("Stretch factor must be > 1")
    
    if factor == int(factor):
        return np.repeat(img, int(factor), axis=axis)
    
def preprocess_pipeline(df, dt, output_dir, show_steps=True):
    """
    1. Takes the absolute value of the data (after centering).
    2. Computes the 1D FFT for each channel and visualizes the averaged magnitude spectrum.
    3. Returns the data as a DataFrame (df_processed).
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("FREQUENCY CONTENT ANALYSIS (on Absolute Value Data)")
    print("=" * 70)

    # 1. Preprocessing: Center and take Absolute Value
    # Center the data first to remove channel-specific DC offset
    data_centered = df.values - df.values.mean(axis=0)
    data_abs = np.abs(data_centered)
    print("Preprocessing: Data centered by channel mean, then absolute value taken.")

    # Create the DataFrame to be returned
    df_processed = pd.DataFrame(data=data_abs, index=df.index, columns=df.columns)

    # 2. Compute FFT on Absolute Data for Visualization
    fft_data = np.fft.rfft(data_abs, axis=0)
    magnitude = np.abs(fft_data)

    # Calculate the frequency axis
    n_samples = df.shape[0]
    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)

    # Calculate average magnitude for plotting
    avg_magnitude = np.mean(magnitude, axis=1)

    # 3. Visualization: Average Magnitude Spectrum

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(freqs, avg_magnitude, linewidth=1.5, color="blue")

    # Find and highlight the dominant frequency (excluding DC/zero-frequency bin)
    max_freq_index = np.argmax(avg_magnitude[1:]) + 1
    max_freq = freqs[max_freq_index]

    ax.scatter(
        max_freq,
        avg_magnitude[max_freq_index],
        color="red",
        s=50,
        label=f"Peak @ {max_freq:.2f} Hz",
        zorder=5,
    )

    ax.set_title(f"Average Magnitude Spectrum (on Absolute Value Data)", fontsize=14)
    ax.set_xlabel("Frequency [Hz]", fontsize=11)
    ax.set_ylabel("Average Magnitude", fontsize=11)
    ax.set_xlim([0, fs / 2])  # Limit to Nyquist frequency
    ax.grid(alpha=0.4)
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, "absolute_fft_spectrum_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to: {save_path}")
    print(f"Dominant frequency: {max_freq:.2f} Hz")

    # 4. Return df_processed
    return df_processed


def frequency_filter_fft(df, dt=0.0016, lowcut=2.0, highcut=80.0):
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

    import pandas as pd

    df_filtered = pd.DataFrame(data=filtered_data, index=df.index, columns=df.columns)

    return df_filtered
