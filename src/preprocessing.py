import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cv2
import os

def set_axis(x, no_labels=7):
    nx = x.shape[0]
    step_x = int(nx / (no_labels - 1))
    x_positions = np.arange(0, nx, step_x)
    x_labels = x[::step_x]
    return x_positions, x_labels

def visualize_das(df, output_dir, title, vmin_percentile=3, vmax_percentile=99, figsize=(12, 16)):
    import os
    

    # Build filename automatically from the title
    safe_title = title.replace(" ", "_").replace(":", "_")
    save_path = os.path.join(output_dir, f"{safe_title}.png")

    fig = plt.figure(figsize=figsize)
    ax = plt.axes()

    data = df.values.copy()
    data -= data.mean()
    data = np.abs(data)

    low, high = np.percentile(data, [vmin_percentile, vmax_percentile])
    norm = Normalize(vmin=low, vmax=high, clip=True)

    im = ax.imshow(data, interpolation='none', aspect='auto', norm=norm)
    plt.ylabel('time')
    plt.xlabel('space [m]')
    plt.title(title)

    cax = fig.add_axes([ax.get_position().x1 + 0.06,
                        ax.get_position().y0,
                        0.02,
                        ax.get_position().height])
    plt.colorbar(im, cax=cax)

    x_positions, x_labels = set_axis(df.columns.values)
    ax.set_xticks(x_positions, np.round(x_labels))

    time_array = np.array([t.strftime('%H:%M:%S') for t in df.index.time])
    y_positions, y_labels = set_axis(time_array)
    ax.set_yticks(y_positions, y_labels)

    plt.tight_layout()

    # Save instead of showing
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved figure to: {save_path}")

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


def gaussian_blur(df, kernel_size=5):
    data = df.values.copy()
    data_normalized = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(data_normalized, (kernel_size, kernel_size), 0)

    data_blurred = blurred.astype(np.float64) / 255.0 * (data.max() - data.min()) + data.min()

    import pandas as pd
    df_blurred = pd.DataFrame(data=data_blurred, index=df.index, columns=df.columns)

    return df_blurred


def morphological_closing(df, kernel_size=3):
    data = df.values.copy()

    threshold = np.percentile(np.abs(data), 90)
    data_binary = (np.abs(data) > threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(data_binary, cv2.MORPH_CLOSE, kernel)

    data_closed = data.copy()
    data_closed[closed == 0] = 0

    import pandas as pd
    df_closed = pd.DataFrame(data=data_closed, index=df.index, columns=df.columns)

    return df_closed

def threshold_percentile(df, percentile_low=3, percentile_high=99):
    data = df.values.copy()
    data -= data.mean()
    data_abs = np.abs(data)

    low_thresh = np.percentile(data_abs, percentile_low)
    high_thresh = np.percentile(data_abs, percentile_high)

    data_thresholded = data_abs.copy()
    data_thresholded[data_abs < low_thresh] = 0
    data_thresholded = np.clip(data_thresholded, 0, high_thresh)

    print(f"Threshold range: {low_thresh:.2e} to {high_thresh:.2e}")
    print(
        f"Kept {((data_abs >= low_thresh) & (data_abs <= high_thresh)).sum() / data_abs.size * 100:.2f}% of data points")

    import pandas as pd
    df_thresholded = pd.DataFrame(data=data_thresholded, index=df.index, columns=df.columns)

    return df_thresholded

def remove_spatial_median(df):
    """Remove spatial median to suppress noisy channels"""
    data = df.values.copy()
    spatial_median = np.median(data, axis=1, keepdims=True)
    data_cleaned = data - spatial_median
    
    import pandas as pd
    return pd.DataFrame(data=data_cleaned, index=df.index, columns=df.columns)

def remove_temporal_median_spectrum(df, dt=0.0016):
    """
    For each channel, compute the median frequency spectrum over time
    and subtract it. This removes persistent frequency components (constant "hum").
    """
    n_samples = df.shape[0]
    n_channels = df.shape[1]
    
    data_cleaned = df.values.copy()
    
    print(f"\nRemoving temporal median spectrum from each channel...")
    
    for ch in range(n_channels):
        signal = df.iloc[:, ch].values
        
        # Compute spectrogram (time-frequency representation)
        window_size = 512
        hop_size = 128
        n_windows = (n_samples - window_size) // hop_size + 1
        
        spectrogram = []
        for i in range(n_windows):
            start = i * hop_size
            end = start + window_size
            if end > n_samples:
                break
            
            window = signal[start:end]
            fft = np.fft.fft(window)
            spectrogram.append(fft)
        
        spectrogram = np.array(spectrogram)
        
        # Find median spectrum (persistent frequency profile)
        median_spectrum = np.median(spectrogram, axis=0)
        
        # Subtract median spectrum from each window
        cleaned_spectrogram = spectrogram - median_spectrum
        
        # Reconstruct signal using overlap-add
        cleaned_signal = np.zeros(n_samples)
        window_count = np.zeros(n_samples)
        
        for i in range(n_windows):
            start = i * hop_size
            end = start + window_size
            if end > n_samples:
                break
            
            reconstructed_window = np.fft.ifft(cleaned_spectrogram[i])
            cleaned_signal[start:end] += np.real(reconstructed_window)
            window_count[start:end] += 1
        
        # Average overlapping regions
        cleaned_signal = cleaned_signal / (window_count + 1e-10)
        data_cleaned[:, ch] = cleaned_signal
        
        print(f"  ✓ Channel {ch} (@ {df.columns[ch]:.1f}m): Denoised")
    
    import pandas as pd
    return pd.DataFrame(data=data_cleaned, index=df.index, columns=df.columns)
def preprocess_pipeline(df, dt, output_dir, show_steps=True):
    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    if show_steps:
        visualize_das(df, output_dir, "0_Raw DAS Data")



    print("\nStep 1: FFT frequency filtering (60-90 Hz)")
    df_filtered = frequency_filter_fft(df, dt=dt, lowcut=60.0, highcut=90.0)
    if show_steps:
        visualize_das(df_filtered, output_dir, "After FFT Bandpass Filter")
        print("\nStep 0.5: Remove spatial median (suppress noisy channels)")
    
        print("\nStep 0.5: Remove temporal median spectrum (denoise vertical stripes)")
    df_filtered = remove_temporal_median_spectrum(df_filtered, dt=dt)
    if show_steps:
        visualize_das(df, output_dir, "1.5_After_Median_Spectrum_Removal")


    df_filtered = remove_spatial_median(df_filtered)
    if show_steps:
        visualize_das(df, output_dir, "After Spatial Median Removal1")

    print("\nStep 2: Gaussian blur (noise reduction)")
    print("Smooth noise while preserving vehicle signals")
    df_blurred = gaussian_blur(df_filtered, kernel_size=5)
    if show_steps:
        visualize_das(df_blurred, output_dir, "After Gaussian Blur")

    print("\nStep 3: Morphological closing")
    print("Connect broken vehicle tracks, fill small gaps")
    df_closed = morphological_closing(df_blurred, kernel_size=3)
    if show_steps:
        visualize_das(df_closed, output_dir, "After Morphological Closing", vmin_percentile=0, vmax_percentile=100)

    print("\nStep 4: Thresholding (3rd-99th percentile)")
    df_processed = threshold_percentile(df_closed, percentile_low=3, percentile_high=99)
    if show_steps:
        visualize_das(df_processed, output_dir, "Final: Moving Objects Only", vmin_percentile=3, vmax_percentile=99)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    return df_processed