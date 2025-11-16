import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cv2


def set_axis(x, no_labels=7):
    nx = x.shape[0]
    step_x = int(nx / (no_labels - 1))
    x_positions = np.arange(0, nx, step_x)
    x_labels = x[::step_x]
    return x_positions, x_labels


def visualize_das(df, title, vmin_percentile=3, vmax_percentile=99, figsize=(12, 16)):
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

    cax = fig.add_axes([ax.get_position().x1 + 0.06, ax.get_position().y0, 0.02, ax.get_position().height])
    plt.colorbar(im, cax=cax)

    x_positions, x_labels = set_axis(df.columns.values)
    ax.set_xticks(x_positions, np.round(x_labels))

    time_array = np.array([t.strftime('%H:%M:%S') for t in df.index.time])
    y_positions, y_labels = set_axis(time_array)
    ax.set_yticks(y_positions, y_labels)

    plt.tight_layout()
    plt.show()


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


def threshold_otsu(df):
    data = df.values.copy()
    data -= data.mean()
    data = np.abs(data)

    data_normalized = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)

    _, thresholded = cv2.threshold(data_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    data_thresholded = thresholded.astype(np.float64) / 255.0

    import pandas as pd
    df_thresholded = pd.DataFrame(data=data_thresholded, index=df.index, columns=df.columns)

    return df_thresholded


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


def downsample(df, time_factor=2, space_factor=1):
    data_downsampled = df.values[::time_factor, ::space_factor]

    import pandas as pd
    df_downsampled = pd.DataFrame(
        data=data_downsampled,
        index=df.index[::time_factor],
        columns=df.columns[::space_factor]
    )

    return df_downsampled


def preprocess_pipeline(df, dt=0.0016, show_steps=True):
    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    if show_steps:
        print("\nStep 0: Raw Data")
        visualize_das(df, "Raw DAS Data")

    print("\nStep 1: FFT frequency filtering (2-80 Hz)")
    print("Vehicle vibrations are in 2-80 Hz range")
    df_filtered = frequency_filter_fft(df, dt=dt, lowcut=1.0, highcut=50.0)
    if show_steps:
        visualize_das(df_filtered, "After FFT Bandpass Filter (2-80 Hz)")

    print("\nStep 2: Gaussian blur (noise reduction)")
    print("Smooth noise while preserving vehicle signals")
    df_blurred = gaussian_blur(df_filtered, kernel_size=5)
    if show_steps:
        visualize_das(df_blurred, "After Gaussian Blur")

    print("\nStep 3: Morphological closing")
    print("Connect broken vehicle tracks, fill small gaps")
    df_closed = morphological_closing(df_blurred, kernel_size=3)
    if show_steps:
        visualize_das(df_closed, "After Morphological Closing", vmin_percentile=0, vmax_percentile=100)

    print("\nStep 4: Thresholding (3rd-99th percentile)")
    df_processed = threshold_percentile(df_closed, percentile_low=3, percentile_high=99)
    if show_steps:
        visualize_das(df_processed, "Final: Moving Objects Only", vmin_percentile=3, vmax_percentile=99)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    return df_processed