import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.colors import Normalize
from skimage.filters import threshold_li
from skimage.morphology import binary_closing, disk, remove_small_objects


def squeeze_dataframe(df, axis, factor):
    """Reduce size by averaging - e.g. factor=0.5 makes half size."""
    if not 0 < factor < 1:
        raise ValueError("Squeeze factor must be between 0 and 1")
    
    int_factor = int(1 / factor)
    data = df.values
    
    if axis == 0:  # squeeze time
        new_height = data.shape[0] // int_factor
        cropped = data[:new_height * int_factor]
        squeezed = cropped.reshape(new_height, int_factor, data.shape[1]).mean(axis=1)
        return pd.DataFrame(squeezed, index=df.index[::int_factor][:new_height], columns=df.columns)
    else:  # squeeze space
        new_width = data.shape[1] // int_factor
        cropped = data[:, :new_width * int_factor]
        squeezed = cropped.reshape(data.shape[0], new_width, int_factor).mean(axis=2)
        return pd.DataFrame(squeezed, index=df.index, columns=df.columns[::int_factor][:new_width])


def stretch_dataframe(df, axis, factor):
    """Increase size by repeating values - factor must be integer."""
    if factor <= 1:
        raise ValueError("Stretch factor must be > 1")
    
    int_factor = int(factor)
    stretched = np.repeat(df.values, int_factor, axis=axis)
    
    if axis == 0:  # stretch time
        return pd.DataFrame(stretched, index=df.index.repeat(int_factor), columns=df.columns)
    else:  # stretch space
        return pd.DataFrame(stretched, index=df.index, columns=df.columns.repeat(int_factor))


def resize_to_square(df, target_size=750):
    """Make image square by squeezing time and stretching space."""
    print(f"\nResizing to {target_size}×{target_size}...")
    print(f"Initial shape: {df.shape[0]} × {df.shape[1]}")
    
    squeeze_factor_time = target_size / df.shape[0]
    stretch_factor_space = int(target_size / df.shape[1])
    
    print(f"Time squeeze factor: {squeeze_factor_time:.4f}")
    print(f"Space stretch factor: {stretch_factor_space}")
    
    df_resized = squeeze_dataframe(df, axis=0, factor=squeeze_factor_time)
    df_resized = stretch_dataframe(df_resized, axis=1, factor=stretch_factor_space)
    
    print(f"Final shape: {df_resized.shape[0]} × {df_resized.shape[1]}")
    print("✓ Resizing complete")
    return df_resized


def set_axis(x, no_labels=7):
    """Pick evenly spaced positions for axis labels."""
    nx = x.shape[0]
    step_x = int(nx / (no_labels - 1))
    x_positions = np.arange(0, nx, step_x)
    x_labels = x[::step_x]
    return x_positions, x_labels


def visualize_das(df, title="DAS Data", figsize=(12, 16), save_path=None, normalize=True):
    """Plot DAS data with proper scaling and time/space labels."""
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    
    if normalize:
        # Clip to 3rd-99th percentile for better contrast
        low, high = np.percentile(df, [3, 99]) 
        norm = Normalize(vmin=low, vmax=high, clip=True)
        im = ax.imshow(df, interpolation='none', aspect='auto', norm=norm)
    else:
        im = ax.imshow(df, interpolation='none', aspect='auto')
    
    plt.ylabel('Time')
    plt.xlabel('Space [m]')
    plt.title(title)
    
    # Add colorbar
    cax = fig.add_axes([ax.get_position().x1+0.06, ax.get_position().y0, 
                        0.02, ax.get_position().height])
    plt.colorbar(im, cax=cax)
    
    # Set ticks
    x_positions, x_labels = set_axis(df.columns.values)
    ax.set_xticks(x_positions, np.round(x_labels))
    
    y_positions, y_labels = set_axis(df.index.time)
    ax.set_yticks(y_positions, y_labels)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def preprocess_das_data(df_raw, target_size=750, threshold_multiplier=0.8, output_dir="."):
    """
    Full preprocessing: center → abs → resize → threshold → morphology.
    Returns binary image for detection and original processed for visualization.
    """
    print("\n" + "=" * 70)
    print("PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # Step 1: Center and take absolute value
    print("\n1. Centering data and taking absolute value...")
    df_display = df_raw.copy()
    df_display -= df_display.mean()
    df_abs = np.abs(df_display)
    
    # Step 2: Resize to square
    print("\n2. Resizing to square...")
    df_square = resize_to_square(df_abs, target_size=target_size)
    visualize_das(df_square, title="Resized to Square", 
                  save_path=os.path.join(output_dir, '01_resized_square.png'))
    
    # Step 3: Percentile thresholding
    print("\n3. Applying percentile-based thresholding...")
    low_thresh = np.percentile(df_square, 3)
    high_thresh = np.percentile(df_square, 99)
    data_thresholded = df_square.copy()
    data_thresholded[df_square < low_thresh] = 0
    data_thresholded = np.clip(data_thresholded, 0, high_thresh)
    print(f"   Threshold range: {low_thresh:.2e} to {high_thresh:.2e}")
    
    df_thresholded = pd.DataFrame(data=data_thresholded, 
                                   index=df_square.index, 
                                   columns=df_square.columns)
    visualize_das(df_thresholded, title="Percentile Thresholded Data", 
                  normalize=False,
                  save_path=os.path.join(output_dir, '02_percentile_thresholded.png'))
    
    # Step 4: Normalize to [0, 1]
    print("\n4. Normalizing to [0, 1]...")
    image = df_thresholded.values.copy()
    image_norm = (image - image.min()) / (image.max() - image.min())
    
    # Step 5: Li adaptive thresholding - finds optimal threshold automatically
    print("\n5. Applying Li adaptive thresholding...")
    thresh = threshold_li(image_norm) * threshold_multiplier
    print(f"   Li threshold: {threshold_li(image_norm):.4f}")
    print(f"   Adjusted threshold: {thresh:.4f} (multiplier: {threshold_multiplier})")
    
    binary = image_norm > thresh
    print(f"   Pixels above threshold: {binary.sum() / binary.size * 100:.2f}%")
    
    binary_df = pd.DataFrame(binary, index=df_thresholded.index, columns=df_thresholded.columns)
    visualize_das(binary_df, title="Binary Image (Li Threshold)", 
                  normalize=False,
                  save_path=os.path.join(output_dir, '03_binary_thresholded.png'))
    
    # Step 6: Morphological closing - fills small gaps
    print("\n6. Applying morphological closing...")
    binary = binary_closing(binary, disk(2))
    binary_df = pd.DataFrame(binary, index=df_thresholded.index, columns=df_thresholded.columns)
    visualize_das(binary_df, title="After Morphological Closing", 
                  normalize=False,
                  save_path=os.path.join(output_dir, '04_morphological_closing.png'))
    
    # Step 7: Remove small noise objects
    print("\n7. Removing small objects...")
    binary = remove_small_objects(binary, min_size=100)
    binary_df = pd.DataFrame(binary, index=df_thresholded.index, columns=df_thresholded.columns)
    visualize_das(binary_df, title="After Removing Small Objects", 
                  normalize=False,
                  save_path=os.path.join(output_dir, '05_cleaned_binary.png'))
    
    print("\n✓ Preprocessing complete")
    print("=" * 70)
    
    return binary_df, binary, df_thresholded, image


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