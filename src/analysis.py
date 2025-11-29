import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage.transform import radon
# Assuming frequency_filter_fft is defined in preprocessing.py or available
from preprocessing import frequency_filter_fft
def analyze_statistics(df, dt=0.0016, dx=5.106500953873407, 
                        segment_name="segment", output_dir=None):
    """
    Compute basic statistics and identify active regions in the data.
    """

    data_flat = df.values.flatten()

    # Calculate key metrics
    mean = np.mean(data_flat)
    std = np.std(data_flat)
    min_val = np.min(data_flat)
    max_val = np.max(data_flat)
    median = np.median(data_flat)

    # Percentiles
    p95 = np.percentile(data_flat, 95)
    p99 = np.percentile(data_flat, 99)

    # Per-channel metrics
    # Variance across the time axis for each channel - used to identify high variance channels (possible noise)
    channel_vars = df.var(axis=0).values
    # Total energy - sum of squared strain at each time across all channels - used to identify events such as car passing
    time_energy = (df ** 2).sum(axis=1).values

    print("Basic statistic for a segment")
    print(f"Mean:   {mean:.6e}")
    print(f"Std:    {std:.6e}")
    print(f"Range (min-max values):  [{min_val:.2e}, {max_val:.2e}]")
    print(f"Median: {median:.6e}")
    print(f"95th percentile: {p95:.2e}")
    print(f"99th percentile: {p99:.2e}")

    # For visualization
    stats_dict = {
        'channel_vars': channel_vars,
        'time_energy': time_energy,
        'p99': p99
    }
    
    if output_dir:
        save_path = f"{output_dir}/{segment_name}_statistics.png"
        plot_statistical_analysis(df, stats_dict, dx=dx, dt=dt, save_path=save_path)
    
    return stats_dict


def plot_statistical_analysis(df, stats, dx=5.106500953873407, dt=0.0016, save_path=None):
    """
    Creates 2 plots:
    1. Temporal Variance per Channel - e. g. to detect high variance channels (possible noise)
    2. Total Energy Over Time - e. g. to identify events such as car passing
    """

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Statistical Analysis: Variance per Channel & Energy over Time', fontsize=14, fontweight='bold')

    # Plot 1: Variance per Channel
    ax = axes[0]
    channel_positions = df.columns.values
    ax.plot(channel_positions, stats['channel_vars'], linewidth=1.5, color='red', marker='o', markersize=4, label='Channel Variance')
    ax.set_xlabel('Position [m]', fontsize=11)
    ax.set_ylabel('Variance', fontsize=11)
    ax.set_title('Temporal Variance in Spatial Domain (per Channel)', fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend()

    # Plot 2: Total Energy Over Time
    ax = axes[1]
    time_axis = np.arange(len(stats['time_energy'])) * dt
    ax.plot(time_axis, stats['time_energy'], linewidth=0.8, color='purple')
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Total Energy - sum of squared strain', fontsize=11)
    ax.set_title('Total Energy Over Time, peaks suggest vehicles passing', fontsize=11)
    ax.grid(alpha=0.3)

    energy_threshold = stats['p99'] ** 2 * df.shape[1]  # Approximate 99th percentile for total energy
    events = stats['time_energy'] > energy_threshold
    event_times = time_axis[events]
    event_energies = stats['time_energy'][events]
    ax.scatter(event_times, event_energies, color='red', s=15,
                alpha=0.6, label='High-energy events')
    ax.axhline(y=energy_threshold, color='red', linestyle='--',
                alpha=0.5, linewidth=1.5, label='Event threshold')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()

def analyze_frequency_content(df, fs=625, output_dir='../output'):
    """
    Comprehensive frequency analysis using FFT:
    1. Space-Frequency Map (WHERE + WHAT frequencies)
    2. Spatial Variance of Frequencies (0-10 Hz zoom) (WHICH frequencies show movement)
    3. Spatial Variance of Frequencies (Full scale) (WHICH frequencies show movement)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("FREQUENCY CONTENT ANALYSIS")
    print("="*70)
    
    # Compute FFT
    fft_data = np.fft.rfft(df.values, axis=0)
    magnitude = np.abs(fft_data)
    power = magnitude ** 2
    avg_power = np.mean(power, axis=1)
    freqs = np.fft.rfftfreq(df.shape[0], d=1/fs)
    power_variance = np.var(power, axis=1)
    
    # Create plots (Now 1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(24, 6)) # <--- Changed to 1, 3 and increased size
    fig.suptitle('Frequency Content Analysis', fontsize=14, fontweight='bold')
    
    # =========================================================================
    # Plot 1 (axes[0]): Space-Frequency Map
    # =========================================================================
    ax = axes[0]
    magnitude_scaled = magnitude * 1e6
    magnitude_scaled[0, :] = 0  # Remove DC
    v_max = np.percentile(magnitude_scaled, 98)
    
    extent = [0, df.shape[1], 0, fs/2]
    im = ax.imshow(magnitude_scaled, aspect='auto', origin='lower', 
                    cmap='viridis', extent=extent, vmax=v_max, vmin=0)
    plt.colorbar(im, ax=ax, label='Magnitude ($10^{-6}$)')
    ax.set_ylim([0, 200])
    ax.set_xlabel('Space [Channel]', fontsize=11)
    ax.set_ylabel('Frequency [Hz]', fontsize=11)
    ax.set_title('Space-Frequency Map\n(WHERE + WHAT: Bright areas = Active frequencies at locations)', fontsize=11)
    
    # =========================================================================
    # Plot 2 (axes[1]): Spatial Variance (0-10 Hz Zoom) <--- NEW PLOT
    # =========================================================================
    ax = axes[1]
    ax.plot(freqs, power_variance, linewidth=1.0, color='blue') # Changed color for distinction
    ax.set_xlabel('Frequency [Hz]', fontsize=11)
    ax.set_ylabel('Spatial Variance', fontsize=11)
    ax.set_title('Spatial Variance of Frequencies\n(Zoom: 0-5 Hz)', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 5]) # <--- Set the x-limit to 0-5 Hz
    
    # =========================================================================
    # Plot 3 (axes[2]): Spatial Variance (Full Scale)
    # =========================================================================
    ax = axes[2]
    ax.plot(freqs, power_variance, linewidth=1.0, color='red')
    ax.set_xlabel('Frequency [Hz]', fontsize=11)
    ax.set_ylabel('Spatial Variance', fontsize=11)
    ax.set_title('Spatial Variance of Frequencies\n(Full Scale: 0-200 Hz)', fontsize=11) # Adjusted title
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 200]) 
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'frequency_content_analysis_3plots.png') # Changed filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")
    
    max_variance_freq = freqs[1:][np.argmax(power_variance[1:])]
    print(f"Maximum spatial variance at {max_variance_freq:.1f} Hz (likely vehicles)")
    
    return {
        'freqs': freqs,
        'avg_power': avg_power,
        'power_variance': power_variance,
        'magnitude': magnitude,
        'power': power
    }


def visualize_filtered_comparison(df, fs=625, output_dir='../output'):
    """
    Visual comparison of data filtered to different frequency bands.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # NOTE: This function requires the 'frequency_filter_fft' function from 'preprocessing'
    # which is not provided in this context, so it will raise an error if called
    # without a definition for 'frequency_filter_fft'.
    
    test_bands = [
        (0.2, 0.8, 'Very Low (0.2-0.8 Hz)'),
        (1.5, 3, 'Low (1.5-3 Hz)'),
        (2, 2.5, 'Mid-Low (2-2.5 Hz)'),
        (3.2, 4, 'Mid (3.2-4 Hz)'),
        (1, 4, 'Mid-High (1-4 Hz)'),
        (0.4, 0.6, 'High (0.4-0.6 Hz)')
    ]
    
    dt = 1.0 / fs
        
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    fig.suptitle('Filtered Data Comparison by Frequency Band', 
                    fontsize=14, fontweight='bold')
        
    for idx, (low, high, label) in enumerate(test_bands):
            df_filtered = frequency_filter_fft(df, dt=dt, lowcut=low, highcut=high)
            
            ax = axes[idx]
            data = np.abs(df_filtered.values)
            im = ax.imshow(data, aspect='auto', cmap='viridis',
                        vmin=data.min(), vmax=data.max(), interpolation='none')
            ax.set_title(f"{label}\n{low}-{high} Hz", fontsize=10, fontweight='bold')
            ax.set_ylabel('Time')
            ax.set_xlabel('Space (Channel)')
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    save_path = f"{output_dir}/filtered_comparison.png"
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")