import numpy as np
import matplotlib.pyplot as plt
import os
from preprocessing import frequency_filter_fft


def analyze_statistics(df, dt=0.0016, dx=5.106500953873407, 
                        segment_name="segment", output_dir=None):
    """Calculate basic stats and identify high-energy events (likely vehicles)."""
    data_flat = df.values.flatten()

    mean = np.mean(data_flat)
    std = np.std(data_flat)
    min_val = np.min(data_flat)
    max_val = np.max(data_flat)
    median = np.median(data_flat)
    p95 = np.percentile(data_flat, 95)
    p99 = np.percentile(data_flat, 99)

    # Variance per channel - helps spot noisy channels
    channel_vars = df.var(axis=0).values
    # Total energy over time - spikes indicate events
    time_energy = (df ** 2).sum(axis=1).values

    print("Basic statistics for segment")
    print(f"Mean:   {mean:.6e}")
    print(f"Std:    {std:.6e}")
    print(f"Range:  [{min_val:.2e}, {max_val:.2e}]")
    print(f"Median: {median:.6e}")
    print(f"95th percentile: {p95:.2e}")
    print(f"99th percentile: {p99:.2e}")

    stats_dict = {
        'channel_vars': channel_vars,
        'time_energy': time_energy,
        'p99': p99
    }
    
    if output_dir:
        save_path = f"{output_dir}/{segment_name}_statistics.png"
        plot_statistical_analysis(df, stats_dict, dx=dx, dt=dt, save_path=save_path)
        save_path = f"{output_dir}/{segment_name}_distributions.png"
        plot_distributions(df, save_path)
    
    return stats_dict


def plot_distributions(df, save_path):
    original_values = df.values.flatten()
    
    # Apply thresholding
    low_thresh = np.percentile(original_values, 3)
    high_thresh = np.percentile(original_values, 99)
    thresholded_values = original_values.copy()
    thresholded_values[original_values < low_thresh] = 0
    thresholded_values = np.clip(thresholded_values, 0, high_thresh)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Original
    axes[0].hist(original_values, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Value', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Original Distribution', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Right: Thresholded
    axes[1].hist(thresholded_values, bins=100, edgecolor='black', alpha=0.7, color='green')
    axes[1].set_xlabel('Value', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('After [3%, 99%] Percentile Thresholding', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_statistical_analysis(df, stats, dx=5.106500953873407, dt=0.0016, save_path=None):
    """Plot variance per channel and energy over time."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Statistical Analysis: Variance per Channel & Energy over Time', 
                 fontsize=14, fontweight='bold')

    # Plot 1: Variance per channel - spots problematic channels
    ax = axes[0]
    channel_positions = df.columns.values
    ax.plot(channel_positions, stats['channel_vars'], linewidth=1.5, color='red', 
            marker='o', markersize=4, label='Channel Variance')
    ax.set_xlabel('Position [m]', fontsize=11)
    ax.set_ylabel('Variance', fontsize=11)
    ax.set_title('Temporal Variance in Spatial Domain (per Channel)', fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend()

    # Plot 2: Total energy over time - peaks suggest vehicles
    ax = axes[1]
    time_axis = np.arange(len(stats['time_energy'])) * dt
    ax.plot(time_axis, stats['time_energy'], linewidth=0.8, color='purple')
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Total Energy (sum of squared strain)', fontsize=11)
    ax.set_title('Total Energy Over Time - peaks suggest vehicles passing', fontsize=11)
    ax.grid(alpha=0.3)

    # Mark high-energy events
    energy_threshold = stats['p99'] ** 2 * df.shape[1]
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
    FFT analysis: space-frequency map + spatial variance plots.
    Helps identify dominant frequencies (vehicles typically 1-4 Hz).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("FREQUENCY CONTENT ANALYSIS")
    print("="*70)
    
    # Compute FFT for all channels
    fft_data = np.fft.rfft(df.values, axis=0)
    magnitude = np.abs(fft_data)
    power = magnitude ** 2
    freqs = np.fft.rfftfreq(df.shape[0], d=1/fs)
    power_variance = np.var(power, axis=1)  # variance across space
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle('Frequency Content Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: Space-Frequency heatmap - where and what frequencies
    ax = axes[0]
    magnitude_scaled = magnitude * 1e6
    magnitude_scaled[0, :] = 0  # remove DC component
    v_max = np.percentile(magnitude_scaled, 98)
    
    extent = [0, df.shape[1], 0, fs/2]
    im = ax.imshow(magnitude_scaled, aspect='auto', origin='lower', 
                    cmap='viridis', extent=extent, vmax=v_max, vmin=0)
    plt.colorbar(im, ax=ax, label='Magnitude ($10^{-6}$)')
    ax.set_ylim([0, 200])
    ax.set_xlabel('Space [Channel]', fontsize=11)
    ax.set_ylabel('Frequency [Hz]', fontsize=11)
    ax.set_title('Space-Frequency Map\n(Bright = active frequencies at locations)', fontsize=11)
    
    # Plot 2: Spatial variance zoomed to low frequencies
    ax = axes[1]
    ax.plot(freqs, power_variance, linewidth=1.0, color='blue')
    ax.set_xlabel('Frequency [Hz]', fontsize=11)
    ax.set_ylabel('Spatial Variance', fontsize=11)
    ax.set_title('Spatial Variance of Frequencies\n(Zoom: 0-5 Hz)', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 5])
    
    # Plot 3: Spatial variance full scale
    ax = axes[2]
    ax.plot(freqs, power_variance, linewidth=1.0, color='red')
    ax.set_xlabel('Frequency [Hz]', fontsize=11)
    ax.set_ylabel('Spatial Variance', fontsize=11)
    ax.set_title('Spatial Variance of Frequencies\n(Full Scale: 0-200 Hz)', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 200])
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'frequency_content_analysis_3plots.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")
    
    max_variance_freq = freqs[1:][np.argmax(power_variance[1:])]
    print(f"Maximum spatial variance at {max_variance_freq:.1f} Hz (likely vehicles)")
    
    return {
        'freqs': freqs,
        'power_variance': power_variance,
        'magnitude': magnitude,
        'power': power
    }


def visualize_filtered_comparison(df, fs=625, output_dir='../output'):
    """Compare data filtered to different frequency bands - helps choose best range."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Different frequency bands to test
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