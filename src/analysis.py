import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage.transform import radon


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

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
    channel_vars = df.var(axis=0).values
    time_energy = (df ** 2).sum(axis=1).values

    print("\n" + "=" * 60)
    print("BASIC STATISTICS")
    print("=" * 60)
    print(f"Mean:   {mean:.6e}")
    print(f"Std:    {std:.6e}")
    print(f"Range:  [{min_val:.2e}, {max_val:.2e}]")
    print(f"Median: {median:.6e}")
    print(f"\nNoise floor (95%): ±{p95:.2e}")
    print(f"Vehicle threshold (99%): >{p99:.2e}")
    print("=" * 60)

    # Create visualization
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
    Create 2 essential statistical plots:
    1. Temporal Variance per Channel (WHERE are vehicles?)
    2. Total Energy Over Time (WHEN are vehicles?)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Statistical Analysis: Activity Detection', fontsize=14, fontweight='bold')

    # Plot 1: Temporal Variance per Channel
    ax = axes[0]
    channel_positions = df.columns.values
    ax.plot(channel_positions, stats['channel_vars'], linewidth=1.5, color='red')
    ax.set_xlabel('Position [m]', fontsize=11)
    ax.set_ylabel('Variance', fontsize=11)
    ax.set_title('Temporal Variance per Channel\n(WHERE: High variance = Active traffic zones)', fontsize=11)
    ax.grid(alpha=0.3)

    threshold = np.percentile(stats['channel_vars'], 75)
    ax.axhline(y=threshold, color='green', linestyle='--',
               alpha=0.7, linewidth=1.5, label='75th percentile')
    ax.legend()

    # Plot 2: Total Energy Over Time
    ax = axes[1]
    time_axis = np.arange(len(stats['time_energy'])) * dt
    ax.plot(time_axis, stats['time_energy'], linewidth=0.8, color='purple')
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Total Energy (sum of squared strain)', fontsize=11)
    ax.set_title('Total Energy Over Time\n(WHEN: Peaks = Vehicle events)', fontsize=11)
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


# ============================================================================
# FREQUENCY ANALYSIS
# ============================================================================

def compute_diagonal_energy(signal_2d):
    """
    Compute energy along diagonal directions using Radon transform.
    Moving objects create diagonal lines in space-time plots.
    """
    signal_norm = signal_2d - signal_2d.mean()
    signal_norm = signal_norm / (np.std(signal_norm) + 1e-10)
    
    theta = np.linspace(0, 180, 180, endpoint=False)
    sinogram = radon(np.abs(signal_norm), theta=theta, circle=False)
    
    # Focus on diagonal angles (30-60° and 120-150°)
    diagonal_angles_1 = (theta >= 30) & (theta <= 60)
    diagonal_angles_2 = (theta >= 120) & (theta <= 150)
    diagonal_mask = diagonal_angles_1 | diagonal_angles_2
    
    diagonal_energy = np.sum(sinogram[:, diagonal_mask] ** 2)
    total_energy = np.sum(sinogram ** 2)
    normalized_diagonal_energy = diagonal_energy / (total_energy + 1e-10)
    
    return normalized_diagonal_energy


def analyze_frequency_bands(df, fs=625, output_dir='../output'):
    """
    Analyze different frequency bands to understand signal characteristics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("FREQUENCY BAND ANALYSIS")
    print("="*70)
    
    freq_bands = [
        (2, 20, 'Very Low (2-20 Hz)'),
        (20, 40, 'Low (20-40 Hz)'),
        (40, 60, 'Mid-Low (40-60 Hz)'),
        (60, 90, 'Mid (60-90 Hz)'),
        (90, 120, 'Mid-High (90-120 Hz)'),
        (120, 200, 'High (120-200 Hz)')
    ]
    
    results = []
    
    # Compute FFT
    fft_data = np.fft.rfft(df.values, axis=0)
    freqs = np.fft.rfftfreq(df.shape[0], d=1/fs)
    magnitude = np.abs(fft_data)
    power = magnitude ** 2
    
    print("\nAnalyzing each frequency band...")
    print("-" * 70)
    
    for low, high, label in freq_bands:
        band_mask = (freqs >= low) & (freqs <= high)
        fft_band = fft_data.copy()
        fft_band[~band_mask, :] = 0
        
        band_signal = np.fft.irfft(fft_band, n=df.shape[0], axis=0)
        
        # Diagonal Energy (movement patterns)
        DOWNSAMPLE_FACTOR = 100 
        band_signal_downsampled = band_signal[::DOWNSAMPLE_FACTOR, :]
        diagonal_energy = compute_diagonal_energy(band_signal_downsampled)
        
        # Total Power (energy/noise)
        band_power = power[band_mask, :].sum()
        
        results.append({
            'band': (low, high),
            'label': label,
            'diagonal_energy': diagonal_energy,
            'total_power': band_power
        })
        
        print(f"{label:20s} | Diagonal: {diagonal_energy:.3e} | Power: {band_power:.3e}")
    
    # Create tradeoff plot
    create_tradeoff_plot(results, output_dir)
    
    print("\n" + "="*70)
    
    return results


def create_tradeoff_plot(results, output_dir):
    """
    Visualize tradeoff between diagonal structure (signal) and total power (noise).
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Normalize
    diagonal_vals = [r['diagonal_energy'] for r in results]
    power_vals = [r['total_power'] for r in results]
    
    diag_min, diag_max = min(diagonal_vals), max(diagonal_vals)
    power_min, power_max = min(power_vals), max(power_vals)
    
    x_norm = [(p - power_min) / (power_max - power_min + 1e-10) for p in power_vals]
    y_norm = [(d - diag_min) / (diag_max - diag_min + 1e-10) for d in diagonal_vals]
    
    # Plot
    for i, r in enumerate(results):
        ax.scatter(x_norm[i], y_norm[i], s=150, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.annotate(r['label'], (x_norm[i], y_norm[i]), 
                   textcoords="offset points", xytext=(5, -5), 
                   ha='left', fontsize=9)
    
    ax.set_xlabel('Total Power (Normalized)\n[Higher = More Energy/Noise]', fontsize=11)
    ax.set_ylabel('Diagonal Energy (Normalized)\n[Higher = Better Moving Object Structure]', fontsize=11)
    ax.set_title('Frequency Band Tradeoff Analysis\nGoal: High Structure, Low Power', 
                fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    ax.axvspan(0, 0.5, color='green', alpha=0.1, label='Low Power Zone')
    ax.axhspan(0.5, 1.0, color='orange', alpha=0.1, label='High Structure Zone')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    save_path = f"{output_dir}/frequency_band_tradeoff.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


def analyze_frequency_content(df, fs=625, output_dir='../output'):
    """
    Comprehensive frequency analysis using FFT:
    1. Space-Frequency Map (WHERE + WHAT frequencies)
    2. Spatial Variance of Frequencies (WHICH frequencies show movement)
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
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Frequency Content Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: Space-Frequency Map
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
    
    # Plot 2: Spatial Variance
    ax = axes[1]
    ax.plot(freqs, power_variance, linewidth=0.8, color='red')
    ax.set_xlabel('Frequency [Hz]', fontsize=11)
    ax.set_ylabel('Spatial Variance', fontsize=11)
    ax.set_title('Spatial Variance of Frequencies\n(WHICH: High variance = Moving objects, Low = Noise)', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 200])
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'frequency_content_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()
    
    # Print insights
    print("\nKey Insights:")
    print("-" * 70)
    peak_freq = freqs[1:][np.argmax(avg_power[1:])]
    print(f"  • Peak power at {peak_freq:.1f} Hz")
    
    max_variance_freq = freqs[1:][np.argmax(power_variance[1:])]
    print(f"  • Maximum spatial variance at {max_variance_freq:.1f} Hz (likely vehicles)")
    
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
    
    test_bands = [
        (2, 20, 'Very Low (2-20 Hz)'),
        (20, 40, 'Low (20-40 Hz)'),
        (40, 60, 'Mid-Low (40-60 Hz)'),
        (60, 90, 'Mid (60-90 Hz)'),
        (90, 120, 'Mid-High (90-120 Hz)'),
        (120, 200, 'High (120-200 Hz)')
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
        
        vmin, vmax = np.percentile(data, [5, 95])
        im = ax.imshow(data, aspect='auto', cmap='viridis',
                      vmin=vmin, vmax=vmax, interpolation='none')
        ax.set_title(f"{label}\n{low}-{high} Hz", fontsize=10, fontweight='bold')
        ax.set_ylabel('Time')
        ax.set_xlabel('Space (Channel)')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    save_path = f"{output_dir}/filtered_comparison.png"
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def frequency_filter_fft(df, dt, lowcut, highcut):
    """
    Simple FFT bandpass filter.
    """
    n_samples = df.shape[0]
    n_channels = df.shape[1]
    filtered_data = np.zeros_like(df.values)
    
    for i in range(n_channels):
        signal_data = df.iloc[:, i].values
        fft = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(n_samples, dt)
        
        mask = (np.abs(freqs) >= lowcut) & (np.abs(freqs) <= highcut)
        fft_filtered = fft * mask
        
        filtered_signal = np.fft.ifft(fft_filtered)
        filtered_data[:, i] = np.real(filtered_signal)
    
    return pd.DataFrame(data=filtered_data, index=df.index, columns=df.columns)