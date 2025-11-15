"""
Data Analysis Module
Statistical and frequency analysis of DAS data
Calls visualization functions directly
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import find_peaks
from visualizations import (plot_raw_waterfall, plot_statistical_analysis,
                            plot_frequency_analysis)


def analyze_and_visualize_segment(df, dt=0.0016, dx=5.106500953873407,
                                   segment_name="segment", output_dir=None,
                                   verbose=True):

    data_flat = df.values.flatten()

    # Calculate key metrics
    mean = np.mean(data_flat)
    std = np.std(data_flat)
    min_val = np.min(data_flat)
    max_val = np.max(data_flat)
    median = np.median(data_flat)

    # Percentiles
    p1 = np.percentile(data_flat, 1)
    p5 = np.percentile(data_flat, 5)
    p25 = np.percentile(data_flat, 25)
    p75 = np.percentile(data_flat, 75)
    p95 = np.percentile(data_flat, 95)
    p99 = np.percentile(data_flat, 99)

    # Per-channel metrics
    channel_means = df.mean(axis=0).values
    channel_stds = df.std(axis=0).values
    channel_vars = df.var(axis=0).values
    time_energy = (df ** 2).sum(axis=1).values

    if verbose:
        print("\n" + "="*60)
        print("BASIC STATISTICS")
        print("="*60)
        print(f"Mean:   {mean:.6e}")
        print(f"Std:    {std:.6e}")
        print("Value is positive when it is stretched (tension is put on a cable), negative when compressed.")
        print(f"Range:  [{min_val:.2e}, {max_val:.2e}]")
        print(f"Median: {median:.6e}")
        print(f"\nNoise floor (95%): ±{p95:.2e}")
        print(f"Vehicle threshold (99%): >{p99:.2e}")
        print("="*60)

    # Create statistical plot
    stats_dict = {
        'mean': mean, 'std': std, 'min': min_val, 'max': max_val,
        'median': median, 'percentile_1': p1, 'percentile_5': p5,
        'percentile_25': p25, 'percentile_75': p75,
        'percentile_95': p95, 'percentile_99': p99,
        'channel_means': channel_means, 'channel_stds': channel_stds,
        'channel_vars': channel_vars, 'time_energy': time_energy
    }

    save_path = f"{output_dir}/{segment_name}_statistics.png" if output_dir else None
    plot_statistical_analysis(df, stats_dict, dx=dx, dt=dt,
                             title=f"Statistical Analysis: {segment_name}",
                             save_path=save_path)

    # ==========================================
    # 2. FREQUENCY ANALYSIS
    # ==========================================

    if verbose:
        print("\nPerforming frequency analysis...")

    sampling_rate = 1 / dt
    nyquist_freq = sampling_rate / 2

    # Select high-variance channels (top 10)
    top_indices = np.argsort(channel_vars)[-10:]

    # Compute average spectrum across sampled channels
    all_channel_ffts = []
    for channel in df.columns[::10]:
        signal = df[channel].values
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), dt)
        pos_mask = freqs >= 0
        fft_mag = np.abs(fft[pos_mask])
        all_channel_ffts.append(fft_mag)

    avg_spectrum = np.mean(all_channel_ffts, axis=0)
    std_spectrum = np.std(all_channel_ffts, axis=0)
    freqs_pos = freqs[pos_mask]

    # Find dominant frequencies
    freq_limit = 150
    freq_mask = freqs_pos < freq_limit
    peaks, _ = find_peaks(
        avg_spectrum[freq_mask],
        height=np.percentile(avg_spectrum[freq_mask], 90),
        distance=10
    )

    dominant_freqs = freqs_pos[freq_mask][peaks]
    dominant_mags = avg_spectrum[freq_mask][peaks]

    # Compute FFT for selected high-variance channels
    fft_results = []
    for i in top_indices[:5]:
        channel = df.columns[i]
        signal = df[channel].values
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), dt)
        pos_mask = freqs >= 0
        fft_results.append({
            'channel': channel,
            'channel_index': int(channel / dx),
            'freqs': freqs[pos_mask],
            'magnitude': np.abs(fft[pos_mask])
        })

    if verbose:
        print("\n" + "="*60)
        print("FREQUENCY ANALYSIS")
        print("="*60)
        print(f"Sampling rate: {sampling_rate:.1f} Hz")
        print(f"Nyquist frequency: {nyquist_freq:.1f} Hz")
        print(f"\nDominant frequencies (< {freq_limit} Hz):")
        for i, (freq, mag) in enumerate(zip(dominant_freqs[:5], dominant_mags[:5])):
            print(f"  {i+1}. {freq:.2f} Hz (magnitude: {mag:.2e})")
        print("="*60)

    # Create frequency plot
    freq_dict = {
        'channel_ffts': fft_results,
        'avg_spectrum': avg_spectrum,
        'std_spectrum': std_spectrum,
        'freqs': freqs_pos,
        'dominant_frequencies': dominant_freqs,
        'dominant_magnitudes': dominant_mags,
        'peaks_indices': peaks
    }

    save_path = f"{output_dir}/{segment_name}_frequency.png" if output_dir else None
    plot_frequency_analysis(freq_dict, dt=dt,
                           title=f"Frequency Analysis: {segment_name}",
                           save_path=save_path)

    # ==========================================
    # 4. SNR AND ACTIVE REGIONS
    # ==========================================

    # Estimate noise
    sorted_vars = np.sort(channel_vars)
    noise_estimate = np.mean(sorted_vars[:len(sorted_vars)//4])

    # Calculate SNR - high snr - high variance - clear signals, a lot of vehicles
    snr = channel_vars / (noise_estimate + 1e-10)
    snr_db = 10 * np.log10(snr)

    # Identify active regions (high variance)
    threshold = np.percentile(channel_vars, 75)
    active_indices = np.where(channel_vars > threshold)[0]
    active_positions = active_indices * dx

    if verbose:
        print("\n" + "="*60)
        print("ACTIVE REGIONS")
        print("="*60)
        print(f"Active channels: {len(active_indices)} - areas of the street")
        if len(active_positions) > 0:
            print(f"Spatial range: {active_positions[0]:.1f} - {active_positions[-1]:.1f} m")
        print("="*60)

    # ==========================================
    # 5. WATERFALL AND COMPARISON PLOTS
    # ==========================================

    # Waterfall plot
    save_path = f"{output_dir}/{segment_name}_waterfall.png" if output_dir else None
    plot_raw_waterfall(df, title=f"Raw DAS Data: {segment_name}",
                      save_path=save_path)
