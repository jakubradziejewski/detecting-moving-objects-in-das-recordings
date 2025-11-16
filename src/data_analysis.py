import numpy as np
from visualizations import plot_statistical_analysis


def find_peaks_numpy(data, height, distance):
    """
    Minimal numpy-only implementation of find_peaks with height and distance.
    """
    # 1. Find all indices that are local maxima
    # (data[i] > data[i-1]) and (data[i] > data[i+1])
    # We add 1 to the index to account for the diff shift
    peaks = np.where(
        (data[1:-1] > data[0:-2]) & (data[1:-1] > data[2:])
    )[0] + 1

    # 2. Filter by height
    peaks = peaks[data[peaks] >= height]

    # 3. Filter by distance
    if distance > 1 and len(peaks) > 0:
        # Sort peaks by magnitude (highest first)
        peak_mags = data[peaks]
        sort_indices = np.argsort(-peak_mags)
        peaks_sorted = peaks[sort_indices]

        # Keep track of peaks to keep
        keep = np.ones(len(data), dtype=bool)
        final_peaks = []

        for i in peaks_sorted:
            if keep[i]:
                # This is a valid peak, keep it
                final_peaks.append(i)

                # Suppress all other peaks within 'distance'
                start = max(0, i - distance)
                end = min(len(data), i + distance + 1)
                keep[start:end] = False

        # Return peaks sorted by their original index
        final_peaks.sort()
        peaks = np.array(final_peaks)

    return peaks


def analyze_and_visualize_segment(df, dt=0.0016, dx=5.106500953873407,
                                  segment_name="segment", output_dir=None):
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

    print("\n" + "=" * 60)
    print("BASIC STATISTICS")
    print("=" * 60)
    print(f"Mean:   {mean:.6e}")
    print(f"Std:    {std:.6e}")
    print("Value is positive when it is stretched (tension is put on a cable), negative when compressed.")
    print(f"Range:  [{min_val:.2e}, {max_val:.2e}]")
    print(f"Median: {median:.6e}")
    print(f"\nNoise floor (95%): ±{p95:.2e}")
    print(f"Vehicle threshold (99%): >{p99:.2e}")
    print("=" * 60)

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
    print("Frequency analysis")
    sampling_rate = 1 / dt
    nyquist_freq = sampling_rate / 2

    # Compute average spectrum across sampled channels
    all_channel_ffts = []
    all_freqs = None

    for i, channel in enumerate(df.columns):
        signal = df[channel].values

        # FFT for this channel
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), dt)

        # Only positive frequencies
        pos_mask = freqs >= 0
        freqs_pos = freqs[pos_mask]
        fft_mag = np.abs(fft[pos_mask])

        all_channel_ffts.append(fft_mag)

        if all_freqs is None:
            all_freqs = freqs_pos

    # Convert to array for easier manipulation
    all_channel_ffts = np.array(all_channel_ffts)

    # Compute statistics across channels
    avg_spectrum = np.mean(all_channel_ffts, axis=0)

    # Find dominant frequencies in average spectrum
    freq_limit = 150
    freq_mask = all_freqs < freq_limit

    # --- THIS IS THE REPLACEMENT ---
    data_to_search = avg_spectrum[freq_mask]
    height_threshold = np.percentile(data_to_search, 90)

    peaks = find_peaks_numpy(
        data_to_search,
        height=height_threshold,
        distance=10
    )
    # --- END REPLACEMENT ---

    dominant_freqs = all_freqs[freq_mask][peaks]
    dominant_mags = avg_spectrum[freq_mask][peaks]

    # Select a few representative channels for visualization (high variance)
    top_indices = np.argsort(channel_vars)[-5:]  # Top 5 for plotting

    fft_results = []
    for i in top_indices:
        channel = df.columns[i]
        fft_results.append({
            'channel': channel,
            'channel_index': int(channel / dx),
            'freqs': all_freqs,
            'magnitude': all_channel_ffts[i]
        })

    print("\n" + "=" * 60)
    print("FREQUENCY ANALYSIS")
    print("=" * 60)
    print(f"FFT computed for ALL {len(df.columns)} channels independently")
    print(f"Sampling rate: {sampling_rate:.1f} Hz")
    print(f"Nyquist frequency: {nyquist_freq:.1f} Hz")
    print(f"Frequency resolution: {all_freqs[1]:.4f} Hz")
    print(f"\nDominant frequencies (< {freq_limit} Hz):")
    for i, (freq, mag) in enumerate(zip(dominant_freqs[:5], dominant_mags[:5])):
        print(f"  {i + 1}. {freq:.2f} Hz (magnitude: {mag:.2e})")
    print("=" * 60)

