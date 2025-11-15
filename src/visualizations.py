import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def plot_raw_waterfall(df, title="Raw DAS Data", figsize=(16, 10), save_path=None):

    fig, ax = plt.subplots(figsize=figsize)

    # Center data and normalize
    data_centered = df.values - df.values.mean()
    low, high = np.percentile(np.abs(data_centered), [1, 99.5])
    norm = Normalize(vmin=-high, vmax=high, clip=True)

    # Plot
    im = ax.imshow(data_centered,
                   interpolation='none',
                   aspect='auto',
                   cmap='seismic',
                   norm=norm)

    ax.set_ylabel('Time Sample', fontsize=12)
    ax.set_xlabel('Channel', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Time labels
    n_time_labels = 7
    time_step = len(df) // (n_time_labels - 1)
    time_positions = np.arange(0, len(df), time_step)
    time_labels = [df.index[min(i, len(df)-1)].strftime('%H:%M:%S')
                   for i in time_positions]
    ax.set_yticks(time_positions)
    ax.set_yticklabels(time_labels)

    # Spatial labels
    n_space_labels = 7
    space_step = len(df.columns) // (n_space_labels - 1)
    space_positions = np.arange(0, len(df.columns), space_step)
    space_labels = [f'{df.columns[min(i, len(df.columns)-1)]:.0f}'
                    for i in space_positions]
    ax.set_xticks(space_positions)
    ax.set_xticklabels(space_labels)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Strain Rate (centered)', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_statistical_analysis(df, stats, dx=5.106500953873407, dt=0.0016,
                              title="Statistical Analysis", figsize=(18, 10),
                              save_path=None):
    """
    Create comprehensive statistical analysis plots

    Parameters:
    -----------
    df : pd.DataFrame
        Data as DataFrame
    stats : dict
        Statistics dictionary from analysis
    dx : float
        Spatial resolution
    dt : float
        Temporal resolution
    title : str
        Figure title
    figsize : tuple
        Figure size
    save_path : str or None
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 1. Overall amplitude distribution (histogram)
    ax = axes[0, 0]
    data_flat = df.values.flatten()
    counts, bins, patches = ax.hist(data_flat, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('Strain Rate', fontsize=11)
    ax.set_ylabel('Frequency (count)', fontsize=11)
    ax.set_title('Overall Amplitude Distribution', fontsize=12)
    ax.set_yscale('log')
    ax.grid(alpha=0.3)

    # Add statistics text
    stats_text = f'Mean: {stats["mean"]:.2e}\n'
    stats_text += f'Std: {stats["std"]:.2e}\n'
    stats_text += f'Min: {stats["min"]:.2e}\n'
    stats_text += f'Max: {stats["max"]:.2e}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Spatial variation (mean ± std per channel)
    ax = axes[0, 1]
    channel_positions = df.columns.values
    ax.plot(channel_positions, stats['channel_means'],
            label='Mean', linewidth=1.5, color='blue')
    ax.fill_between(channel_positions,
                    stats['channel_means'] - stats['channel_stds'],
                    stats['channel_means'] + stats['channel_stds'],
                    alpha=0.3, label='±1 Std', color='blue')
    ax.set_xlabel('Position [m]', fontsize=11)
    ax.set_ylabel('Mean Strain Rate', fontsize=11)
    ax.set_title('Spatial Variation (Mean per Channel)', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)

    # 3. Temporal variance per channel
    ax = axes[1, 0]
    ax.plot(channel_positions, stats['channel_vars'],
            linewidth=1.5, color='red')
    ax.set_xlabel('Position [m]', fontsize=11)
    ax.set_ylabel('Variance', fontsize=11)
    ax.set_title('Temporal Variance per Channel', fontsize=12)
    ax.grid(alpha=0.3)

    # Highlight active regions
    threshold = np.percentile(stats['channel_vars'], 75)
    ax.axhline(y=threshold, color='green', linestyle='--',
               alpha=0.7, linewidth=1, label='75th percentile')
    ax.legend()

    # 4. Total energy over time
    ax = axes[1, 1]
    time_axis = np.arange(len(stats['time_energy'])) * dt
    ax.plot(time_axis, stats['time_energy'], linewidth=0.5, color='purple')
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Total Energy (sum of squared strain)', fontsize=11)
    ax.set_title('Total Energy Over Time', fontsize=12)
    ax.grid(alpha=0.3)

    # Add event markers
    energy_threshold = np.percentile(stats['time_energy'], 99)
    events = stats['time_energy'] > energy_threshold
    event_times = time_axis[events]
    event_energies = stats['time_energy'][events]
    ax.scatter(event_times, event_energies, color='red', s=10,
               alpha=0.5, label='Top 1% events')
    ax.axhline(y=energy_threshold, color='red', linestyle='--',
               alpha=0.5, linewidth=1)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_frequency_analysis(freq_results, dt=0.0016, title="Frequency Analysis",
                            figsize=(16, 10), save_path=None):

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    sampling_rate = 1 / dt
    freqs = freq_results['freqs']
    avg_spectrum = freq_results['avg_spectrum']
    std_spectrum = freq_results['std_spectrum']

    # 1. Individual channel FFTs
    ax = axes[0, 0]
    for i, fft_data in enumerate(freq_results['channel_ffts'][:5]):
        ax.plot(fft_data['freqs'], fft_data['magnitude'],
                alpha=0.7, linewidth=1,
                label=f"Ch {fft_data['channel_index']}")

    ax.set_xlim([0, 150])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Magnitude')
    ax.set_title('FFT: Individual Channels')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_yscale('log')

    # 2. Average frequency spectrum
    ax = axes[0, 1]
    ax.plot(freqs, avg_spectrum, linewidth=2, color='blue', label='Mean')
    ax.fill_between(freqs,
                     avg_spectrum - std_spectrum,
                     avg_spectrum + std_spectrum,
                     alpha=0.3, color='blue', label='±1 Std')
    ax.set_xlim([0, 150])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Magnitude')
    ax.set_title('Average Frequency Spectrum')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')

    # 3. Dominant frequencies
    ax = axes[1, 0]
    freq_limit = 150
    freq_mask = freqs < freq_limit

    ax.plot(freqs[freq_mask], avg_spectrum[freq_mask], linewidth=2, color='green')

    # Plot dominant frequencies
    dom_freqs = freq_results['dominant_frequencies']
    dom_mags = freq_results['dominant_magnitudes']

    ax.plot(dom_freqs, dom_mags, 'rx', markersize=10, label='Dominant Frequencies')

    # Annotate top 5 peaks
    for i, (freq, mag) in enumerate(zip(dom_freqs[:5], dom_mags[:5])):
        ax.annotate(f'{freq:.1f} Hz',
                   xy=(freq, mag),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=8, bbox=dict(boxstyle='round', alpha=0.7))

    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Magnitude')
    ax.set_title('Dominant Frequency Components')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')

    # 4. Frequency content summary
    ax = axes[1, 1]
    # Show frequency bands
    bands = {
        'Low (0-5 Hz)': (0, 5),
        'Vehicle (5-50 Hz)': (5, 50),
        'Mid (50-100 Hz)': (50, 100),
        'High (100-150 Hz)': (100, 150)
    }

    band_energies = []
    band_labels = []
    for label, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        energy = np.sum(avg_spectrum[mask])
        band_energies.append(energy)
        band_labels.append(label)

    colors = ['red', 'green', 'blue', 'purple']
    ax.bar(range(len(band_labels)), band_energies, tick_label=band_labels,
           color=colors, alpha=0.7)
    ax.set_ylabel('Total Energy')
    ax.set_title('Energy Distribution by Frequency Band')
    ax.grid(alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
