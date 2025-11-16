import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd


def set_axis(x, no_labels=7):
    nx = x.shape[0]
    if nx == 0:
        return np.array([]), np.array([])

    step_x = max(1, int(nx / (no_labels - 1)))
    x_positions = np.arange(0, nx, step_x)

    if isinstance(x, pd.Series):
        x_labels = x.iloc[::step_x]
    else:
        x_labels = x[::step_x]

    return x_positions, x_labels


def plot_raw_waterfall(df, title="Raw DAS Data", figsize=(16, 10),
                       cmap='gray_r', save_path=None):
    fig, ax = plt.subplots(figsize=figsize)

    data_to_plot = df.values

    data_abs = np.abs(data_to_plot - np.mean(data_to_plot))

    low, high = np.percentile(data_abs, [5, 99.5])
    norm = Normalize(vmin=low, vmax=high, clip=True)

    im = ax.imshow(data_abs,
                   interpolation='none',
                   aspect='auto',
                   cmap=cmap,
                   norm=norm)

    ax.set_ylabel('Time', fontsize=12)
    ax.set_xlabel('Space [m]', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    x_positions, x_labels = set_axis(df.columns)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(np.round(x_labels).astype(int))

    time_labels = df.index.strftime('%H:%M:%S')
    y_positions, y_labels = set_axis(time_labels)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Absolute Strain Rate (Normalized)', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_statistical_analysis(df, stats, dx=5.106500953873407, dt=0.0016,
                              title="Statistical Analysis", figsize=(18, 10),
                              save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    ax = axes[0, 0]
    data_flat = df.values.flatten()
    counts, bins, patches = ax.hist(data_flat, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('Strain Rate', fontsize=11)
    ax.set_ylabel('Frequency (count)', fontsize=11)
    ax.set_title('Overall Amplitude Distribution', fontsize=12)
    ax.set_yscale('log')
    ax.grid(alpha=0.3)

    stats_text = f'Mean: {stats["mean"]:.2e}\n'
    stats_text += f'Std: {stats["std"]:.2e}\n'
    stats_text += f'Min: {stats["min"]:.2e}\n'
    stats_text += f'Max: {stats["max"]:.2e}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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

    ax = axes[1, 0]
    ax.plot(channel_positions, stats['channel_vars'],
            linewidth=1.5, color='red')
    ax.set_xlabel('Position [m]', fontsize=11)
    ax.set_ylabel('Variance', fontsize=11)
    ax.set_title('Temporal Variance per Channel', fontsize=12)
    ax.grid(alpha=0.3)

    threshold = np.percentile(stats['channel_vars'], 75)
    ax.axhline(y=threshold, color='green', linestyle='--',
               alpha=0.7, linewidth=1, label='75th percentile')
    ax.legend()

    ax = axes[1, 1]
    time_axis = np.arange(len(stats['time_energy'])) * dt
    ax.plot(time_axis, stats['time_energy'], linewidth=0.5, color='purple')
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Total Energy (sum of squared strain)', fontsize=11)
    ax.set_title('Total Energy Over Time', fontsize=12)
    ax.grid(alpha=0.3)

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

