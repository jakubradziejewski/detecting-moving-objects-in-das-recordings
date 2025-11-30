import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
from scipy.signal import savgol_filter
import os


def compute_velocity_field(df_raw, dx, dt):
    """Compute velocity using Structure Tensor method."""
    data = np.abs(df_raw.values.astype(float))
    
    # Smooth and compute gradients
    smooth = gaussian_filter(data, sigma=[15, 2])
    grad_t = sobel(smooth, axis=0)
    grad_x = sobel(smooth, axis=1)
    
    # Tensor components
    Ixx = gaussian_filter(grad_x * grad_x, sigma=[20, 3])
    Ixt = gaussian_filter(grad_x * grad_t, sigma=[20, 3])
    Iyy = gaussian_filter(grad_t * grad_t, sigma=[20, 3])
    
    # Velocity calculation
    epsilon = np.percentile(Ixx, 50) * 0.5
    v_pixels = -Ixt / (Ixx + epsilon)
    v_kmh = v_pixels * (dx / dt) * 3.6
    
    # Coherence (quality metric)
    coherence = Ixx / (Ixx + Iyy + 1e-6)
    
    return v_kmh, coherence


def sample_velocity_along_line(v_field, coherence, binary_mask, raw_shape, line_params, 
                                processed_shape=(750, 750), margin=20):
    """Sample velocities along a detected line using binary mask."""
    raw_h, raw_w = raw_shape
    proc_h, proc_w = processed_shape
    angle, dist, nominal_vel, x0, y0 = line_params
    
    scale_y = raw_h / proc_h
    scale_x = raw_w / proc_w
    
    sin_a, cos_a = np.sin(angle), np.cos(angle)
    if abs(cos_a) < 1e-3:
        return None, None
    
    # Sample points along line
    t_indices = np.arange(0, raw_h, 4)
    t_proc = t_indices / scale_y
    x_proc = (dist - t_proc * sin_a) / cos_a
    x_indices = (x_proc * scale_x).astype(int)
    
    # Filter valid coordinates
    valid = (x_indices >= 0) & (x_indices < raw_w)
    t_indices, x_indices = t_indices[valid], x_indices[valid]
    
    times, velocities = [], []
    window = int(margin * scale_x)
    
    for t, center_x in zip(t_indices, x_indices):
        # Define sampling window
        x_start = max(0, center_x - window)
        x_end = min(raw_w, center_x + window)
        
        # Check binary mask
        y_proc = int(t / scale_y)
        if y_proc >= proc_h:
            continue
        x_start_proc = max(0, int(x_start / scale_x))
        x_end_proc = min(proc_w, int(x_end / scale_x))
        
        if np.sum(binary_mask[y_proc, x_start_proc:x_end_proc]) == 0:
            continue  # Skip background
        
        # Extract velocities and coherence
        v_vals = v_field[t, x_start:x_end]
        c_vals = coherence[t, x_start:x_end]
        
        # Filter by velocity range (±30% of nominal)
        if nominal_vel > 0:
            mask = (v_vals > nominal_vel * 0.7) & (v_vals < nominal_vel * 1.3)
        else:
            mask = (v_vals > nominal_vel * 1.3) & (v_vals < nominal_vel * 0.7)
        
        # Filter by coherence
        coh_thresh = np.percentile(c_vals, 70)
        mask &= (c_vals > coh_thresh)
        
        v_valid, c_valid = v_vals[mask], c_vals[mask]
        
        if len(v_valid) > 0:
            velocities.append(np.average(v_valid, weights=c_valid))
            times.append(t)
    
    return np.array(times), np.array(velocities)


def analyze_velocity_over_time(df_raw, binary_image, lines, dx, dt, output_dir="."):
    """Analyze velocity variations over time for detected lines."""
    print(f"\n{'='*70}\nVELOCITY ANALYSIS\n{'='*70}")
    
    # Compute velocity field
    v_field, coherence = compute_velocity_field(df_raw, dx, dt)
    
    if len(lines) == 0:
        print("No lines to analyze.")
        return []
    
    # Setup plotting
    n_cols = min(3, len(lines))
    n_rows = (len(lines) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    axes = np.atleast_1d(axes).flatten()
    
    velocity_profiles = []
    
    for idx, line in enumerate(lines):
        angle, dist, nominal_kmh, x0, y0 = line
        
        # Sample velocities along line
        times_idx, vels = sample_velocity_along_line(
            v_field, coherence, binary_image, df_raw.shape, line
        )
        
        if times_idx is None or len(vels) < 10:
            axes[idx].text(0.5, 0.5, "Insufficient Signal", ha='center', va='center')
            axes[idx].axis('off')
            continue
        
        times_sec = times_idx * dt - times_idx[0] * dt
        
        # Calculate start time in the original data
        start_time_str = df_raw.index[int(times_idx[0])].strftime('%H:%M:%S')
        
        # Apply smoothing
        vels_smooth = gaussian_filter(vels, sigma=1) if len(vels) > 10 else vels
        
        if len(vels) > 15:
            win_len = min(len(vels) // 5 * 2 + 1, len(vels) - 1)
            win_len = max(win_len, 5)
            vels_trend = savgol_filter(vels, window_length=win_len, polyorder=3)
        else:
            vels_trend = vels
        
        mean_vel = np.mean(vels_trend)
        std_vel = np.std(vels_trend)
        
        velocity_profiles.append({
            'line_index': idx,
            'time': times_sec,
            'velocity_smooth': vels_smooth,
            'velocity_trend': vels_trend,
            'mean': mean_vel,
            'std': std_vel
        })
        
        # Plot
        ax = axes[idx]
        ax.plot(times_sec, vels_smooth, 'b-', lw=1, alpha=0.4, label='Smoothed')
        ax.plot(times_sec, vels_trend, 'r-', lw=2, label='Trend')
        ax.axhline(nominal_kmh, c='g', ls='--', alpha=0.8, label=f'Nominal: {nominal_kmh:.1f}')
        ax.fill_between(times_sec, mean_vel - std_vel, mean_vel + std_vel, 
                        color='red', alpha=0.1)
        
        # Add start time to help distinguish lines
        ax.set_title(f"Line {idx+1} (Start: {start_time_str}) | Nom: {nominal_kmh:.1f} km/h\n"
                    f"Measured: {mean_vel:.1f} ± {std_vel:.1f} km/h", 
                    fontweight='bold', fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (km/h)")
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)
        
        y_range = max(20, std_vel * 4)
        ax.set_ylim(mean_vel - y_range, mean_vel + y_range)
        
        print(f"Line {idx+1}: {mean_vel:.1f} ± {std_vel:.1f} km/h")
    
    # Hide unused subplots
    for i in range(len(lines), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "velocity_profiles_over_time.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Saved: {save_path}\n")
    return velocity_profiles