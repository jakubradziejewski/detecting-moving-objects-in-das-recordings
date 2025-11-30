import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
import os

def compute_velocity_structure_tensor(df_raw, dx, dt, sigma_t=15, sigma_x=2, window_t=20, window_x=2):
    """
    Computes velocity using the Structure Tensor (Local Least Squares) method.
    This is much more robust than the simple gradient ratio.
    
    Formula: v = - <Ix * It> / <Ix^2>
    (The brackets <> denote smoothing/averaging over a local window)
    """
    # 1. Prepare Envelope (Intensity)
    data = df_raw.values.astype(float)
    data = data - np.median(data, axis=0) # Remove channel DC
    envelope = np.abs(data)
    
    # 2. Smooth Envelope (Pre-smoothing)
    # Reduces high-freq noise before derivative calculation
    env_smooth = gaussian_filter(envelope, sigma=[sigma_t, sigma_x])
    
    # 3. Compute Gradients
    It = sobel(env_smooth, axis=0) # Time derivative
    Ix = sobel(env_smooth, axis=1) # Space derivative
    
    # 4. Compute Tensor Components
    # We need the products of derivatives
    Ixx = Ix * Ix
    Ixt = Ix * It
    
    # 5. Integrate over Local Window (Post-smoothing)
    # This sums the components over the neighborhood (the "Structure Tensor")
    # This is effectively the "Sum" in the least squares formula
    # We use a window matching the feature scale
    Sxx = gaussian_filter(Ixx, sigma=[window_t, window_x])
    Sxt = gaussian_filter(Ixt, sigma=[window_t, window_x])
    
    # 6. Calculate Velocity
    # v_pixel = - Sxt / (Sxx + epsilon)
    # Epsilon prevents division by zero in empty regions
    epsilon = np.percentile(Sxx, 50) * 0.1 # Dynamic epsilon
    
    v_pix = - Sxt / (Sxx + epsilon)
    
    # 7. Convert to Physical Units
    # v_phys = v_pix * (dx / dt)
    factor = dx / dt
    v_phys = v_pix * factor
    v_kmh = v_phys * 3.6
    
    # 8. Compute Coherence/Confidence
    # We trust regions where Sxx (spatial contrast) is high
    coherence = Sxx / (np.max(Sxx) + 1e-6)
    
    return v_kmh, coherence

def sample_velocity_along_line(velocity_field, coherence, raw_shape, line_params, 
                               processed_shape=(750, 750), width_margin=20):
    """Samples velocities using coherence weighting."""
    raw_h, raw_w = raw_shape
    proc_h, proc_w = processed_shape
    angle, dist, nominal_vel, x0, y0 = line_params
    
    scale_y = raw_h / proc_h 
    scale_x = raw_w / proc_w
    
    sin_t = np.sin(angle)
    cos_t = np.cos(angle)
    
    if abs(cos_t) < 1e-3: return None, None
    
    # Generate path
    t_indices = np.arange(0, raw_h, 4) # Step 4 for speed
    t_proc = t_indices / scale_y
    x_proc = (dist - t_proc * sin_t) / cos_t
    x_indices = (x_proc * scale_x).astype(int)
    
    valid = (x_indices >= 0) & (x_indices < raw_w)
    t_indices = t_indices[valid]
    x_indices = x_indices[valid]
    
    times = []
    velocities = []
    
    # Window width in raw pixels
    window = int(width_margin * scale_x)
    
    for t, center_x in zip(t_indices, x_indices):
        x_start = max(0, center_x - window)
        x_end = min(raw_w, center_x + window)
        
        # Get data
        v_vals = velocity_field[t, x_start:x_end]
        c_vals = coherence[t, x_start:x_end]
        
        # Filter based on nominal direction and physical limits
        # We assume the object won't go 3x its nominal speed or reverse
        min_v = 10
        max_v = max(150, abs(nominal_vel) * 2.5)
        
        if nominal_vel > 0:
            mask = (v_vals > min_v) & (v_vals < max_v)
        else:
            mask = (v_vals < -min_v) & (v_vals > -max_v)
            
        # Additional Coherence Threshold
        mask = mask & (c_vals > np.percentile(c_vals, 50))
        
        v_valid = v_vals[mask]
        c_valid = c_vals[mask]
        
        if len(v_valid) > 0:
            # Weighted Mean based on Coherence (trust strong edges more)
            weighted_v = np.average(v_valid, weights=c_valid)
            velocities.append(weighted_v)
            times.append(t)
            
    return np.array(times), np.array(velocities)

def analyze_velocity_over_time(df_raw, lines, dx, dt, method='gradient', output_dir="."):
    print("\n" + "=" * 70)
    print(f"VELOCITY ANALYSIS: STRUCTURE TENSOR METHOD")
    print("=" * 70)
    
    # 1. Compute Tensor Field
    print("Computing Structure Tensor velocity field...")
    # sigma_t is high because dt is tiny (0.0016s) vs dx (5m)
    v_field, coherence = compute_velocity_structure_tensor(
        df_raw, dx, dt, 
        sigma_t=15, sigma_x=2,  # Pre-smoothing
        window_t=40, window_x=4 # Tensor Integration Window
    )
    
    velocity_profiles = []
    
    # Setup Plots
    n_lines = len(lines)
    n_cols = min(3, n_lines)
    n_rows = (n_lines + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_lines == 1: axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, line in enumerate(lines):
        nominal_kmh = line[2]
        print(f"  Line {idx+1}: Nominal {nominal_kmh:.1f} km/h")
        
        times_idx, vels = sample_velocity_along_line(
            v_field, coherence, df_raw.shape, line
        )
        
        if times_idx is None or len(vels) < 10:
            print("    -> Track failed (insufficient signal).")
            continue
            
        times_sec = times_idx * dt
        times_sec = times_sec - times_sec[0]
        
        # Smooth result
        if len(vels) > 10:
            vels_smooth = gaussian_filter(vels, sigma=3)
        else:
            vels_smooth = vels
            
        mean_vel = np.mean(vels_smooth)
        std_vel = np.std(vels_smooth)
        
        print(f"    -> Tracked: {mean_vel:.1f} ± {std_vel:.1f} km/h")
        
        velocity_profiles.append({
            'line_index': idx,
            'time': times_sec,
            'velocity': vels_smooth,
            'mean': mean_vel
        })
        
        # Plot
        ax = axes[idx]
        ax.plot(times_sec, vels_smooth, 'b-', lw=2, label='Tensor Velocity')
        ax.axhline(nominal_kmh, c='r', ls='--', label=f'Nominal: {nominal_kmh:.0f}')
        ax.fill_between(times_sec, mean_vel-std_vel, mean_vel+std_vel, color='blue', alpha=0.1)
        
        ax.set_title(f"Line {idx+1}: {mean_vel:.0f} km/h")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (km/h)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(120, nominal_kmh * 2))

    # Cleanup
    for i in range(len(velocity_profiles), len(axes)): axes[i].axis('off')
    plt.tight_layout()
    save_path = os.path.join(output_dir, "velocity_structure_tensor.png")
    plt.savefig(save_path)
    plt.close()
    print(f"\nSaved plot: {save_path}")
    
    return velocity_profiles

def plot_signal_and_velocity(velocity_profiles, output_dir):
    pass