import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks
from preprocessing import set_axis, visualize_das
import os


def detect_lines_hough(binary_image, threshold_ratio=0.85):
    """
    Detect lines using Hough transform on binary image.
    
    Parameters:
    - binary_image: binary numpy array
    - threshold_ratio: threshold for Hough peak detection (0-1)
    
    Returns:
    - lines: list of (angle, dist) tuples
    """
    print(f"\n  Running Hough transform (threshold ratio: {threshold_ratio})...")
    
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(binary_image, theta=tested_angles)
    
    _, angles, dists = hough_line_peaks(h, theta, d, threshold=np.max(h) * threshold_ratio)
    
    lines = list(zip(angles, dists))
    print(f"  Detected {len(lines)} lines")
    
    return lines


def calculate_velocities(lines, dx_effective, dt_effective, velocity_min=30, velocity_max=115):
    """
    Calculate velocities for detected lines and filter by realistic velocity range.
    
    Parameters:
    - lines: list of (angle, dist) tuples
    - dx_effective: effective spatial resolution (m/pixel)
    - dt_effective: effective temporal resolution (s/pixel)
    - velocity_min: minimum realistic velocity (km/h)
    - velocity_max: maximum realistic velocity (km/h)
    
    Returns:
    - lines_with_velocity: list of (angle, dist, velocity_kmh, x0, y0) tuples
    """
    print(f"\n  Calculating velocities and filtering ({velocity_min}-{velocity_max} km/h)...")
    
    lines_with_velocity = []
    
    for angle, dist in lines:
        # Calculate point on the line
        x0, y0 = dist * np.array([np.cos(angle), np.sin(angle)])
        
        # Calculate slope in image coordinates = dy/dx (pixels)
        slope_pixels = np.tan(angle + np.pi / 2)
        
        # Calculate velocity: distance/time = dx_effective / (dt_effective * slope_pixels)
        velocity_ms = abs(dx_effective / (dt_effective * slope_pixels))  # m/s
        velocity_kmh = velocity_ms * 3.6  # Convert to km/h
        
        # Filter by realistic velocity range
        if velocity_min <= velocity_kmh <= velocity_max:
            lines_with_velocity.append((angle, dist, velocity_kmh, x0, y0))
    
    print(f"  Lines after velocity filter: {len(lines)} → {len(lines_with_velocity)}")
    
    return lines_with_velocity


def cluster_lines(lines, dx_effective, dt_effective, angle_threshold=np.radians(10), dist_threshold=35):
    """
    Cluster similar lines and return representative line for each cluster.
    
    Parameters:
    - lines: list of (angle, dist, velocity_kmh, x0, y0) tuples
    - dx_effective: effective spatial resolution
    - dt_effective: effective temporal resolution
    - angle_threshold: maximum angle difference for clustering (radians)
    - dist_threshold: maximum distance difference for clustering (pixels)
    
    Returns:
    - lines_clustered: list of representative lines from each cluster
    """
    print(f"\n  Clustering lines (angle threshold: {np.degrees(angle_threshold):.1f}°, distance threshold: {dist_threshold} px)...")
    
    lines.sort(key=lambda x: x[0])  # Sort by angle
    
    clusters = []
    used = set()
    
    for i, line1 in enumerate(lines):
        if i in used:
            continue
        
        # Start a new cluster with this line
        cluster = [line1]
        used.add(i)
        
        # Find all similar lines
        for j, line2 in enumerate(lines):
            if j in used or j <= i:
                continue
            
            angle1, dist1, velocity_kmh1, x1, y1 = line1
            angle2, dist2, velocity_kmh2, x2, y2 = line2
            
            # Check if angle and distance are within thresholds
            angle_diff = abs(angle1 - angle2)
            # Handle angle wrap-around at boundaries (-π/2 and π/2)
            if angle_diff > np.pi / 2:
                angle_diff = np.pi - angle_diff
            
            dist_diff = abs(dist1 - dist2)
            
            if angle_diff <= angle_threshold and dist_diff <= dist_threshold:
                cluster.append(line2)
                used.add(j)
        
        clusters.append(cluster)
    
    # Extract representative line from each cluster
    lines_clustered = []
    
    for cluster in clusters:
        if len(cluster) % 2 != 0:
            # Odd number: take middle line
            middle_idx = int((len(cluster) - 1) / 2)
            middle_line = cluster[middle_idx]
            lines_clustered.append(middle_line)
        else:
            # Even number: average the two middle lines
            middle_line1 = cluster[len(cluster) // 2 - 1]
            middle_line2 = cluster[len(cluster) // 2]
            
            avg_angle = (middle_line1[0] + middle_line2[0]) / 2
            avg_dist = (middle_line1[1] + middle_line2[1]) / 2
            avg_x0, avg_y0 = avg_dist * np.array([np.cos(avg_angle), np.sin(avg_angle)])
            
            avg_slope_pixels = np.tan(avg_angle + np.pi / 2)
            avg_velocity_ms = abs(dx_effective / (dt_effective * avg_slope_pixels))
            avg_velocity_kmh = avg_velocity_ms * 3.6
            
            lines_clustered.append((avg_angle, avg_dist, avg_velocity_kmh, avg_x0, avg_y0))
    
    print(f"  Lines after clustering: {len(lines)} → {len(lines_clustered)} ({len(clusters)} clusters)")
    
    return lines_clustered


def visualize_detected_lines(df, image, lines, dx_effective, dt_effective, title="Detected Lines", save_path=None):
    """
    Visualize detected lines overlaid on the original image.
    
    Parameters:
    - df: DataFrame with proper index/columns for axis labels
    - image: numpy array of the image data
    - lines: list of (angle, dist, velocity_kmh, x0, y0) tuples
    - dx_effective: effective spatial resolution
    - dt_effective: effective temporal resolution
    - title: plot title
    - save_path: path to save the figure
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Display image
    im = ax.imshow(image, cmap='viridis', aspect='auto', interpolation='none')
    
    # Draw lines with velocity labels
    for angle, dist, velocity_kmh, x0, y0 in lines:
        ax.axline((x0, y0), slope=np.tan(angle + np.pi / 2), color='red', linewidth=2)
        
        # Add velocity label
        text_x = x0 + 20
        text_y = y0
        ax.text(text_x, text_y, f'v={velocity_kmh:.2f} km/h', 
                color='yellow', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Set axis limits and labels
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_ylabel('Time')
    ax.set_xlabel('Space [m]')
    ax.set_title(f'{title} ({len(lines)} lines)')
    
    # X-axis: space in meters
    space_positions = np.linspace(0, image.shape[1], 6)
    space_labels = space_positions * dx_effective
    ax.set_xticks(space_positions, np.round(space_labels, 1))
    
    # Y-axis: time labels
    time_array = np.array([t.strftime('%H:%M:%S') for t in df.index.time])
    y_positions, y_labels = set_axis(time_array)
    y_positions_scaled = y_positions * (image.shape[0] / len(df))
    ax.set_yticks(y_positions_scaled, y_labels)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Amplitude')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Saved visualization: {save_path}")
    
    plt.close()


def detect_lines(binary_df, binary_image, original_df, original_image, 
                 dx=5.106500953873407, dt=0.0016, 
                 vertical_factor=0.01, horizontal_factor=14,
                 threshold_ratio=0.85, output_dir="."):
    """
    Complete line detection pipeline.
    
    Parameters:
    - binary_df: preprocessed binary DataFrame (for visualization)
    - binary_image: preprocessed binary numpy array
    - original_df: original preprocessed DataFrame (for final visualization)
    - original_image: original preprocessed image array (for final visualization)
    - dx: spatial resolution (m)
    - dt: temporal resolution (s)
    - vertical_factor: vertical scaling factor applied during preprocessing
    - horizontal_factor: horizontal scaling factor applied during preprocessing
    - threshold_ratio: Hough threshold ratio (0-1)
    - output_dir: directory to save visualizations
    
    Returns:
    - lines_final: list of final detected lines with velocities
    """
    print("\n" + "=" * 70)
    print("LINE DETECTION PIPELINE")
    print("=" * 70)
    
    # Adjust dx and dt based on preprocessing scaling factors
    dx_effective = dx / horizontal_factor
    dt_effective = dt / vertical_factor
    print(f"\nEffective resolution: dx={dx_effective:.4f} m/pixel, dt={dt_effective:.6f} s/pixel")
    
    # Step 1: Hough line detection
    print("\n1. HOUGH LINE DETECTION")
    lines = detect_lines_hough(binary_image, threshold_ratio=threshold_ratio)
    
    # Step 2: Calculate velocities and filter
    print("\n2. VELOCITY CALCULATION AND FILTERING")
    lines_with_velocity = calculate_velocities(lines, dx_effective, dt_effective)
    
    # Visualize lines before clustering (on binary)
    visualize_detected_lines(
        binary_df, binary_image, lines_with_velocity, 
        dx_effective, dt_effective,
        title="Detected Lines (Before Clustering) - Binary",
        save_path=os.path.join(output_dir, "06_lines_before_clustering_binary.png")
    )
    
    # Step 3: Cluster lines
    print("\n3. LINE CLUSTERING")
    lines_clustered = cluster_lines(lines_with_velocity.copy(), dx_effective, dt_effective)
    
    # Visualize lines after clustering (on binary)
    visualize_detected_lines(
        binary_df, binary_image, lines_clustered,
        dx_effective, dt_effective,
        title="Detected Lines (After Clustering) - Binary",
        save_path=os.path.join(output_dir, "07_lines_after_clustering_binary.png")
    )
    
    # Step 4: Final visualization on ORIGINAL image
    print("\n4. FINAL VISUALIZATION ON ORIGINAL IMAGE")
    visualize_detected_lines(
        original_df, original_image, lines_with_velocity,
        dx_effective, dt_effective,
        title="Detected Lines (Before Clustering) - Original",
        save_path=os.path.join(output_dir, "08_lines_before_clustering_original.png")
    )
    
    visualize_detected_lines(
        original_df, original_image, lines_clustered,
        dx_effective, dt_effective,
        title="Final Detected Lines - Original Image",
        save_path=os.path.join(output_dir, "09_lines_final_original.png")
    )
    
    print("\n✓ Line detection complete")
    print(f"  Final result: {len(lines_clustered)} lines")
    print("=" * 70)
    
    return lines_clustered