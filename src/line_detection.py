import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks
from preprocessing import set_axis
import os


def detect_lines_hough(binary_image, threshold_ratio=0.85):
    """Run Hough transform to find lines in binary image."""
    print(f"\n  Running Hough transform (threshold ratio: {threshold_ratio})...")
    
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(binary_image, theta=tested_angles)
    _, angles, dists = hough_line_peaks(h, theta, d, threshold=np.max(h) * threshold_ratio)
    
    lines = list(zip(angles, dists))
    print(f"  Detected {len(lines)} lines")
    return lines


def calculate_velocities(lines, dx_effective, dt_effective, velocity_min=30, velocity_max=115):
    """Convert line angles to velocities, filter unrealistic speeds."""
    print(f"\n  Calculating velocities and filtering ({velocity_min}-{velocity_max} km/h)...")
    
    lines_with_velocity = []
    for angle, dist in lines:
        x0, y0 = dist * np.array([np.cos(angle), np.sin(angle)])
        slope_pixels = np.tan(angle + np.pi / 2)
        
        # velocity = distance/time in image coordinates
        velocity_ms = abs(dx_effective / (dt_effective * slope_pixels))
        velocity_kmh = velocity_ms * 3.6
        
        if velocity_min <= velocity_kmh <= velocity_max:
            lines_with_velocity.append((angle, dist, velocity_kmh, x0, y0))
    
    print(f"  Lines after velocity filter: {len(lines)} → {len(lines_with_velocity)}")
    return lines_with_velocity


def cluster_lines(lines, dx_effective, dt_effective, angle_threshold=np.radians(10), dist_threshold=35):
    """Merge similar lines - group by angle and distance."""
    print(f"\n  Clustering lines (angle threshold: {np.degrees(angle_threshold):.1f}°, distance threshold: {dist_threshold} px)...")
    
    lines.sort(key=lambda x: x[0])
    clusters = []
    used = set()
    
    for i, line1 in enumerate(lines):
        if i in used:
            continue
        
        cluster = [line1]
        used.add(i)
        
        for j, line2 in enumerate(lines):
            if j in used or j <= i:
                continue
            
            angle1, dist1, velocity_kmh1, x1, y1 = line1
            angle2, dist2, velocity_kmh2, x2, y2 = line2
            
            angle_diff = abs(angle1 - angle2)
            if angle_diff > np.pi / 2:  # handle wrap-around
                angle_diff = np.pi - angle_diff
            
            dist_diff = abs(dist1 - dist2)
            
            if angle_diff <= angle_threshold and dist_diff <= dist_threshold:
                cluster.append(line2)
                used.add(j)
        
        clusters.append(cluster)
    
    # Take middle line from each cluster (or average if even count)
    lines_clustered = []
    for cluster in clusters:
        if len(cluster) % 2 != 0:
            middle_idx = int((len(cluster) - 1) / 2)
            lines_clustered.append(cluster[middle_idx])
        else:
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
    """Draw detected lines on image with velocity labels."""
    fig, ax = plt.subplots(figsize=(16, 12))
    im = ax.imshow(image, cmap='viridis', aspect='auto', interpolation='none')
    
    # Draw lines and labels
    for angle, dist, velocity_kmh, x0, y0 in lines:
        ax.axline((x0, y0), slope=np.tan(angle + np.pi / 2), color='red', linewidth=2)
        ax.text(x0 + 20, y0, f'v={velocity_kmh:.2f} km/h', 
                color='yellow', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
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
    
    plt.colorbar(im, ax=ax, label='Amplitude')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Saved visualization: {save_path}")
    
    plt.close()


def detect_lines(binary_df, binary_image, original_df, original_image, 
                 dx=5.106500953873407, dt=0.0016, 
                 vertical_factor=0.01, horizontal_factor=14,
                 threshold_ratio=0.85, output_dir=".",
                 enable_thickness_clustering=True,
                 max_clusters=None,
                 distance_threshold=1.5,
                 use_velocity=False):
    """
    Full pipeline: detect lines, calculate velocities, cluster by position, optionally cluster by thickness.
    """
    from thickness_clustering import cluster_lines_by_thickness, visualize_thickness_clusters
    
    print("\n" + "=" * 70)
    print("LINE DETECTION PIPELINE")
    print("=" * 70)
    
    # Adjust for preprocessing scaling
    dx_effective = dx / horizontal_factor
    dt_effective = dt / vertical_factor
    print(f"\nEffective resolution: dx={dx_effective:.4f} m/pixel, dt={dt_effective:.6f} s/pixel")
    
    # Step 1: Hough line detection
    print("\n1. HOUGH LINE DETECTION")
    lines = detect_lines_hough(binary_image, threshold_ratio=threshold_ratio)
    
    # Step 2: Calculate velocities
    print("\n2. VELOCITY CALCULATION AND FILTERING")
    lines_with_velocity = calculate_velocities(lines, dx_effective, dt_effective)
    
    visualize_detected_lines(
        binary_df, binary_image, lines_with_velocity, 
        dx_effective, dt_effective,
        title="Detected Lines (Before Clustering) - Binary",
        save_path=os.path.join(output_dir, "06_lines_before_clustering_binary.png")
    )
    
    # Step 3: Spatial clustering
    print("\n3. LINE CLUSTERING (SPATIAL)")
    lines_clustered = cluster_lines(lines_with_velocity.copy(), dx_effective, dt_effective)
    
    visualize_detected_lines(
        binary_df, binary_image, lines_clustered,
        dx_effective, dt_effective,
        title="Detected Lines (After Spatial Clustering) - Binary",
        save_path=os.path.join(output_dir, "07_lines_after_clustering_binary.png")
    )
    
    # Step 4: Thickness clustering (optional)
    thickness_clusters = None
    if enable_thickness_clustering:
        print("\n4. THICKNESS-BASED CLUSTERING")
        thickness_clusters, thickness_data = cluster_lines_by_thickness(
            lines_clustered, 
            binary_image, 
            original_image,
            max_clusters=max_clusters,
            distance_threshold=distance_threshold
        )
        
        visualize_thickness_clusters(
            binary_df, binary_image, 
            thickness_clusters, thickness_data,
            dx_effective, dt_effective,
            title="Lines Clustered by Thickness - Binary",
            save_path=os.path.join(output_dir, "08_thickness_clusters_binary.png")
        )
        
        visualize_thickness_clusters(
            original_df, original_image, 
            thickness_clusters, thickness_data,
            dx_effective, dt_effective,
            title="Lines Clustered by Thickness - Original Processed",
            save_path=os.path.join(output_dir, "09_thickness_clusters_original.png")
        )
    
    # Step 5: Final visualizations on original image
    print(f"\n{5 if enable_thickness_clustering else 4}. FINAL VISUALIZATION ON ORIGINAL IMAGE")
    visualize_detected_lines(
        original_df, original_image, lines_with_velocity,
        dx_effective, dt_effective,
        title="Detected Lines (Before Clustering) - Original",
        save_path=os.path.join(output_dir, "10_lines_before_clustering_original.png")
    )
    
    visualize_detected_lines(
        original_df, original_image, lines_clustered,
        dx_effective, dt_effective,
        title="Final Detected Lines - Original Image",
        save_path=os.path.join(output_dir, "11_lines_final_original.png")
    )
    
    print("\n✓ Line detection complete")
    print(f"  Final result: {len(lines_clustered)} lines")
    if enable_thickness_clustering and thickness_clusters:
        print(f"  Thickness clusters: {len(thickness_clusters)}")
    print("=" * 70)
    
    return lines_clustered, thickness_clusters