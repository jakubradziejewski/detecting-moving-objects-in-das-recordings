import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks
from preprocessing import set_axis
import os


def detect_lines_hough(binary_image, threshold_ratio=0.85):
    """Detect lines using Hough transform."""
    tested_angles = np.linspace(-np.pi/2, np.pi/2, 360, endpoint=False)
    h, theta, d = hough_line(binary_image, theta=tested_angles)
    _, angles, dists = hough_line_peaks(h, theta, d, threshold=np.max(h) * threshold_ratio)
    return list(zip(angles, dists))


def calculate_velocities(lines, dx_eff, dt_eff, v_min=30, v_max=115):
    """Calculate velocities from line angles and filter by speed range."""
    lines_with_velocity = []
    
    for angle, dist in lines:
        x0, y0 = dist * np.array([np.cos(angle), np.sin(angle)])
        slope = np.tan(angle + np.pi/2)
        velocity_kmh = abs(dx_eff / (dt_eff * slope)) * 3.6
        
        if v_min <= velocity_kmh <= v_max:
            lines_with_velocity.append((angle, dist, velocity_kmh, x0, y0))
    
    return lines_with_velocity


def cluster_lines(lines, dx_eff, dt_eff, angle_thresh=np.radians(10), dist_thresh=35):
    """Merge similar lines by angle and distance."""
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
            
            angle_diff = abs(line1[0] - line2[0])
            if angle_diff > np.pi/2:
                angle_diff = np.pi - angle_diff
            
            if angle_diff <= angle_thresh and abs(line1[1] - line2[1]) <= dist_thresh:
                cluster.append(line2)
                used.add(j)
        
        clusters.append(cluster)
    
    # Select representative line from each cluster
    lines_clustered = []
    for cluster in clusters:
        if len(cluster) % 2 != 0:
            lines_clustered.append(cluster[len(cluster) // 2])
        else:
            # Average middle two lines
            mid1, mid2 = cluster[len(cluster)//2 - 1], cluster[len(cluster)//2]
            avg_angle = (mid1[0] + mid2[0]) / 2
            avg_dist = (mid1[1] + mid2[1]) / 2
            avg_x0, avg_y0 = avg_dist * np.array([np.cos(avg_angle), np.sin(avg_angle)])
            avg_velocity = abs(dx_eff / (dt_eff * np.tan(avg_angle + np.pi/2))) * 3.6
            lines_clustered.append((avg_angle, avg_dist, avg_velocity, avg_x0, avg_y0))
    
    return lines_clustered


def visualize_lines(df, image, lines, dx_eff, dt_eff, title="Detected Lines", save_path=None):
    """Visualize detected lines on DAS image."""
    fig, ax = plt.subplots(figsize=(16, 12))
    im = ax.imshow(image, cmap='viridis', aspect='auto', interpolation='none')
    
    for angle, dist, velocity_kmh, x0, y0 in lines:
        ax.axline((x0, y0), slope=np.tan(angle + np.pi/2), color='red', linewidth=2)
        ax.text(x0 + 20, y0, f'{velocity_kmh:.1f} km/h', 
                color='yellow', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_ylabel('Time')
    ax.set_xlabel('Space [m]')
    ax.set_title(f'{title} ({len(lines)} lines)')
    
    # Axis labels
    space_pos = np.linspace(0, image.shape[1], 6)
    ax.set_xticks(space_pos, np.round(space_pos * dx_eff, 1))
    
    time_array = np.array([t.strftime('%H:%M:%S') for t in df.index.time])
    y_pos, y_labels = set_axis(time_array)
    ax.set_yticks(y_pos * (image.shape[0] / len(df)), y_labels)
    
    plt.colorbar(im, ax=ax, label='Amplitude')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def detect_lines(binary_df, binary_image, original_df, original_image, 
                 dx, dt, vertical_factor, horizontal_factor,
                 threshold_ratio, output_dir,
                 enable_thickness_clustering=True,
                 max_clusters=None, distance_threshold=1.5, use_velocity=False):
    """Full line detection pipeline."""
    from thickness_clustering import cluster_lines_by_thickness, visualize_thickness_clusters
    
    print(f"\n{'='*70}\nLINE DETECTION\n{'='*70}")
    
    dx_eff = dx / horizontal_factor
    dt_eff = dt / vertical_factor
    
    # Detect lines
    lines = detect_lines_hough(binary_image, threshold_ratio)
    
    # Calculate velocities and filter
    lines_with_velocity = calculate_velocities(lines, dx_eff, dt_eff)
    
    visualize_lines(binary_df, binary_image, lines_with_velocity, dx_eff, dt_eff,
                   "Before Clustering", 
                   os.path.join(output_dir, "06_lines_before_clustering_binary.png"))
    
    # Spatial clustering
    lines_clustered = cluster_lines(lines_with_velocity.copy(), dx_eff, dt_eff)
    
    visualize_lines(binary_df, binary_image, lines_clustered, dx_eff, dt_eff,
                   "After Spatial Clustering",
                   os.path.join(output_dir, "07_lines_after_clustering_binary.png"))
    
    # Thickness clustering
    thickness_clusters = None
    if enable_thickness_clustering:
        thickness_clusters, thickness_data = cluster_lines_by_thickness(
            lines_clustered, binary_image, original_image,
            max_clusters, distance_threshold
        )
        
        visualize_thickness_clusters(
            binary_df, binary_image, thickness_clusters, thickness_data,
            dx_eff, dt_eff, "Thickness Clusters - Binary",
            os.path.join(output_dir, "08_thickness_clusters_binary.png")
        )
        
        visualize_thickness_clusters(
            original_df, original_image, thickness_clusters, thickness_data,
            dx_eff, dt_eff, "Thickness Clusters - Original",
            os.path.join(output_dir, "09_thickness_clusters_original.png")
        )
    
    # Final visualizations
    visualize_lines(original_df, original_image, lines_with_velocity, dx_eff, dt_eff,
                   "Before Clustering - Original",
                   os.path.join(output_dir, "10_lines_before_clustering_original.png"))
    
    visualize_lines(original_df, original_image, lines_clustered, dx_eff, dt_eff,
                   "Final Lines - Original",
                   os.path.join(output_dir, "11_lines_final_original.png"))
    
    print(f"✓ Detected {len(lines_clustered)} lines")
    if thickness_clusters:
        print(f"✓ {len(thickness_clusters)} thickness clusters")
    
    return lines_clustered, thickness_clusters