import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from preprocessing import set_axis


def measure_line_thickness(binary_image, angle, dist, window_half_width=100, 
                           num_samples=500, percentile=90):
    """
    Measure line thickness by sampling perpendicular profiles along the line.
    Takes 90th percentile of measurements for robustness against noise.
    """
    height, width = binary_image.shape
    
    x0 = dist * np.cos(angle)
    y0 = dist * np.sin(angle)
    
    # Directions: along line and perpendicular
    line_dir = np.array([np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)])
    perp_dir = np.array([-np.sin(angle + np.pi / 2), np.cos(angle + np.pi / 2)])
    
    # Sample many points along the line
    line_length = max(height, width) * 1.5
    t_values = np.linspace(-line_length/2, line_length/2, num_samples)
    
    thickness_measurements = []
    
    for t in t_values:
        px = x0 + t * line_dir[0]
        py = y0 + t * line_dir[1]
        
        if not (0 <= px < width and 0 <= py < height):
            continue
        
        # Sample perpendicular profile
        perp_profile = []
        for s in range(-window_half_width, window_half_width + 1):
            sample_x = int(px + s * perp_dir[0])
            sample_y = int(py + s * perp_dir[1])
            
            if 0 <= sample_x < width and 0 <= sample_y < height:
                perp_profile.append(binary_image[sample_y, sample_x])
            else:
                perp_profile.append(0)
        
        perp_profile = np.array(perp_profile, dtype=bool)
        
        # Find longest consecutive True sequence
        if len(perp_profile) > 0 and np.any(perp_profile):
            changes = np.diff(np.concatenate([[False], perp_profile, [False]]).astype(int))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            
            if len(starts) > 0 and len(ends) > 0:
                max_thickness = np.max(ends - starts)
                thickness_measurements.append(max_thickness)
    
    if len(thickness_measurements) == 0:
        return 0, np.array([])
    
    thickness = np.percentile(thickness_measurements, percentile)
    return thickness, np.array(thickness_measurements)


def cluster_lines_by_thickness(lines, binary_image, original_image, 
                                max_clusters=None, distance_threshold=None):
    """
    Group lines by thickness using hierarchical clustering.
    Either specify max_clusters (fixed number) or distance_threshold (natural grouping).
    """
    print(f"\n  Measuring line thickness for {len(lines)} lines...")
    
    # Measure thickness for each line
    thickness_data = {}
    thicknesses = []
    
    for i, (angle, dist, velocity_kmh, x0, y0) in enumerate(lines):
        thickness, profile = measure_line_thickness(binary_image, angle, dist)
        thicknesses.append(thickness)
        
        thickness_data[i] = {
            'thickness': thickness,
            'velocity': velocity_kmh,
            'profile': profile
        }
        
        print(f"    Line {i}: thickness={thickness:.1f}px, velocity={velocity_kmh:.1f}km/h")
    
    features = np.array(thicknesses).reshape(-1, 1)
    print(f"\n  Clustering based on: thickness only")
    
    # Normalize features - important for clustering algorithm
    features_normalized = features.copy()
    if features.shape[0] > 1:
        col_std = features[:, 0].std()
        if col_std > 0:
            features_normalized[:, 0] = (features[:, 0] - features[:, 0].mean()) / col_std
    
    # Setup clustering
    if max_clusters is None and distance_threshold is None:
        max_clusters = 3
        print(f"\n  No clustering parameters specified, using default max_clusters=3")
    
    if distance_threshold is not None:
        print(f"\n  Using distance threshold: {distance_threshold}")
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage='ward'
        )
    else:
        n_clusters_target = min(max_clusters, len(lines))
        print(f"\n  Using max clusters: {max_clusters} (actual: {n_clusters_target})")
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters_target,
            linkage='ward'
        )
    
    labels = clustering.fit_predict(features_normalized)
    n_clusters = len(set(labels))
    print(f"  Created {n_clusters} thickness-based clusters")
    
    # Organize lines by cluster
    clustered_lines = {}
    for i, label in enumerate(labels):
        if label not in clustered_lines:
            clustered_lines[label] = []
        clustered_lines[label].append((i, lines[i]))
    
    # Sort clusters by mean thickness (thinnest = 0, thickest = n)
    cluster_summaries = []
    for cluster_id, cluster_lines in clustered_lines.items():
        thicknesses = [thickness_data[i]['thickness'] for i, _ in cluster_lines]
        velocities = [line[2] for _, line in cluster_lines]
        
        cluster_summaries.append({
            'id': cluster_id,
            'lines': cluster_lines,
            'mean_thickness': np.mean(thicknesses),
            'std_thickness': np.std(thicknesses),
            'mean_velocity': np.mean(velocities),
            'std_velocity': np.std(velocities),
            'count': len(cluster_lines)
        })
    
    cluster_summaries.sort(key=lambda x: x['mean_thickness'])
    
    # Reassign IDs based on thickness order
    clustered_lines_sorted = {}
    for new_id, summary in enumerate(cluster_summaries):
        clustered_lines_sorted[new_id] = summary['lines']
        
        print(f"\n  Cluster {new_id} (n={summary['count']}):")
        print(f"    Thickness: {summary['mean_thickness']:.1f} ± {summary['std_thickness']:.1f} px")
        print(f"    Velocity: {summary['mean_velocity']:.1f} ± {summary['std_velocity']:.1f} km/h")
    
    return clustered_lines_sorted, thickness_data


def visualize_thickness_clusters(df, image, clustered_lines, thickness_data, 
                                 dx_effective, dt_effective, 
                                 title="Lines Clustered by Thickness", 
                                 save_path=None):
    """Draw lines colored by cluster, showing thickness and velocity."""
    fig, ax = plt.subplots(figsize=(16, 12))
    
    im = ax.imshow(image, cmap='viridis', aspect='auto', interpolation='none')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(clustered_lines)))
    
    # Draw each cluster with its color
    for cluster_id, cluster_lines in clustered_lines.items():
        color = colors[cluster_id % len(colors)]
        
        for line_idx, (angle, dist, velocity_kmh, x0, y0) in cluster_lines:
            thickness = thickness_data[line_idx]['thickness']
            
            ax.axline((x0, y0), slope=np.tan(angle + np.pi / 2), 
                     color=color, linewidth=2, alpha=0.8)
            
            ax.text(x0 + 20, y0, 
                   f'C{cluster_id}: v={velocity_kmh:.1f}km/h\nt={thickness:.0f}px', 
                   color='yellow', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_ylabel('Time')
    ax.set_xlabel('Space [m]')
    ax.set_title(f'{title} ({len(clustered_lines)} clusters)')
    
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
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[cid % len(colors)], 
                            label=f'Cluster {cid} ({len(lines)} lines)')
                      for cid, lines in clustered_lines.items()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Saved visualization: {save_path}")
    
    plt.close()