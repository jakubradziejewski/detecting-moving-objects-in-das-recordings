import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import AgglomerativeClustering


def measure_line_thickness(binary_image, angle, dist, window_half_width=100, 
                           num_samples=500, percentile=90):
    """
    Measure the thickness of a line feature in the binary image.
    
    Parameters:
    - binary_image: binary numpy array
    - angle: line angle (radians)
    - dist: line distance from origin
    - window_half_width: width of sampling window perpendicular to line (increased for better coverage)
    - num_samples: number of points to sample along the line (increased for better accuracy)
    - percentile: percentile of thickness measurements to use (default 90th for robustness)
    
    Returns:
    - thickness: measured thickness in pixels
    - intensity_profile: profile of values perpendicular to line
    """
    height, width = binary_image.shape
    
    # Get line parameters
    x0 = dist * np.cos(angle)
    y0 = dist * np.sin(angle)
    
    # Direction along the line
    line_dir = np.array([np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)])
    
    # Direction perpendicular to the line
    perp_dir = np.array([-np.sin(angle + np.pi / 2), np.cos(angle + np.pi / 2)])
    
    # Sample points along the line - use more samples for better accuracy
    line_length = max(height, width) * 1.5
    t_values = np.linspace(-line_length/2, line_length/2, num_samples)
    
    thickness_measurements = []
    
    for t in t_values:
        # Point on the line
        px = x0 + t * line_dir[0]
        py = y0 + t * line_dir[1]
        
        # Check if point is within image bounds
        if not (0 <= px < width and 0 <= py < height):
            continue
        
        # Sample perpendicular to the line
        perp_profile = []
        for s in range(-window_half_width, window_half_width + 1):
            sample_x = int(px + s * perp_dir[0])
            sample_y = int(py + s * perp_dir[1])
            
            if 0 <= sample_x < width and 0 <= sample_y < height:
                perp_profile.append(binary_image[sample_y, sample_x])
            else:
                perp_profile.append(0)
        
        perp_profile = np.array(perp_profile, dtype=bool)
        
        # Measure thickness: find the longest consecutive True sequence
        if len(perp_profile) > 0 and np.any(perp_profile):
            # Find all True regions
            changes = np.diff(np.concatenate([[False], perp_profile, [False]]).astype(int))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            
            if len(starts) > 0 and len(ends) > 0:
                # Get the longest consecutive True region
                max_thickness = np.max(ends - starts)
                thickness_measurements.append(max_thickness)
    
    if len(thickness_measurements) == 0:
        return 0, np.array([])
    
    # Use high percentile instead of median for more robust measurement
    # This captures the maximum thickness while being robust to noise
    thickness = np.percentile(thickness_measurements, percentile)
    
    return thickness, np.array(thickness_measurements)


def cluster_lines_by_thickness(lines, binary_image, original_image, 
                                max_clusters=None,
                                distance_threshold=None):
    """
    Cluster lines based on their thickness only.
    
    Parameters:
    - lines: list of (angle, dist, velocity_kmh, x0, y0) tuples
    - binary_image: binary numpy array
    - original_image: grayscale numpy array
    - max_clusters: maximum number of clusters (default: None, uses distance_threshold instead)
    - distance_threshold: distance threshold for clustering (default: None, uses max_clusters instead)
                         Higher values = fewer clusters, lower values = more clusters
    
    Returns:
    - clustered_lines: list of lists, each containing lines in same cluster
    - thickness_data: dict mapping line index to thickness info
    
    Note: Either max_clusters OR distance_threshold should be specified, not both.
          If both are None, defaults to max_clusters=3.
    """
    print(f"\n  Measuring line thickness for {len(lines)} lines...")
    
    # Measure thickness for each line
    thickness_data = {}
    thicknesses = []
    velocities = []
    
    for i, (angle, dist, velocity_kmh, x0, y0) in enumerate(lines):
        thickness, profile = measure_line_thickness(binary_image, angle, dist)
        thicknesses.append(thickness)
        velocities.append(velocity_kmh)
        
        thickness_data[i] = {
            'thickness': thickness,
            'velocity': velocity_kmh,
            'profile': profile
        }
        
        print(f"    Line {i}: thickness={thickness:.1f}px, velocity={velocity_kmh:.1f}km/h")
    

    features = np.array(thicknesses).reshape(-1, 1)
    print(f"\n  Clustering based on: thickness only")

    
    # Normalize features (important for thickness values)
    features_normalized = features.copy()
    if features.shape[0] > 1 and features.shape[1] > 0:
        for col in range(features.shape[1]):
            col_std = features[:, col].std()
            if col_std > 0:
                features_normalized[:, col] = (features[:, col] - features[:, col].mean()) / col_std
    
    # Determine clustering parameters
    if max_clusters is None and distance_threshold is None:
        # Default: use max_clusters = 3
        max_clusters = 3
        print(f"\n  No clustering parameters specified, using default max_clusters=3")
    
    if distance_threshold is not None:
        # Use distance threshold (allows natural number of clusters)
        print(f"\n  Using distance threshold: {distance_threshold}")
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage='ward'
        )
    else:
        # Use fixed number of clusters
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
    
    # Print cluster summary sorted by mean thickness
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
    
    # Sort by thickness and reassign cluster IDs (0 = thinnest, n = thickest)
    cluster_summaries.sort(key=lambda x: x['mean_thickness'])
    
    # Reorganize with new IDs
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
    """
    Visualize lines colored by their thickness cluster.
    
    Parameters:
    - df: DataFrame with proper index/columns for axis labels
    - image: numpy array of the image data
    - clustered_lines: dict mapping cluster_id to list of (index, line) tuples
    - thickness_data: dict with thickness information
    - dx_effective: effective spatial resolution
    - dt_effective: effective temporal resolution
    - title: plot title
    - save_path: path to save the figure
    """
    from preprocessing import set_axis
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Display image
    im = ax.imshow(image, cmap='viridis', aspect='auto', interpolation='none')
    
    # Color palette for clusters
    colors = plt.cm.tab10(np.linspace(0, 1, len(clustered_lines)))
    
    # Draw lines colored by cluster
    for cluster_id, cluster_lines in clustered_lines.items():
        color = colors[cluster_id % len(colors)]
        
        for line_idx, (angle, dist, velocity_kmh, x0, y0) in cluster_lines:
            thickness = thickness_data[line_idx]['thickness']
            
            # Draw line
            ax.axline((x0, y0), slope=np.tan(angle + np.pi / 2), 
                     color=color, linewidth=2, alpha=0.8)
            
            # Add label
            text_x = x0 + 20
            text_y = y0
            ax.text(text_x, text_y, 
                   f'C{cluster_id}: v={velocity_kmh:.1f}km/h\nt={thickness:.0f}px', 
                   color='yellow', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Set axis limits and labels
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
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Amplitude')
    
    # Add legend for clusters
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