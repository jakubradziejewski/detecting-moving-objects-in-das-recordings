import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from preprocessing import set_axis


def measure_line_thickness(binary_image, angle, dist, window=100, samples=500, percentile=90):
    """Measure line thickness by sampling perpendicular profiles."""
    height, width = binary_image.shape
    
    x0, y0 = dist * np.cos(angle), dist * np.sin(angle)
    line_dir = np.array([np.cos(angle + np.pi/2), np.sin(angle + np.pi/2)])
    perp_dir = np.array([-np.sin(angle + np.pi/2), np.cos(angle + np.pi/2)])
    
    line_length = max(height, width) * 1.5
    t_vals = np.linspace(-line_length/2, line_length/2, samples)
    
    thicknesses = []
    
    for t in t_vals:
        px, py = x0 + t * line_dir[0], y0 + t * line_dir[1]
        
        if not (0 <= px < width and 0 <= py < height):
            continue
        
        # Sample perpendicular profile
        profile = []
        for s in range(-window, window + 1):
            sx = int(px + s * perp_dir[0])
            sy = int(py + s * perp_dir[1])
            
            if 0 <= sx < width and 0 <= sy < height:
                profile.append(binary_image[sy, sx])
            else:
                profile.append(0)
        
        profile = np.array(profile, dtype=bool)
        
        # Find longest consecutive True sequence
        if np.any(profile):
            changes = np.diff(np.concatenate([[False], profile, [False]]).astype(int))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            
            if len(starts) > 0 and len(ends) > 0:
                thicknesses.append(np.max(ends - starts))
    
    if len(thicknesses) == 0:
        return 0, np.array([])
    
    return np.percentile(thicknesses, percentile), np.array(thicknesses)


def cluster_lines_by_thickness(lines, binary_image, original_image, 
                                max_clusters=None, distance_threshold=None):
    """Cluster lines by thickness using hierarchical clustering."""
    print(f"\nMeasuring thickness for {len(lines)} lines...")
    
    thickness_data = {}
    thicknesses = []
    
    for i, (angle, dist, velocity, x0, y0) in enumerate(lines):
        thick, profile = measure_line_thickness(binary_image, angle, dist)
        thicknesses.append(thick)
        thickness_data[i] = {'thickness': thick, 'velocity': velocity, 'profile': profile}
    
    features = np.array(thicknesses).reshape(-1, 1)
    
    # Normalize
    if len(features) > 1 and features.std() > 0:
        features = (features - features.mean()) / features.std()
    
    # Clustering
    if distance_threshold is not None:
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold, linkage='ward'
        )
    else:
        n_clusters = min(max_clusters or 3, len(lines))
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    
    labels = clustering.fit_predict(features)
    
    # Organize by cluster
    clustered = {}
    for i, label in enumerate(labels):
        clustered.setdefault(label, []).append((i, lines[i]))
    
    # Sort clusters by mean thickness
    summaries = []
    for cid, cluster_lines in clustered.items():
        thicks = [thickness_data[i]['thickness'] for i, _ in cluster_lines]
        vels = [line[2] for _, line in cluster_lines]
        
        summaries.append({
            'id': cid,
            'lines': cluster_lines,
            'mean_thick': np.mean(thicks),
            'mean_vel': np.mean(vels)
        })
    
    summaries.sort(key=lambda x: x['mean_thick'])
    
    # Reassign IDs by thickness order
    clustered_sorted = {}
    for new_id, summary in enumerate(summaries):
        clustered_sorted[new_id] = summary['lines']
        print(f"Cluster {new_id}: {summary['mean_thick']:.1f}px, {summary['mean_vel']:.1f}km/h")
    
    return clustered_sorted, thickness_data


def visualize_thickness_clusters(df, image, clusters, thickness_data, 
                                 dx_eff, dt_eff, title="Thickness Clusters", 
                                 save_path=None):
    """Visualize lines colored by thickness cluster."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(image, cmap='viridis', aspect='auto', interpolation='none')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
    
    for cid, cluster_lines in clusters.items():
        color = colors[cid % len(colors)]
        
        for line_idx, (angle, dist, velocity, x0, y0) in cluster_lines:
            thick = thickness_data[line_idx]['thickness']
            
            ax.axline((x0, y0), slope=np.tan(angle + np.pi/2), 
                     color=color, linewidth=2, alpha=0.8)
            ax.text(x0 + 20, y0, f'C{cid}: {velocity:.1f}km/h\n{thick:.0f}px',
                   color='yellow', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_ylabel('Time')
    ax.set_xlabel('Space [m]')
    ax.set_title(f'{title} ({len(clusters)} clusters)')
    
    # Axis labels
    space_pos = np.linspace(0, image.shape[1], 6)
    ax.set_xticks(space_pos, np.round(space_pos * dx_eff, 1))
    
    time_array = np.array([t.strftime('%H:%M:%S') for t in df.index.time])
    y_pos, y_labels = set_axis(time_array)
    ax.set_yticks(y_pos * (image.shape[0] / len(df)), y_labels)
    
    # Legend
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=colors[cid % len(colors)], 
                   label=f'Cluster {cid} ({len(lines)} lines)')
             for cid, lines in clusters.items()]
    ax.legend(handles=legend, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()