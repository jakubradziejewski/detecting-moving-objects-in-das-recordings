from skimage.filters import threshold_otsu, threshold_li, gaussian, threshold_yen
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects, skeletonize
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks
from sklearn.cluster import DBSCAN
from preprocessing import set_axis
import os
from skimage.filters import gaussian

def cluster_lines(lines, dx_effective, dt_effective, angle_threshold=np.radians(15), dist_threshold=45):    
    lines.sort(key=lambda x: x[0]) # sort by angle

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
            # Lines at -π/2 and π/2 represent the same orientation
            if angle_diff > np.pi / 2:
                angle_diff = np.pi - angle_diff
        
            dist_diff = abs(dist1 - dist2)
        
            if angle_diff <= angle_threshold and dist_diff <= dist_threshold:
                cluster.append(line2)
                used.add(j)
    
        clusters.append(cluster)
    
    lines_clustered = []

    for cluster in clusters:
        if len(cluster) % 2 != 0: 
            middle_idx = int((len(cluster)-1)/2)
            middle_line = cluster[middle_idx]
            lines_clustered.append(middle_line)
        else:
            middle_line1 = cluster[len(cluster)//2-1]
            middle_line2 = cluster[len(cluster)//2]

            avg_angle = (middle_line1[0] + middle_line2[0]) / 2
            avg_dist = (middle_line1[1] + middle_line2[1]) / 2
            avg_x0, avg_y0 = avg_dist * np.array([np.cos(avg_angle), np.sin(avg_angle)])

            avg_slope_pixels = np.tan(avg_angle + np.pi / 2)
            avg_velocity_ms = abs(dx_effective / (dt_effective * avg_slope_pixels))
            avg_velocity_kmh = avg_velocity_ms * 3.6

            lines_clustered.append((avg_angle, avg_dist, avg_velocity_kmh, avg_x0, avg_y0))

    return lines_clustered


def process_lines(lines, dx_effective, dt_effective):
    lines_processed = []
    for angle, dist in lines:
        x0, y0 = dist * np.array([np.cos(angle), np.sin(angle)])
        
        # slope in image coordinates = dy/dx (pixels)
        slope_pixels = np.tan(angle + np.pi / 2)
        
        # velocity = distance/time = dx_effective / (dt_effective * slope_pixels)
        velocity_ms = abs(dx_effective / (dt_effective * slope_pixels))  # m/s
        velocity_kmh = velocity_ms * 3.6  # Convert to km/h
        
        if velocity_kmh > 115 or velocity_kmh < 30:
            continue  # Skip unrealistic velocities
        lines_processed.append((angle, dist, velocity_kmh, x0, y0))
    return lines_processed
from skimage.filters import threshold_otsu, threshold_li, gaussian, threshold_yen
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects, skeletonize

def extract_lines_thick(image, threshold_ratio=0.85):
    """
    Extract thick lines using direct thresholding + morphology.
    """
    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    
    # Apply adaptive thresholding otsu or li works well
    thresh = threshold_li(image)
    binary = image > thresh
    
    # Clean up with morphological operations
    binary = binary_closing(binary, disk(2))
    binary = remove_small_objects(binary, min_size=100)

    # Apply Hough transform directly on binary image
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(binary, theta=tested_angles)
    
    _, angles, dists = hough_line_peaks(h, theta, d, threshold=np.max(h) * threshold_ratio)
    
    return list(zip(angles, dists))

def extract_lines_thin(image, threshold_ratio=0.7, sigma=1.0):
    """
    Extract thin/weak lines using more sensitive detection.
    Uses lower threshold and skeletonization for thin structures.
    """
    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    
    # Apply gentle smoothing to reduce noise
    smoothed = gaussian(image, sigma=sigma)
    
    # Use lower threshold (more sensitive)
    # Try adaptive threshold or percentile-based
    thresh = np.percentile(smoothed, 60)  # Lower threshold to catch weak lines
    binary = smoothed > thresh
    
    # Skeletonize to get thin line representation
    skeleton = skeletonize(binary)
    
    # Remove very small objects (noise)
    skeleton = remove_small_objects(skeleton, min_size=20)
    
    # Apply Hough transform
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(skeleton, theta=tested_angles)
    
    # Use lower threshold for thin lines
    _, angles, dists = hough_line_peaks(h, theta, d, threshold=np.max(h) * threshold_ratio)
    
    return list(zip(angles, dists))


def show_lines_comparison(df, image, vertical_factor, horizontal_factor, 
                          thick_threshold=0.85, thin_threshold=0.7, sigma=1.0,
                          dx=None, dt=None, output_dir="."):
    """
    Show SIX images: thick, thin, combined - each with NOT clustered and clustered versions.
    """
    
    dx = 5.106500953873407
    dt = 0.0016
    print("\n=== EXTRACTING LINES ===")
    
    # Extract lines using different methods
    print("\n1. THICK LINES:")
    thick_lines = extract_lines_thick(image, threshold_ratio=thick_threshold)
    
    print("\n2. THIN LINES:")
    thin_lines = extract_lines_thin(image, threshold_ratio=thin_threshold, sigma=sigma)
    print("\n3. COMBINING LINES:")
    # Combine and remove duplicates
    all_lines = thick_lines + thin_lines
    unique_lines = []
    for line in all_lines:
        is_duplicate = False
        for existing in unique_lines:
            angle_diff = abs(line[0] - existing[0])
            if angle_diff > np.pi / 2:
                angle_diff = np.pi - angle_diff
            dist_diff = abs(line[1] - existing[1])
            
            if angle_diff < np.radians(5) and dist_diff < 20:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_lines.append(line)
    
    combined_lines = unique_lines
    print(f"Combined: {len(thick_lines)} thick + {len(thin_lines)} thin = {len(all_lines)} total → {len(combined_lines)} unique")
    
    # Adjust dx and dt based on stretching/squeezing factors
    dx_effective = dx / horizontal_factor
    dt_effective = dt / vertical_factor
    
    # Process all line sets
    print("\n=== PROCESSING LINES (velocity filter) ===")
    thick_processed = process_lines(thick_lines, dx_effective, dt_effective)
    thin_processed = process_lines(thin_lines, dx_effective, dt_effective)
    combined_processed = process_lines(combined_lines, dx_effective, dt_effective)
    
    print(f"After velocity filter: Thick {len(thick_lines)}→{len(thick_processed)}, Thin {len(thin_lines)}→{len(thin_processed)}, Combined {len(combined_lines)}→{len(combined_processed)}")
    
    # Cluster all line sets
    print("\n=== CLUSTERING LINES ===")
    thick_clustered = cluster_lines(thick_processed.copy(), dx_effective, dt_effective)
    thin_clustered = cluster_lines(thin_processed.copy(), dx_effective, dt_effective)
    combined_clustered = cluster_lines(combined_processed.copy(), dx_effective, dt_effective)
    
    print(f"After clustering: Thick {len(thick_processed)}→{len(thick_clustered)}, Thin {len(thin_processed)}→{len(thin_clustered)}, Combined {len(combined_processed)}→{len(combined_clustered)}")
    
    # Create SIX subplots (3 methods × 2 versions)
    fig, axes = plt.subplots(3, 2, figsize=(20, 24))
    
    methods = [
        ('Thick Lines', thick_processed, thick_clustered),
        ('Thin Lines', thin_processed, thin_clustered),
        ('Combined', combined_processed, combined_clustered)
    ]
    
    for row, (method_name, lines_not_clustered, lines_clustered) in enumerate(methods):
        # Left column: NOT CLUSTERED
        ax_left = axes[row, 0]
        im = ax_left.imshow(image, cmap='viridis', aspect='auto', interpolation='none')
        
        for angle, dist, velocity_kmh, x0, y0 in lines_not_clustered:
            ax_left.axline((x0, y0), slope=np.tan(angle + np.pi / 2), color='red', linewidth=2)
            text_x = x0 + 20
            text_y = y0
            ax_left.text(text_x, text_y, f'v={velocity_kmh:.2f} km/h', 
                    color='yellow', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        ax_left.set_xlim(0, image.shape[1])
        ax_left.set_ylim(image.shape[0], 0)
        ax_left.set_ylabel('time')
        ax_left.set_xlabel('space [m]')
        ax_left.set_title(f'{method_name} - NOT Clustered ({len(lines_not_clustered)} lines)')
        
        # Right column: CLUSTERED
        ax_right = axes[row, 1]
        im = ax_right.imshow(image, cmap='viridis', aspect='auto', interpolation='none')
        
        for angle, dist, velocity_kmh, x0, y0 in lines_clustered:
            ax_right.axline((x0, y0), slope=np.tan(angle + np.pi / 2), color='red', linewidth=2)
            text_x = x0 + 20
            text_y = y0
            ax_right.text(text_x, text_y, f'v={velocity_kmh:.2f} km/h', 
                    color='yellow', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        ax_right.set_xlim(0, image.shape[1])
        ax_right.set_ylim(image.shape[0], 0)
        ax_right.set_ylabel('time')
        ax_right.set_xlabel('space [m]')
        ax_right.set_title(f'{method_name} - Clustered ({len(lines_clustered)} lines)')
        
        # Set axes for both
        for ax in [ax_left, ax_right]:
            # X-axis: space in meters
            space_positions = np.linspace(0, image.shape[1], 6)
            space_labels = space_positions * dx_effective
            ax.set_xticks(space_positions, np.round(space_labels, 1))
            
            # Y-axis: time labels
            time_array = np.array([t.strftime('%H:%M:%S') for t in df.index.time])
            y_positions, y_labels = set_axis(time_array)
            y_positions_scaled = y_positions * (image.shape[0] / len(df))
            ax.set_yticks(y_positions_scaled, y_labels)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "lines_comparison.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved comparison: {save_path}")
    plt.close()
    
    return combined_clustered


def detect_lines(df, image, thick_threshold=0.4, thin_threshold=0.6, sigma=1.0, output_dir="."):
    """
    Updated wrapper with correct parameter passing.
    
    Parameters:
    - thick_threshold: threshold_ratio for thick line detection (0.3-0.9)
    - thin_threshold: threshold_ratio for thin line detection (0.5-0.8)
    """
    vertical_factor = 0.01
    horizontal_factor = 14
    
    # Show comparison of all three methods
    combined_lines = show_lines_comparison(
        df, image, vertical_factor, horizontal_factor,
        thick_threshold=thick_threshold, 
        thin_threshold=thin_threshold, 
        sigma=sigma,
        output_dir=output_dir
    )
    
    print(f"\n=== FINAL RESULT: {len(combined_lines)} lines after clustering ===")