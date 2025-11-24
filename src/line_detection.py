import numpy as np

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage import data
from scipy.ndimage import zoom

import matplotlib.pyplot as plt
from matplotlib import cm

DX = 5.106500953873407
DT = 0.0016


def squeeze_image(img, axis, factor):
    """
    Squeeze image along specified axis using averaging for sharp results.
    
    Parameters:
    - img: input image
    - axis: 0 for vertical squeeze, 1 for horizontal squeeze
    - factor: squeeze factor (< 1). e.g., 0.5 = squeeze to half size
    
    Returns:
    - squeezed image
    """
    if factor >= 1:
        raise ValueError("Squeeze factor must be < 1")
    
    int_factor = int(1 / factor)
    
    if axis == 0:  # Vertical squeeze
        new_height = img.shape[0] // int_factor
        return img[:new_height * int_factor].reshape(new_height, int_factor, img.shape[1]).mean(axis=1)
    else:  # Horizontal squeeze
        new_width = img.shape[1] // int_factor
        return img[:, :new_width * int_factor].reshape(img.shape[0], new_width, int_factor).mean(axis=2)


def stretch_image(img, axis, factor):
    """
    Stretch image along specified axis using repetition for sharp results.
    
    Parameters:
    - img: input image
    - axis: 0 for vertical stretch, 1 for horizontal stretch
    - factor: stretch factor (> 1). e.g., 2.0 = stretch to double size
    
    Returns:
    - stretched image
    """
    if factor <= 1:
        raise ValueError("Stretch factor must be > 1")
    
    if factor == int(factor):
        return np.repeat(img, int(factor), axis=axis)
    

def extract_lines(image, threshold_ratio=0.85, sigma=1.0):
    """
    Extract lines from an image using Hough transform.
    
    Parameters:
    - image: input grayscale image (any range/dtype)
    - threshold_ratio: ratio of max hough space value to use as threshold
    - sigma: standard deviation for Canny edge detector
    
    Returns:
    - lines: list of tuples (angle, dist) for each detected line
    """
    image = (image - image.min()) / (image.max() - image.min())

    # Apply edge detection first
    edges = canny(image, sigma=sigma)
    
    # Now apply Hough transform on the binary edge image
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(edges, theta=tested_angles)
    
    # Extract peaks (detected lines)
    _, angles, dists = hough_line_peaks(h, theta, d, threshold=np.max(h) * threshold_ratio)
    
    # Return as list of (angle, distance) tuples
    lines = list(zip(angles, dists))
    
    return lines


def get_points(lines, width, height):
    """
    Find intersection points of a line with image boundaries.
    
    Parameters:
    - angle: angle from Hough transform (in radians)
    - dist: distance from Hough transform
    - x0, y0: a point on the line
    - width, height: image dimensions
    
    Returns:
    - List of ((x0, y0),(xk,yk),km_h) intersection points within image boundaries
    """
    lines_transformed = []

    for line in lines:
        angle, dist, km_h, x0, y0 = line
    
        # Line direction (perpendicular to the normal angle from Hough)
        dx = -np.sin(angle)
        dy = np.cos(angle)


        if dx == 0 or dy == 0:
            continue

        points = []

        # Left boundary (x = 0)
        t = (0 - x0) / dx
        y = y0 + t * dy
        if 0 <= y <= height:
            points.append((0, y))
    
        # Right boundary (x = width)
        t = (width - x0) / dx
        y = y0 + t * dy
        if 0 <= y <= height:
            points.append((width, y))

        # Top boundary (y = 0)
        t = (0 - y0) / dy
        x = x0 + t * dx
        if 0 <= x <= width:
            points.append((x, 0))
    
        # Bottom boundary (y = height)
        t = (height - y0) / dy
        x = x0 + t * dx
        if 0 <= x <= width:
            points.append((x, height))

        if points[0][0] < points[1][0]:
            lines_transformed.append((points[0], points[1], km_h))

    return lines_transformed


def cluster_lines(lines):    
    lines.sort(key=lambda x: x[1])
    #print(len(lines), "lines detected before clustering.")

    # Define thresholds for clustering
    angle_threshold = np.radians(15)
    dist_threshold = 45
    clusters = []
    used = set()

    for i, line1 in enumerate(lines):
        print(line1)
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

    for cluster in clusters:
        if len(cluster) > 1:
            # Average the lines in the cluster
            avg_angle = np.mean([line[0] for line in cluster])
            avg_dist = np.mean([line[1] for line in cluster])
            avg_velocity_kmh = np.mean([line[2] for line in cluster])
            avg_x0 = np.mean([line[3] for line in cluster])
            avg_y0 = np.mean([line[4] for line in cluster])
        
            # Remove individual lines from lines1
            for line in cluster:
                lines.remove(line)
        
            # Add the averaged line back to lines1
            lines.append((avg_angle, avg_dist, avg_velocity_kmh, avg_x0, avg_y0))
    
    #print(len(lines), "lines detected after clustering.")

    return lines


def compute_line_distance(line1, line2):
    """
    Compute Manhattan distance between two lines' endpoints.
    Tries both orientations and returns minimum.
    """
    (x0_1, y0_1), (xk_1, yk_1), _ = line1
    (x0_2, y0_2), (xk_2, yk_2), _ = line2
    
    # Try both orientations
    dist1 = abs(x0_1 - x0_2) + abs(y0_1 - y0_2) + abs(xk_1 - xk_2) + abs(yk_1 - yk_2)
    dist2 = abs(x0_1 - xk_2) + abs(y0_1 - yk_2) + abs(xk_1 - x0_2) + abs(yk_1 - y0_2)
    
    return min(dist1, dist2)


def build_distance_matrix(lines):
    """
    Build n x n distance matrix for all lines.
    """
    n = len(lines)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = compute_line_distance(lines[i], lines[j])
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    
    # Set diagonal to infinity so we don't cluster a line with itself
    np.fill_diagonal(dist_matrix, np.inf)
    
    return dist_matrix


def average_lines(lines, indices):
    """
    Compute average line from a list of line indices.
    """
    x0_sum = y0_sum = xk_sum = yk_sum = vel_sum = 0
    count = len(indices)
    
    for idx in indices:
        (x0, y0), (xk, yk), velocity = lines[idx]
        x0_sum += x0
        y0_sum += y0
        xk_sum += xk
        yk_sum += yk
        vel_sum += velocity
    
    return (
        (x0_sum / count, y0_sum / count),
        (xk_sum / count, yk_sum / count),
        vel_sum / count
    )


def hierarchical_cluster_lines(lines_processed, threshold):
    """
    Agglomerative hierarchical clustering of lines.
    
    Algorithm:
    1. Compute n x n distance matrix
    2. Find minimum distance pair
    3. If distance < threshold, merge them
    4. Update distance matrix with centroid distances
    5. Repeat until no more pairs below threshold
    
    Parameters:
    - lines_processed: list of ((x0, y0), (xk, yk), velocity_kmh)
    - threshold: maximum distance for merging
    
    Returns:
    - List of merged lines
    """
    n = len(lines_processed)
    
    # Each cluster is represented by a list of original line indices
    clusters = [[i] for i in range(n)]
    
    # Build initial distance matrix
    dist_matrix = build_distance_matrix(lines_processed)
    
    # Keep track of active clusters (not yet merged)
    active = [True] * n
    
    while True:
        # Find minimum distance among active clusters
        min_dist = np.inf
        min_i = min_j = -1
        
        for i in range(n):
            if not active[i]:
                continue
            for j in range(i + 1, n):
                if not active[j]:
                    continue
                if dist_matrix[i][j] < min_dist:
                    min_dist = dist_matrix[i][j]
                    min_i = i
                    min_j = j
        
        # If minimum distance exceeds threshold, stop clustering
        if min_dist >= threshold:
            break
        
        # Merge clusters i and j
        clusters[min_i].extend(clusters[min_j])
        active[min_j] = False
        
        # Compute centroid of merged cluster
        centroid_line = average_lines(lines_processed, clusters[min_i])
        
        # Update distances from merged cluster to all other active clusters
        for k in range(n):
            if not active[k] or k == min_i:
                continue
            
            # Compute distance from centroid to cluster k's centroid
            centroid_k = average_lines(lines_processed, clusters[k])
            new_dist = compute_line_distance(centroid_line, centroid_k)
            
            dist_matrix[min_i][k] = new_dist
            dist_matrix[k][min_i] = new_dist
    
    # Extract final clusters and compute their average lines
    merged_lines = []
    for i in range(n):
        if active[i]:
            merged_line = average_lines(lines_processed, clusters[i])
            merged_lines.append(merged_line)
    
    return merged_lines


def process_lines(lines, dx_effective, dt_effective):
    lines_processed = []
    for angle, dist in lines:
        x0, y0 = dist * np.array([np.cos(angle), np.sin(angle)])
        
        # slope in image coordinates = dy/dx (pixels)
        slope_pixels = np.tan(angle + np.pi / 2)
        
        # velocity = distance/time = dx_effective / (dt_effective * slope_pixels)
        velocity_ms = abs(dx_effective / (dt_effective * slope_pixels))  # m/s
        velocity_kmh = velocity_ms * 3.6  # Convert to km/h
        
        if velocity_kmh > 200 or velocity_kmh < 1:
            continue  # Skip unrealistic velocities
        lines_processed.append((angle, dist, velocity_kmh, x0, y0))
    return lines_processed


def show_lines_with_velocities(image, lines, vertical_factor, horizontal_factor, dx=DX, dt=DT):
    """
    Display image with detected lines and velocity annotations.
    
    Parameters:
    - scaled_img: input image
    - lines: list of (angle, dist) tuples
    - dx: spatial resolution (meters per pixel) - horizontal, BEFORE any stretching/squeezing
    - dt: temporal resolution (seconds per pixel) - vertical, BEFORE any stretching/squeezing
    - vertical_factor: factor by which image was stretched/squeezed vertically (>1 = stretched, <1 = squeezed)
    - horizontal_factor: factor by which image was stretched/squeezed horizontally (>1 = stretched, <1 = squeezed)
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray', aspect='auto')
    
    dx_effective = dx / horizontal_factor
    dt_effective = dt / vertical_factor

    lines_processed = process_lines(lines, dx_effective, dt_effective)
    lines_processed = get_points(lines_processed, image.shape[1], image.shape[0])
    result = hierarchical_cluster_lines(lines_processed, 100)
    #result = lines_processed

    for (x0, y0), (xk, yk), velocity_kmh in result:
        plt.plot([x0, xk], [y0, yk], color='red', linewidth=2)
        text_x = (x0 + xk) / 2
        text_y = (y0 + yk) / 2
        plt.text(text_x, text_y, f'v={velocity_kmh:.2f} km/h', 
            color='yellow', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    plt.axis('off')
    plt.savefig("hough_lines_clustered.png", dpi=150)
    plt.close()


def detect_lines(image, vertical_factor, horizontal_factor, threshold_ratio=0.85, sigma=1.0):
    """
    Wrapper to extract lines from an image and display them with velocities.
    
    Parameters:
    - image: input grayscale image
    - vertical_factor: factor by which image was stretched/squeezed vertically
    - horizontal_factor: factor by which image was stretched/squeezed horizontally
    - threshold_ratio: ratio of max hough space value to use as threshold
    - sigma: standard deviation for Canny edge detector
    """
    image = squeeze_image(image, axis=0, factor=vertical_factor)
    image = stretch_image(image, axis=1, factor=horizontal_factor)

    lines = extract_lines(image, threshold_ratio=threshold_ratio, sigma=sigma)
    show_lines_with_velocities(image, lines, vertical_factor, horizontal_factor)