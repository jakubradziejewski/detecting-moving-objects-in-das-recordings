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


def cluster_lines(lines):    
    lines.sort(key=lambda x: x[1])
    print(len(lines), "lines detected before clustering.")

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
    
    print(len(lines), "lines detected after clustering.")

    return lines


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
    
    # Adjust dx and dt based on stretching/squeezing factors
    # If stretched (factor > 1), each pixel represents less distance/time
    # If squeezed (factor < 1), each pixel represents more distance/time
    dx_effective = dx / horizontal_factor
    dt_effective = dt / vertical_factor

    lines_processed = process_lines(lines, dx_effective, dt_effective)
    lines_clustered = cluster_lines(lines_processed)
    #lines_clustered = lines_processed

    for angle, dist, velocity_kmh, x0, y0 in lines_clustered:
        plt.axline((x0, y0), slope=np.tan(angle + np.pi / 2), color='red', linewidth=2)
        text_x = x0 + 20
        text_y = y0
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