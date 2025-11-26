import numpy as np
import os
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage import data
from preprocessing import set_axis

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

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
    # normalize to [0, 1] range
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
        
        if velocity_kmh > 140 or velocity_kmh < 30:
            continue  # Skip unrealistic velocities
        lines_processed.append((angle, dist, velocity_kmh, x0, y0))
    return lines_processed


def show_lines_clustered(df, image, lines, vertical_factor, horizontal_factor, dx=DX, dt=DT, output_dir="."):
    """
    Display image with detected lines and velocity annotations.
    
    Parameters:
    - image: input image
    - lines: list of (angle, dist) tuples
    - vertical_factor: factor by which image was stretched/squeezed vertically (>1 = stretched, <1 = squeezed)
    - horizontal_factor: factor by which image was stretched/squeezed horizontally (>1 = stretched, <1 = squeezed)
    - dx: spatial resolution (meters per pixel) - horizontal, BEFORE any stretching/squeezing
    - dt: temporal resolution (seconds per pixel) - vertical, BEFORE any stretching/squeezing
    """
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    
    im = ax.imshow(image, cmap='viridis', aspect='auto', interpolation='none')

    # Adjust dx and dt based on stretching/squeezing factors
    dx_effective = dx / horizontal_factor
    dt_effective = dt / vertical_factor

    lines_processed = process_lines(lines, dx_effective, dt_effective)
    lines_clustered = cluster_lines(lines_processed, dx_effective, dt_effective, angle_threshold=np.radians(15), dist_threshold=45)

    for angle, dist, velocity_kmh, x0, y0 in lines_clustered:
        ax.axline((x0, y0), slope=np.tan(angle + np.pi / 2), color='red', linewidth=2)
        text_x = x0 + 20
        text_y = y0
        ax.text(text_x, text_y, f'v={velocity_kmh:.2f} km/h', 
                color='yellow', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)

    # Set up axes labels
    ax.set_ylabel('time')
    ax.set_xlabel('space [m]')
    ax.set_title('Lines Clustered')

    # X-axis: space in meters
    space_positions = np.linspace(0, image.shape[1], 6)  # 6 tick marks
    space_labels = space_positions * dx_effective
    ax.set_xticks(space_positions, np.round(space_labels, 1))

    # Y-axis: time labels scaled to match transformed image height
    time_array = np.array([t.strftime('%H:%M:%S') for t in df.index.time])
    y_positions, y_labels = set_axis(time_array)
    
    # Scale y_positions to match the actual image height
    y_positions_scaled = y_positions * (image.shape[0] / len(df))
    ax.set_yticks(y_positions_scaled, y_labels)

    # Add colorbar on the right
    cax = fig.add_axes([ax.get_position().x1 + 0.06,
                        ax.get_position().y0,
                        0.02,
                        ax.get_position().height])
    plt.colorbar(im, cax=cax)

    plt.tight_layout()

    save_path = os.path.join(output_dir, "lines_clustered.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
    

def show_lines_not_clustered(df, image, lines, vertical_factor, horizontal_factor, dx=DX, dt=DT, output_dir="."):
    """
    Display image with detected lines and velocity annotations.
    
    Parameters:
    - image: input image
    - lines: list of (angle, dist) tuples
    - vertical_factor: factor by which image was stretched/squeezed vertically (>1 = stretched, <1 = squeezed)
    - horizontal_factor: factor by which image was stretched/squeezed horizontally (>1 = stretched, <1 = squeezed)
    - dx: spatial resolution (meters per pixel) - horizontal, BEFORE any stretching/squeezing
    - dt: temporal resolution (seconds per pixel) - vertical, BEFORE any stretching/squeezing
    """
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    
    im = ax.imshow(image, cmap='viridis', aspect='auto', interpolation='none')

    # Adjust dx and dt based on stretching/squeezing factors
    dx_effective = dx / horizontal_factor
    dt_effective = dt / vertical_factor

    lines_processed = process_lines(lines, dx_effective, dt_effective)

    for angle, dist, velocity_kmh, x0, y0 in lines_processed:
        ax.axline((x0, y0), slope=np.tan(angle + np.pi / 2), color='red', linewidth=2)
        text_x = x0 + 20
        text_y = y0
        ax.text(text_x, text_y, f'v={velocity_kmh:.2f} km/h', 
                color='yellow', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)

    # Set up axes labels
    ax.set_ylabel('time')
    ax.set_xlabel('space [m]')
    ax.set_title('Lines not clustered')

    # X-axis: space in meters
    space_positions = np.linspace(0, image.shape[1], 6)  # 6 tick marks
    space_labels = space_positions * dx_effective
    ax.set_xticks(space_positions, np.round(space_labels, 1))

    # Y-axis: time labels scaled to match transformed image height
    time_array = np.array([t.strftime('%H:%M:%S') for t in df.index.time])
    y_positions, y_labels = set_axis(time_array)
    
    # Scale y_positions to match the actual image height
    y_positions_scaled = y_positions * (image.shape[0] / len(df))
    ax.set_yticks(y_positions_scaled, y_labels)

    # Add colorbar on the right
    cax = fig.add_axes([ax.get_position().x1 + 0.06,
                        ax.get_position().y0,
                        0.02,
                        ax.get_position().height])
    plt.colorbar(im, cax=cax)

    plt.tight_layout()

    save_path = os.path.join(output_dir, "lines_not_clustered.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def detect_lines(df, image, vertical_factor, horizontal_factor, threshold_ratio=0.85, sigma=1.0, output_dir="."):
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

    show_lines_not_clustered(df, image, lines, vertical_factor, horizontal_factor, output_dir=output_dir)
    show_lines_clustered(df, image, lines, vertical_factor, horizontal_factor, output_dir=output_dir)