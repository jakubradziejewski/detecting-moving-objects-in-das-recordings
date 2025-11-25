import numpy as np
import os
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage import data

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
    

def extract_lines(image, threshold=10, line_length=50, line_gap=10, sigma=1.0):
    """
    Extract line segments from an image using Probabilistic Hough transform.
    
    Parameters:
    - image: input grayscale image (any range/dtype)
    - threshold: minimum number of votes (intersections in Hough grid)
    - line_length: minimum accepted length of detected lines
    - line_gap: maximum gap between pixels to still form a line
    - sigma: standard deviation for Canny edge detector
    
    Returns:
    - lines: list of line segments, each as ((x0, y0), (x1, y1))
    """
    # Normalize image to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())

    # Apply edge detection first
    edges = canny(image, sigma=sigma)
    
    # Apply Probabilistic Hough transform
    lines = probabilistic_hough_line(edges, threshold=threshold, 
                                     line_length=line_length, 
                                     line_gap=line_gap)
    
    return lines


def show_lines_with_velocities(image, lines, vertical_factor, horizontal_factor, output_dir,
                               dx=DX, dt=DT):
    """
    Display image with detected line segments and velocity annotations.
    
    Parameters:
    - image: input image
    - lines: list of line segments from probabilistic_hough_line
    - vertical_factor: factor by which image was stretched/squeezed vertically
    - horizontal_factor: factor by which image was stretched/squeezed horizontally
    - dx: spatial resolution (meters per pixel) - horizontal, BEFORE stretching
    - dt: temporal resolution (seconds per pixel) - vertical, BEFORE stretching
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray', aspect='auto')
    
    # Adjust dx and dt based on stretching/squeezing factors
    dx_effective = dx / horizontal_factor
    dt_effective = dt / vertical_factor
    
    for line in lines:
        p0, p1 = line
        x0, y0 = p0
        x1, y1 = p1
        
        # Calculate angle and slope
        dx_line = x1 - x0
        dy_line = y1 - y0
        
        if dx_line == 0:
            continue  # Skip vertical lines (infinite velocity)
        
        # Slope in image coordinates (pixels)
        slope_pixels = dy_line / dx_line
        
        if slope_pixels == 0:
            continue  # Skip horizontal lines (zero velocity)
        
        # Velocity calculation: distance/time
        velocity_ms = abs(dx_effective / (dt_effective * slope_pixels))  # m/s
        velocity_kmh = velocity_ms * 3.6  # Convert to km/h
        
        if velocity_kmh > 300 or velocity_kmh < 1:
            continue  # Skip unrealistic velocities
        
        # Draw the line segment
        plt.plot([x0, x1], [y0, y1], color='red', linewidth=2)
        
        # Place text at midpoint of line
        text_x = (x0 + x1) / 2
        text_y = (y0 + y1) / 2
        plt.text(text_x, text_y, f'v={velocity_kmh:.2f} km/h', 
                color='yellow', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    plt.axis('off')
    save_path = os.path.join(output_dir, "hough_lines.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def detect_lines(image, vertical_factor, horizontal_factor, threshold, line_length, line_gap, sigma, output_dir="."):
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

    lines = extract_lines(image, threshold, line_length, line_gap, sigma)
    show_lines_with_velocities(image, lines, vertical_factor, horizontal_factor, output_dir=output_dir)