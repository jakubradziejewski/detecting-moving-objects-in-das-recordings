import sys
import os
import warnings

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_das_segment
from preprocessing import preprocess_das_data
from line_detection import detect_lines
from analysis import analyze_statistics, analyze_frequency_content, visualize_filtered_comparison
from velocity_analysis import analyze_velocity_over_time

# Configuration
DATA_PATH = '../data'
BASE_OUTPUT_DIR = '../output'
DX = 5.106500953873407
DT = 0.0016
FS = 625

SEGMENTS = [
    {'start': '090522', 'end': '090712', 'name': 'segment_1'},
    {'start': '093252', 'end': '093442', 'name': 'segment_2'},
    {'start': '092112', 'end': '092302', 'name': 'segment_3'}
]


def analyze_segment(segment_info, data_path, dx, dt, fs, base_output_dir):
    """Pipeline: load → analyze → preprocess → detect → velocity analysis."""
    start_time = segment_info['start']
    end_time = segment_info['end']
    segment_name = segment_info['name']
    
    output_dir = os.path.join(base_output_dir, segment_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"PROCESSING {segment_name} ({start_time} to {end_time})")
    print(f"{'='*70}")
    
    # Load data
    df_raw = load_das_segment(start_time, end_time, data_path, dx, dt, verbose=True)
    
    # Statistical analysis
    analyze_statistics(df_raw, dt, dx, f"{start_time}_{end_time}", output_dir)
    analyze_frequency_content(df_raw, fs, output_dir)
    visualize_filtered_comparison(df_raw, fs, output_dir)
    
    # Preprocessing
    binary_df, binary_image, original_df, original_image = preprocess_das_data(
        df_raw, target_size=750, threshold_multiplier=0.8, output_dir=output_dir
    )
    
    # Line detection
    detected_lines, thickness_clusters = detect_lines(
        binary_df, binary_image, original_df, original_image,
        dx, dt, 0.01, 14, 0.6, output_dir,
        enable_thickness_clustering=True, distance_threshold=0.8
    )

    # Velocity analysis
    velocity_profiles = analyze_velocity_over_time(
        df_raw, binary_image, detected_lines, dx, dt, output_dir
    )
    
    print(f"\nCOMPLETE: {len(detected_lines)} lines, {len(velocity_profiles)} profiles")
    return detected_lines, thickness_clusters, velocity_profiles


def main():
    print(f"\nDAS MOVING OBJECT DETECTION - {len(SEGMENTS)} segments\n")
    
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    all_results = []
    
    for i, segment_info in enumerate(SEGMENTS, 1):
        print(f"\n{'#'*70}")
        print(f"SEGMENT {i}/{len(SEGMENTS)}: {segment_info['name']}")
        print(f"{'#'*70}")
        
        lines, clusters, profiles = analyze_segment(
            segment_info, DATA_PATH, DX, DT, FS, BASE_OUTPUT_DIR
        )
        
        all_results.append({
            'segment': segment_info['name'],
            'lines': lines,
            'clusters': clusters,
            'velocity_profiles': profiles
        })
    
    print(f"\n{'='*70}\nPIPELINE COMPLETE\n{'='*70}")
    for r in all_results:
        print(f"{r['segment']}: {len(r['lines'])} lines, {len(r['velocity_profiles'])} profiles")
    
    return all_results


if __name__ == "__main__":
    results = main()