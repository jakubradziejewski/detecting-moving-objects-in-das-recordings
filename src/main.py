import sys
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_das_segment
from preprocessing import preprocess_das_data
from line_detection import detect_lines
from analysis import analyze_statistics, analyze_frequency_content, visualize_filtered_comparison

# Configuration
DATA_PATH = '../data'
BASE_OUTPUT_DIR = '../output'
DATE = '20240507'
DX = 5.106500953873407
DT = 0.0016
FS = 625

# Define segments to analyze
SEGMENTS = [
    {'start': '090522', 'end': '090712', 'name': 'segment_1'},
    {'start': '093252', 'end': '093442', 'name': 'segment_2'},
    {'start': '092112', 'end': '092302', 'name': 'segment_3'}
]


def analyze_segment(segment_info, data_path, dx, dt, fs, base_output_dir):
    """
    Complete analysis pipeline for a single segment.
    
    Pipeline:
    1. Load raw DAS data
    2. Preprocess data (centering, thresholding, morphology)
    3. Detect lines (Hough transform, velocity calculation, clustering)
    """
    start_time = segment_info['start']
    end_time = segment_info['end']
    segment_name = segment_info['name']
    
    # Create segment-specific output directory
    output_dir = os.path.join(base_output_dir, segment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("=" * 70)
    print("STEP 1: Loading DAS Data")
    print("=" * 70)
    print(f"Segment: {segment_name} ({start_time} to {end_time})")
    
    df_raw = load_das_segment(
        start_time=start_time,
        end_time=end_time,
        data_path=data_path,
        dx=dx,
        dt=dt,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("STEP 2: Statistical Analysis")
    print("=" * 70)
    
    analyze_statistics(
        df=df_raw,
        dt=dt,
        dx=dx,
        segment_name=f"{start_time}_{end_time}",
        output_dir=output_dir
    )
    analyze_frequency_content(df_raw, fs=fs, output_dir=output_dir)
    visualize_filtered_comparison(df_raw, fs=fs, output_dir=output_dir)
    print("\n" + "=" * 70)
    print("STEP 3: Preprocessing")
    print("=" * 70)
    
    # Preprocess data: centering, absolute value, resizing, thresholding, morphology
    binary_df, binary_image, original_df, original_image = preprocess_das_data(
        df_raw=df_raw,
        target_size=750,
        threshold_multiplier=0.8,
        output_dir=output_dir
    )
    
    print("\n" + "=" * 70)
    print("STEP 4: Line Detection")
    print("=" * 70)
    
    # Detect lines: Hough transform, velocity calculation, clustering
    detected_lines = detect_lines(
        binary_df=binary_df,
        binary_image=binary_image,
        original_df=original_df,
        original_image=original_image,
        dx=dx,
        dt=dt,
        vertical_factor=0.01,
        horizontal_factor=14,
        threshold_ratio=0.6,
        output_dir=output_dir
    )
    
    print(f"\n{'='*70}")
    print(f"SEGMENT {segment_name} COMPLETE: {len(detected_lines)} lines detected")
    print(f"{'='*70}\n")
    
    return detected_lines


def main():
    print("\n" + "=" * 70)
    print(" DAS MOVING OBJECT DETECTION")
    print("=" * 70)
    print(f"\nAnalyzing {len(SEGMENTS)} segments:")
    for i, seg in enumerate(SEGMENTS, 1):
        print(f"  {i}. {seg['name']}: {seg['start']} - {seg['end']}")
    print("=" * 70 + "\n")
    
    # Create base output directory
    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)
    
    # Analyze each segment
    all_results = []
    for i, segment_info in enumerate(SEGMENTS, 1):
        print(f"\n{'#'*70}")
        print(f"# PROCESSING SEGMENT {i}/{len(SEGMENTS)}: {segment_info['name']}")
        print(f"{'#'*70}\n")
        
        detected_lines = analyze_segment(
            segment_info=segment_info,
            data_path=DATA_PATH,
            dx=DX,
            dt=DT,
            fs=FS,
            base_output_dir=BASE_OUTPUT_DIR
        )
        
        all_results.append({
            'segment': segment_info['name'],
            'lines': detected_lines
        })
    
    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETE")
    print("=" * 70)
    print("\nSummary:")
    for result in all_results:
        print(f"  {result['segment']}: {len(result['lines'])} lines detected")
    print("=" * 70 + "\n")
    
    return all_results


if __name__ == "__main__":
    results = main()