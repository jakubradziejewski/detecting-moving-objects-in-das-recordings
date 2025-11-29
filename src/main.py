import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_das_segment
from preprocessing import preprocess_pipeline, visualize_das, frequency_filter_fft, resize_to_square, squeeze_dataframe, stretch_dataframe
from matplotlib.colors import Normalize
from analysis import (
    analyze_statistics,
    analyze_frequency_content,
    visualize_filtered_comparison
)
from line_detection import detect_lines

DATA_PATH = '../data'
BASE_OUTPUT_DIR = '../output'
OUTPUT_PATH = '../output'
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
    print(f"Loading segment ({start_time} to {end_time})...")
    
    df_raw = load_das_segment(start_time=start_time, end_time=end_time, data_path=data_path, dx=dx, dt=dt, verbose=True)



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

    print("\n" + "=" * 70)
    print("STEP 3: Frequency Analysis")
    print("=" * 70)

    #analyze_frequency_content(df_raw, fs=fs, output_dir=output_dir)
    #visualize_filtered_comparison(df_raw, fs=fs, output_dir=output_dir)
    df_display = df_raw.copy()
    df_display -= df_display.mean()
    df_abs = np.abs(df_display)
    df_square = resize_to_square(df_abs, target_size=750)

    # Visualize resized square data
    visualize_das(df_square, title="Resized to Square (750x750)", 
              save_path=os.path.join(output_dir, 'resized_square_data.png'))
    
    low_thresh = np.percentile(df_square, 3)
    high_thresh = np.percentile(df_square, 99)

    data_thresholded = df_square.copy()
    data_thresholded[df_square < low_thresh] = 0
    data_thresholded = np.clip(data_thresholded, 0, high_thresh)

    print(f"Threshold range: {low_thresh:.2e} to {high_thresh:.2e}")

    import pandas as pd
    df_thresholded = pd.DataFrame(data=data_thresholded, index=df_square.index, columns=df_square.columns)
    visualize_das(df_thresholded, title="Thresholded Absolute Data", 
                save_path=output_dir, normalize=False)

    image = df_thresholded.values.copy()
    detect_lines(
        df_thresholded,
        image,
        thick_threshold=0.5,thin_threshold=0.6, output_dir=output_dir
    )


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
    results = []
    for i, segment_info in enumerate(SEGMENTS, 1):
        print(f"SEGMENT {i}/{len(SEGMENTS)}")
        
        analyze_segment(
            segment_info=segment_info,
            data_path=DATA_PATH,
            dx=DX,
            dt=DT,
            fs=FS,
            base_output_dir=BASE_OUTPUT_DIR
        )

    
    
    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETE")
    print("=" * 70 + "\n")
    
    return results

if __name__ == "__main__":
    results = main()