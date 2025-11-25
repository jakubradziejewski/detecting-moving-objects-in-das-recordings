import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_das_segment
from preprocessing import preprocess_pipeline
from analysis import (
    analyze_statistics,
    analyze_frequency_content,
    analyze_frequency_bands,
    visualize_filtered_comparison
)
from line_detection import detect_lines

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
    {'start': '094022', 'end': '094212', 'name': 'segment_3'}
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
    
    print("\n" + "=" * 70)
    print(f" ANALYZING {segment_name.upper()}: {start_time} - {end_time}")
    print("=" * 70 + "\n")

    print("=" * 70)
    print("STEP 1: Loading DAS Data")
    print("=" * 70)
    print(f"Loading segment ({start_time} to {end_time})...")
    
    data, df = load_das_segment(
        start_time=start_time,
        end_time=end_time,
        data_path=data_path,
        dx=dx,
        dt=dt,
        verbose=True
    )
    
    print(f"✓ Loaded: {df.shape[0]} time samples × {df.shape[1]} channels")
    print(f"  Duration: {df.shape[0] * dt:.1f} seconds")
    print(f"  Spatial extent: {df.shape[1] * dx:.1f} meters")
    
    print("\n" + "=" * 70)
    print("STEP 2: Statistical Analysis")
    print("=" * 70)
    
    stats = analyze_statistics(
        df=df,
        dt=dt,
        dx=dx,
        segment_name=f"{start_time}_{end_time}",
        output_dir=output_dir
    )

    print("\n" + "=" * 70)
    print("STEP 3: Frequency Analysis")
    print("=" * 70)
    
    # 3A: Frequency content (FFT visualization)
    print("\n[3A] Analyzing frequency content (FFT)...")
    freq_analysis = analyze_frequency_content(df, fs=fs, output_dir=output_dir)
    
    # 3B: Compare frequency bands
    print("\n[3B] Comparing frequency bands...")
    band_results = analyze_frequency_bands(df, fs=fs, output_dir=output_dir)
    
    # 3C: Visual comparison of filtered data
    print("\n[3C] Creating filtered comparisons...")
    visualize_filtered_comparison(df, fs=fs, output_dir=output_dir)
    
    print("\n" + "=" * 70)
    print("STEP 4: Preprocessing (Filtering & Enhancement)")
    print("=" * 70)
    
    print("\nApplying preprocessing pipeline...")
    df_processed = preprocess_pipeline(df, dt=dt, output_dir=output_dir, show_steps=True)
    
    print(f"✓ Preprocessing complete")

    print("\n" + "=" * 70)
    print("STEP 5: Line Detection (Hough Transform)")
    print("=" * 70)
    
    print("\nDetecting vehicle tracks...")
    image = df_processed.values.copy()
    detect_lines(
        image, 
        vertical_factor=0.01, 
        horizontal_factor=10.0, 
        threshold=10, 
        line_length=50,
        line_gap=10,
        sigma=1.0,
        output_dir=output_dir
    )

    print("\n" + "=" * 70)
    print(f"{segment_name.upper()} ANALYSIS COMPLETE ✓")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    
    return {
        'segment_name': segment_name,
        'output_dir': output_dir,
        'stats': stats,
        'freq_analysis': freq_analysis,
        'band_results': band_results,
        'df_processed': df_processed
    }

def main():
    print("\n" + "=" * 70)
    print(" DAS MOVING OBJECT DETECTION - MULTI-SEGMENT ANALYSIS")
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
        print(f"\n{'#' * 70}")
        print(f"# SEGMENT {i}/{len(SEGMENTS)}")
        print(f"{'#' * 70}")
        
        result = analyze_segment(
            segment_info=segment_info,
            data_path=DATA_PATH,
            dx=DX,
            dt=DT,
            fs=FS,
            base_output_dir=BASE_OUTPUT_DIR
        )
        results.append(result)
    
    print("\n" + "=" * 70)
    print(" ALL SEGMENTS ANALYZED SUCCESSFULLY ✓")
    print("=" * 70)
    print(f"\nProcessed {len(SEGMENTS)} segments:")
    for result in results:
        print(f"  • {result['segment_name']}: {result['output_dir']}")
    
    
    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETE")
    print("=" * 70 + "\n")
    
    return results

if __name__ == "__main__":
    results = main()