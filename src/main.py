import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_das_segment
from preprocessing import preprocess_pipeline
from data_analysis import analyze_and_visualize_segment

DATA_PATH = '../data/'
OUTPUT_DIR = '../output/preprocessed'

START_TIME = '090522'
END_TIME = '090712'
DATE = '20240507'

DX = 5.106500953873407
DT = 0.0016

print("\n" + "=" * 70)
print(" DAS MOVING OBJECT DETECTION - PREPROCESSING")
print("=" * 70 + "\n")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Loading 2-minute segment...")
data, df = load_das_segment(
    start_time=START_TIME,
    end_time=END_TIME,
    data_path=DATA_PATH,
    dx=DX,
    dt=DT,
    verbose=True
)
print("STEP 3: Performing analysis and generating visualizations")
print("-" * 70)

segment_name = f"{START_TIME}_{END_TIME}"

# Run complete analysis with visualization
analyze_and_visualize_segment(
    df=df,
    dt=DT,
    dx=DX,
    segment_name=segment_name,
    output_dir=OUTPUT_DIR
)

print("\nStarting preprocessing pipeline...")
df_processed = preprocess_pipeline(df, dt=DT, show_steps=True)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)