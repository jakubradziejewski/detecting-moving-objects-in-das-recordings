import sys
import os

# Add src to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DASDataLoader, get_available_files
from data_analysis import analyze_and_visualize_segment
from download_data import download_data

# Paths
DATA_PATH = '../data/'
OUTPUT_DIR = '../output/day1_2_analysis'

# Your assigned time segment
START_TIME = '090522'  # HHMMSS format
END_TIME = '090712'  # 2 minutes later
DATE = '20240507'

# Metadata
DX = 5.106500953873407  # Spatial resolution [m]
DT = 0.0016  # Temporal resolution [s]


print("\n" + "=" * 70)
print(" DAS MOVING OBJECT DETECTION - DAYS 1 & 2")
print(" Data Loading and Comprehensive Analysis")
print("=" * 70 + "\n")

# Create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"✓ Created output directory: {OUTPUT_DIR}\n")


print("STEP 1: Checking data availability")
print("-" * 70)

available_files = get_available_files(DATA_PATH)

if len(available_files) == 0:
    print("⚠ No data files found. Starting download...\n")
    success = download_data(data_path=DATA_PATH)

    print("\n✓ Data downloaded successfully!\n")
else:
    print(f"✓ Found {len(available_files)} data files\n")


print("STEP 2: Loading your assigned 2-minute segment")
print("-" * 70)

loader = DASDataLoader(data_path=DATA_PATH, dx=DX, dt=DT)


data, df, metadata = loader.load_and_prepare(
        start_time=START_TIME,
        end_time=END_TIME,
        date=DATE,
        verbose=True
    )


print("✓ Data loaded successfully!\n")

print("STEP 3: Performing analysis and generating visualizations")
print("-" * 70)

segment_name = f"{START_TIME}_{END_TIME}"

# Run complete analysis with visualization
analyze_and_visualize_segment(
    df=df,
    dt=DT,
    dx=DX,
    segment_name=segment_name,
    output_dir=OUTPUT_DIR,
    verbose=True
)
