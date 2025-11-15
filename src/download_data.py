import os
import glob
import requests
import patoolib


def check_data_exists(data_path='../data/'):
    """Check if .npy files exist in data directory"""
    if not os.path.exists(data_path):
        return False
    npy_files = glob.glob(os.path.join(data_path, '*.npy'))
    return len(npy_files) > 0


def download_data(data_path='../data/'):
    """
    Download and extract DAS data from Google Drive if not already present.
    """
    # Create data directory
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Check if already exists
    if check_data_exists(data_path):
        npy_files = glob.glob(os.path.join(data_path, '*.npy'))
        print(f"✓ Data already exists! Found {len(npy_files)} .npy files")
        return True

    print("\n" + "="*70)
    print(" DOWNLOADING DAS DATA")
    print("="*70)

    # Download URL
    url = "https://drive.usercontent.google.com/download?id=1lJKLz3LsQmnAf9q5GGi6arEYBz-3CMhx&export=download&authuser=0&confirm=t"
    archive_path = os.path.join(data_path, "data.rar")

    # Download file
    print("\nDownloading data...")
    try:
        response = requests.get(url, stream=True, allow_redirects=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(archive_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress = (downloaded / total_size) * 100
                    print(f"\rProgress: {progress:.1f}%", end='')

        print(f"\n✓ Downloaded: {os.path.getsize(archive_path) / (1024*1024):.1f} MB")

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False

    # Extract RAR file
    print("\nExtracting archive")
    try:
        patoolib.extract_archive(archive_path, outdir=data_path)
        print("✓ Extraction complete!")

    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False

    os.remove(archive_path)
    print("✓ Cleaned up rar file")

    npy_files = glob.glob(os.path.join(data_path, '*.npy'))
    print("\n" + "="*70)
    print(f"✓ SUCCESS! Found {len(npy_files)} .npy files")
    print("="*70 + "\n")

    return len(npy_files) > 0
