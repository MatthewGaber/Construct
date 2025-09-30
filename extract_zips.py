import os
import subprocess


def extract_zip_using_unzip(zip_path, extract_path):
    try:
        subprocess.run(['unzip', '-o', zip_path, '-d', extract_path], check=True)
        print(f"Extracted: {zip_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to extract {zip_path}: {e}")

def extract_and_delete_zip_files(folder_path):
    files = os.listdir(folder_path)
    
    for file_name in files:
        if file_name.lower().endswith('.zip'):
            zip_path = os.path.join(folder_path, file_name)
            
            # Extract using `unzip`
            extract_zip_using_unzip(zip_path, folder_path)
            
            # Delete the ZIP file
            #try:
                #os.remove(zip_path)
                #print(f"Deleted: {file_name}")
            #except Exception as e:
                #print(f"Failed to delete {file_name}: {e}")

# Example usage
folder_path = '/Users/matthew/Desktop/zippy'
extract_and_delete_zip_files(folder_path)

"""

from datetime import datetime, timedelta

def convert_to_datetime(timestamp):
    # Convert from 100-nanosecond intervals to seconds
    epoch_offset = timedelta(seconds=11644473600)  # Offset from 1601 to 1970
    seconds = timestamp / 10**7
    unix_time = datetime(1970, 1, 1) + timedelta(seconds=seconds) - epoch_offset
    return unix_time

# Timestamps
timestamp1 = 133540976766122461
timestamp2 = 133540986242754656

# Convert and print
dt1 = convert_to_datetime(timestamp1)
dt2 = convert_to_datetime(timestamp2)

print("Timestamp 1:", dt1)
print("Timestamp 2:", dt2)
"""