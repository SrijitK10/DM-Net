import os
import csv
import random

# Directories containing your images
directory_real = '/Users/srijit/Documents/Projects Personal/FREQUENCY/SyntheticImagesAnalysis/powerspectra/test/0_real'
directory_fake = '/Users/srijit/Documents/Projects Personal/FREQUENCY/SyntheticImagesAnalysis/powerspectra/test/1_fake'

# Get a list of all file paths in the directory and its subdirectories
def get_file_paths(directory, label):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append((os.path.join(root, file), label))
    return file_paths

# Get file paths for real and fake images
real_file_paths = get_file_paths(directory_real, '0')  # '0' is the label for real images
fake_file_paths = get_file_paths(directory_fake, '1')  # '1' is the label for fake images

# Combine real and fake image paths
merged_data = real_file_paths + fake_file_paths

# Path to the merged CSV file
merged_csv_file = 'test.csv'
random.shuffle(merged_data)

# Write the combined data to the new CSV file
with open(merged_csv_file, mode='w', newline='') as merged_file:
    writer = csv.writer(merged_file)
    writer.writerow(['Path', 'Truth'])  # Header row with 'Path' and 'Truth' columns

    for path, label in merged_data:
        writer.writerow([path, label])

# df = df.sample(frac=1, random_state=42)

print(f'File paths with labels written to {merged_csv_file}')
