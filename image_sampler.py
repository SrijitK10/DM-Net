import os
import shutil

# Paths
input_folder = "./images1024x1024"  # Change this to your actual input folder
output_base_folder = "./datasets/real"  # Change this to your actual output folder

# Ensure output folders exist
os.makedirs(output_base_folder, exist_ok=True)

# Get a sorted list of images
image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")], key=lambda x: int(x.split('.')[0]))

# Folder and distribution settings
folder_limits = [39000, 13000, 13000]  # First 3000, then 1000, then 1000
folder_index = 1
count = 0
processed_images = 0

# Create first folder
current_folder = os.path.join(output_base_folder, f"folder_{folder_index}")
os.makedirs(current_folder, exist_ok=True)

# Process images
for i, img_name in enumerate(image_files):
    src_path = os.path.join(input_folder, img_name)
    dest_path = os.path.join(current_folder, f"ffhq{count}.png")

    # Copy and rename image
    shutil.copy(src_path, dest_path)

    count += 1
    processed_images += 1

    # If we reach the limit, move to the next folder
    if count == folder_limits[folder_index - 1]:
        folder_index += 1
        if folder_index <= len(folder_limits):  # Ensure more folders are needed
            current_folder = os.path.join(output_base_folder, f"folder_{folder_index}")
            os.makedirs(current_folder, exist_ok=True)
            count = 0  # Reset count for new folder
        else:
            break  # Stop if all defined folders are created

print(f"Successfully copied and renamed {processed_images} images.")
