import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageFile

# Allow loading truncated images for testing if they can be partially loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

def check_corrupt_images(folder_path):
    """
    Check for corrupt images in the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing images
        
    Returns:
        list: List of corrupt image file paths
    """
    print(f"Checking images in {folder_path}...")
    corrupt_images = []
    truncated_images = []
    total_images = 0
    
    # Get all files in the directory
    try:
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    except Exception as e:
        print(f"Error accessing folder: {e}")
        return corrupt_images
    
    # Check each file
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif','webp')):
            file_path = os.path.join(folder_path, file)
            total_images += 1
            
            # Try multiple methods to open the image
            try:
                # Method 1: OpenCV
                img = cv2.imread(file_path)
                if img is None:
                    raise Exception("OpenCV couldn't load the image")
                
                # Additional check: Make sure image has valid dimensions and data
                if img.size == 0 or len(img.shape) < 2:
                    raise Exception("Invalid image dimensions")
                
            except Exception as cv_error:
                try:
                    # Method 2: PIL/Pillow with special handling for truncated images
                    try:
                        # First attempt with normal loading
                        with Image.open(file_path) as img:
                            img.verify()
                            with Image.open(file_path) as img2:
                                img2.load()
                    except OSError as truncated_error:
                        # Check specifically for truncated image error
                        if "truncated" in str(truncated_error).lower():
                            print(f"Truncated image found: {file_path}")
                            print(f"  Error: {truncated_error}")
                            corrupt_images.append(file_path)
                            truncated_images.append(file_path)
                            continue
                        else:
                            # Re-raise if it's not a truncation error
                            raise
                except Exception as pil_error:
                    # If both methods fail, consider the image corrupt
                    corrupt_images.append(file_path)
                    print(f"Corrupt image found: {file_path}")
                    print(f"  OpenCV error: {cv_error}")
                    print(f"  PIL error: {pil_error}")
            
            # Print progress every 100 images
            if total_images % 100 == 0:
                print(f"Processed {total_images} images so far...")
    
    # Print summary
    print(f"Total images checked: {total_images}")
    print(f"Total corrupt images found: {len(corrupt_images)}")
    print(f"Total truncated images found: {len(truncated_images)}")
    
    return corrupt_images, truncated_images

def main():
    # Set the path to the folder containing fake images
    fake_images_folder = os.path.join('datasets', 'test', '1_fake')
    
    # Get absolute path
    folder_path = os.path.abspath(fake_images_folder)
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return
    
    # Allow user to try loading truncated images
    try_fix_truncated = input("Would you like to attempt to load truncated images? (y/n): ").lower() == 'y'
    if try_fix_truncated:
        print("Will attempt to load truncated images using PIL's LOAD_TRUNCATED_IMAGES option")
        ImageFile.LOAD_TRUNCATED_IMAGES = True
    else:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
    
    # Check for corrupt images
    corrupt_images, truncated_images = check_corrupt_images(folder_path)
    
    # Write corrupt image paths to a file
    if corrupt_images:
        output_file = "corrupt_images_list.txt"
        with open(output_file, 'w') as f:
            f.write("=== CORRUPT IMAGES ===\n")
            for img_path in corrupt_images:
                f.write(f"{img_path}\n")
            
            f.write("\n=== TRUNCATED IMAGES ===\n")
            for img_path in truncated_images:
                f.write(f"{img_path}\n")
                
        print(f"List of corrupt images saved to {output_file}")
    else:
        print("No corrupt images found.")

if __name__ == "__main__":
    main()
