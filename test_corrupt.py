import os
import cv2
from PIL import Image
import argparse
import shutil

def check_corrupt_cv2(image_path):
    """
    Check if an image is corrupt using OpenCV.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        bool: True if image is corrupt, False otherwise
    """
    try:
        # Try to read the image
        img = cv2.imread(image_path)
        if img is None:
            return True
        
        # Check if image has valid dimensions
        if img.shape[0] == 0 or img.shape[1] == 0:
            return True
            
        # Try to decode the image completely
        cv2.imwrite('/tmp/test_cv2.jpg', img)
        os.remove('/tmp/test_cv2.jpg')
        
        return False
    except Exception as e:
        print(f"CV2 error for {image_path}: {e}")
        return True

def check_corrupt_pil(image_path):
    """
    Check if an image is corrupt using PIL.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        bool: True if image is corrupt, False otherwise
    """
    try:
        # Try to open the image
        with Image.open(image_path) as img:
            # Verify the image
            img.verify()
            
        # Re-open and try to load the image data
        with Image.open(image_path) as img:
            img.load()
            
        return False
    except Exception as e:
        print(f"PIL error for {image_path}: {e}")
        return True

def check_and_remove_corrupt_images(folder_path, backup_folder=None, dry_run=False):
    """
    Check for corrupt images in a folder and remove them.
    
    Args:
        folder_path: Path to the folder containing images
        backup_folder: Optional path to backup corrupt images before removal
        dry_run: If True, only report corrupt images without removing them
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return
    
    # Create backup folder if specified
    if backup_folder and not dry_run:
        os.makedirs(backup_folder, exist_ok=True)
    
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
    
    corrupt_images = []
    total_images = 0
    
    print(f"Scanning folder: {folder_path}")
    print("=" * 50)
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Skip if not a file
        if not os.path.isfile(file_path):
            continue
            
        # Check if it's an image file
        _, ext = os.path.splitext(filename.lower())
        if ext not in image_extensions:
            continue
            
        total_images += 1
        
        # Check for corruption using both methods
        cv2_corrupt = check_corrupt_cv2(file_path)
        pil_corrupt = check_corrupt_pil(file_path)
        
        # Consider image corrupt if either method fails
        if cv2_corrupt or pil_corrupt:
            corrupt_images.append(file_path)
            print(f"CORRUPT: {filename} (CV2: {cv2_corrupt}, PIL: {pil_corrupt})")
            
            if not dry_run:
                # Backup if requested
                if backup_folder:
                    backup_path = os.path.join(backup_folder, filename)
                    shutil.copy2(file_path, backup_path)
                    print(f"  Backed up to: {backup_path}")
                
                # Remove the corrupt image
                os.remove(file_path)
                print(f"  Removed: {file_path}")
        else:
            print(f"OK: {filename}")
    
    print("=" * 50)
    print(f"Total images scanned: {total_images}")
    print(f"Corrupt images found: {len(corrupt_images)}")
    
    if dry_run:
        print("DRY RUN: No images were actually removed")
    else:
        print(f"Corrupt images removed: {len(corrupt_images)}")
        if backup_folder:
            print(f"Corrupt images backed up to: {backup_folder}")

def main():
    parser = argparse.ArgumentParser(description='Check for corrupt images and remove them')
    parser.add_argument('folder', help='Path to the folder containing images')
    parser.add_argument('--backup', help='Path to backup corrupt images before removal')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Only report corrupt images without removing them')
    
    args = parser.parse_args()
    
    check_and_remove_corrupt_images(args.folder, args.backup, args.dry_run)

if __name__ == "__main__":
    main()