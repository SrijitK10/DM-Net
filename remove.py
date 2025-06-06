#!/usr/bin/env python3
import os
import argparse

def remove_dot_underscore_files(folder_path):
    """
    Remove all files starting with '._' in the specified folder.
    
    Args:
        folder_path (str): Path to the folder to clean
    
    Returns:
        int: Number of files removed
    """
    count = 0
    
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory")
        return count
    
    print(f"Scanning '{folder_path}' for files starting with '._'...")
    
    # Walk through the directory
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.startswith('._'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                    count += 1
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
    
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove files starting with '._' in a folder")
    parser.add_argument("folder_path", help="Path to the folder to clean")
    parser.add_argument("-r", "--recursive", action="store_true", 
                        help="Process subdirectories recursively")
    
    args = parser.parse_args()
    
    total_removed = remove_dot_underscore_files(args.folder_path)
    
    print(f"\nTotal files removed: {total_removed}")
    print("Done!")