import os
import sys

if len(sys.argv) != 2:
    print("Usage: python remove_dot_underscore.py <directory>")
    sys.exit(1)

dir_path = sys.argv[1]

if not os.path.isdir(dir_path):
    print(f"{dir_path} is not a valid directory.")
    sys.exit(1)

removed = 0
for filename in os.listdir(dir_path):
    if filename.startswith("._"):
        file_path = os.path.join(dir_path, filename)
        try:
            os.remove(file_path)
            print(f"Removed: {file_path}")
            removed += 1
        except Exception as e:
            print(f"Failed to remove {file_path}: {e}")

print(f"Total files removed: {removed}")
