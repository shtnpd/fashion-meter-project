import os
import sys
from PIL import Image
from tqdm import tqdm
import numpy as np

def optimize_images():
    # Get command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python script.py <folder_path> [width,height]")
        return

    folder_path = sys.argv[1]
    custom_resolution_input = sys.argv[2] if len(sys.argv) > 2 else None
    output_folder = f"{folder_path}-rescaled"
    os.makedirs(output_folder, exist_ok=True)

    resolutions = []
    aspect_ratios = []

    # Iterate through all files in the folder
    for file_name in tqdm(os.listdir(folder_path), desc="Processing images"):
        file_path = os.path.join(folder_path, file_name)
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                resolutions.append((width, height))
                aspect_ratios.append(width / height)
        except Exception as e:
            print(f"Could not process file {file_name}: {e}")

    if custom_resolution_input:
        try:
            custom_width, custom_height = map(int, custom_resolution_input.split(","))
            custom_resolution = (custom_width, custom_height)
        except ValueError:
            print("Invalid resolution input. Please use the format width,height.")
            return
    else:
        # Calculate optimal aspect ratio (median of all aspect ratios)
        optimal_aspect_ratio = np.median(aspect_ratios)

        # Find the smallest resolution larger than 500x500 that matches the optimal aspect ratio
        min_resolution = 500
        optimal_width = max(min_resolution, int(min_resolution * optimal_aspect_ratio))
        optimal_height = max(min_resolution, int(min_resolution / optimal_aspect_ratio))
        custom_resolution = (optimal_width, optimal_height)

        print(f"Optimal Aspect Ratio: {optimal_aspect_ratio:.2f}")
        print(f"Optimal Resolution: {custom_resolution}")

    # Resize images to the custom resolution
    for file_name in tqdm(os.listdir(folder_path), desc="Resizing images"):
        file_path = os.path.join(folder_path, file_name)
        output_path = os.path.join(output_folder, file_name)
        try:
            with Image.open(file_path) as img:
                resized_img = img.resize(custom_resolution, Image.Resampling.LANCZOS)
                resized_img.save(output_path)
        except Exception as e:
            print(f"Could not resize file {file_name}: {e}")

    print(f"All optimized images have been saved to: {output_folder}")

# Execute the function
if __name__ == "__main__":
    optimize_images()
