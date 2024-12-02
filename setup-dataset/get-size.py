import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

def analyze_images_with_thresholds():
    folder_path = input("Enter the folder path containing images: ")

    resolutions = []
    aspect_ratios = []



    # Iterate through all files in the folder with tqdm
    for file_name in tqdm(os.listdir(folder_path), desc="Processing images"):
        file_path = os.path.join(folder_path, file_name)
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                resolutions.append((file_name, width, height))
                aspect_ratios.append((file_name, width / height))
        except Exception as e:
            print(f"Could not process file {file_name}: {e}")

    # Calculate statistics
    if resolutions and aspect_ratios:
        # Extract width and height information
        widths = [res[1] for res in resolutions]
        heights = [res[2] for res in resolutions]
        aspect_ratios_only = [ar[1] for ar in aspect_ratios]

        mean_resolution = (np.mean(widths), np.mean(heights))
        median_resolution = (np.median(widths), np.median(heights))
        mean_aspect_ratio = np.mean(aspect_ratios_only)
        median_aspect_ratio = np.median(aspect_ratios_only)

        # Find images with max and min width and height
        max_width_image = max(resolutions, key=lambda x: x[1])
        min_width_image = min(resolutions, key=lambda x: x[1])
        max_height_image = max(resolutions, key=lambda x: x[2])
        min_height_image = min(resolutions, key=lambda x: x[2])

        # Find max and min aspect ratios
        max_aspect_ratio_image = max(aspect_ratios, key=lambda x: x[1])
        min_aspect_ratio_image = min(aspect_ratios, key=lambda x: x[1])

        # Helper to check closeness
        def close_to(value, target, threshold):
            return abs(value - target) / target <= threshold

        # Generate results for different thresholds
        thresholds = [0.05, 0.1, 0.2]
        total_images = len(resolutions)

        results_table = []

        for threshold in thresholds:
            close_to_mean_res = sum(
                close_to(w, mean_resolution[0], threshold)
                and close_to(h, mean_resolution[1], threshold)
                for _, w, h in resolutions
            )
            close_to_median_res = sum(
                close_to(w, median_resolution[0], threshold)
                and close_to(h, median_resolution[1], threshold)
                for _, w, h in resolutions
            )
            close_to_mean_aspect = sum(
                close_to(ar, mean_aspect_ratio, threshold) for _, ar in aspect_ratios
            )
            close_to_median_aspect = sum(
                close_to(ar, median_aspect_ratio, threshold) for _, ar in aspect_ratios
            )

            results_table.append([
                f"{int(threshold * 100)}%",
                f"{close_to_mean_res}/{total_images}",
                f"{close_to_median_res}/{total_images}",
                f"{close_to_mean_aspect}/{total_images}",
                f"{close_to_median_aspect}/{total_images}",
            ])

        # Print statistics
        print(f"\nMean Resolution: {mean_resolution}")
        print(f"Median Resolution: {median_resolution}")
        print(f"Mean Aspect Ratio: {mean_aspect_ratio:.2f}")
        print(f"Median Aspect Ratio: {median_aspect_ratio:.2f}")

        print(f"\nMax Width: {max_width_image[1]} ({max_width_image[0]})")
        print(f"Min Width: {min_width_image[1]} ({min_width_image[0]})")
        print(f"Max Height: {max_height_image[2]} ({max_height_image[0]})")
        print(f"Min Height: {min_height_image[2]} ({min_height_image[0]})")
        print(f"Max Aspect Ratio: {max_aspect_ratio_image[1]:.2f} ({max_aspect_ratio_image[0]})")
        print(f"Min Aspect Ratio: {min_aspect_ratio_image[1]:.2f} ({min_aspect_ratio_image[0]})")

        print("\nStatistics:")
        print(tabulate(
            results_table,
            headers=["Threshold", "Mean Res Close", "Median Res Close", "Mean AR Close", "Median AR Close"],
            tablefmt="grid"
        ))

# Execute the function
analyze_images_with_thresholds()
