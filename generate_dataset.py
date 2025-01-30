"""
Script to generate depth maps from images and keypoints.

Processes images and corresponding JSON keypoints to:
1. Create a mask from keypoints.
2. Generate a depth map using a pre-trained model.
3. Crop images, masks, and depth maps based on mask boundaries.
4. Save cropped images and depth maps to output directories.
"""

import json
from pathlib import Path

import numpy as np
from keypoints_extractor import create_fence_mask_from_json

from transformers import pipeline
from PIL import Image

# Load depth estimation model
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

# Define input and output directories
kp_data = Path("kp_dataset")  # Directory containing images and JSON keypoints
depth_data = Path("depth_dataset")  # Directory to save outputs
depths_dir = depth_data / "depths"  # Directory for cropped depth maps
images_dir = depth_data / "images"  # Directory for cropped images

if __name__ == "__main__":
    # Create output directories if they don't exist
    depths_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Process each JSON keypoint file
    for json_file in kp_data.glob("*.json"):
        with open(json_file, "r") as file:
            # Load keypoints and create mask
            data = json.load(file)
            mask = create_fence_mask_from_json(data).squeeze()

            # Find crop boundaries based on mask
            non_zero = mask > 0
            if np.any(non_zero):
                cols = np.where(non_zero)[1]  # Columns with non-zero values
                left, right = cols.min(), cols.max()  # Left and right boundaries
            else:
                left, right = 0, mask.shape[1] - 1  # Default to full width if no mask

            # Expand boundaries by 150 pixels on both sides
            new_left = max(0, left - 150)
            new_right = right + 150 + 1  # +1 to include the right boundary
            crop_right = min(
                new_right, mask.shape[1]
            )  # Ensure crop is within image bounds

            # Crop mask
            cropped_mask = mask[:, new_left:crop_right]

            # Load corresponding image and generate depth map
            img_path = json_file.with_suffix(".jpg")
            image = Image.open(img_path)
            depth = np.array(pipe(image)["depth"])  # Convert depth map to numpy array

            # Crop depth map and image
            cropped_depth = depth[:, new_left:crop_right]
            crop_box = (new_left, 0, crop_right, image.height)  # Define crop region
            cropped_image = image.crop(crop_box)

            # Set depth to 150 where mask is non-zero
            cropped_depth[cropped_mask > 0] = 150

            # Save cropped image and depth map
            cropped_image.save(images_dir / img_path.name)
            depth_img = Image.fromarray(cropped_depth.astype(np.uint8))
            depth_img.save(depths_dir / f"{img_path.name}")
