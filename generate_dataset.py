"""
Script generates dataset from image and keypoints to create depth map.
"""

import json
from pathlib import Path

import numpy as np
from keypoints_extractor import create_fence_mask_from_json

from transformers import pipeline
from PIL import Image

# Pipe for using depth-estimation model
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")


# Create output directories if they don't exist
kp_data = Path("kp_dataset")
depth_data = Path("depth_dataset")
depths_dir = depth_data / "depths"
images_dir = depth_data / "images"

if __name__ == "__main__":

    for json_file in kp_data.glob("*.json"):
        with open(json_file, "r") as file:
            data = json.load(file)
            mask = create_fence_mask_from_json(data, json_file).squeeze()

            # Determine crop boundaries
            non_zero = mask > 0
            if np.any(non_zero):
                cols = np.where(non_zero)[1]
                left, right = cols.min(), cols.max()
            else:
                left, right = 0, mask.shape[1] - 1

            new_left = max(0, left - 150)
            new_right = right + 150 + 1  # +1 to include right+200 pixel
            crop_right = min(new_right, mask.shape[1])

            # Crop mask
            cropped_mask = mask[:, new_left:crop_right]

            # Load image and process depth
            img_path = json_file.with_suffix(".jpg")
            image = Image.open(img_path)
            depth = np.array(pipe(image)["depth"])

            # Crop depth and image
            cropped_depth = depth[:, new_left:crop_right]
            crop_box = (new_left, 0, crop_right, image.height)
            cropped_image = image.crop(crop_box)

            # Apply mask to depth
            cropped_depth[cropped_mask > 0] = 150

            # Save results
            cropped_image.save(images_dir / img_path.name)
            depth_img = Image.fromarray(cropped_depth.astype(np.uint8))
            depth_img.save(depths_dir / f"{img_path.name}")
