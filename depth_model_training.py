from transformers import AutoModelForDepthEstimation, AutoImageProcessor
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os
import matplotlib.pyplot as plt  # Added for visualization
import numpy as np  # Added for array operations

# Hyperparameters and configurations
IMAGE_DIR = Path("depth_dataset/images")
DEPTH_DIR = Path("depth_dataset/depths")
MODEL_PATH = "saved_models/depth_model_v2"
HF_MODEL = "LiheYoung/depth-anything-base-hf"
SAVE_MODEL_PATH = "saved_models"
SAVE_MODEL_NAME = "depth_model_v2"
PRETRAINED = True
BATCH_SIZE = 1
LEARNING_RATE = 3e-5
NUM_EPOCHS = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results"  # Added visualization directory

# Ensure directories exist
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)  # Create results directory


# Dataset class (unchanged)
class DepthDataset(Dataset):
    def __init__(self, image_dir, depth_dir, processor):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.processor = processor
        self.image_files = sorted(image_dir.glob("*"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_dir / f"{image_path.name}"

        image = Image.open(image_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        depth = depth.resize(pixel_values.shape[-2:][::-1])
        depth = transforms.ToTensor()(depth)

        return {"pixel_values": pixel_values, "labels": depth}


# Modified to include shuffle parameter
def create_dataset(image_dir, depth_dir, processor, batch_size, shuffle=True):
    dataset = DepthDataset(image_dir, depth_dir, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# Load model function (unchanged)
def load_model(pretrained=True):
    if pretrained:
        print(f"Loading model from local path: {MODEL_PATH}")
        model = AutoModelForDepthEstimation.from_pretrained(MODEL_PATH)
    else:
        print(f"Loading model from Hugging Face model hub: {HF_MODEL}")
        model = AutoModelForDepthEstimation.from_pretrained(HF_MODEL)
    model.to(DEVICE)
    return model


# Save model function (unchanged)
def save_model(model, save_path, save_model_name):
    model_save_path = os.path.join(save_path, save_model_name)
    model.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")


# Training function (unchanged)
def train(
    model, train_loader, num_epochs, learning_rate, save_model_path, save_model_name
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} started on device {DEVICE} ...")
        model.train()
        total_loss = 0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values)
            predicted_depth = outputs.predicted_depth

            if predicted_depth.ndim == 3:
                predicted_depth = predicted_depth.unsqueeze(1)

            loss = torch.nn.functional.mse_loss(predicted_depth, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, save_model_path, save_model_name)

    print("Training completed!")


# Added visualization function
def visualize_results(model, dataloader, processor, save_dir, save_model_name):
    model.eval()
    for batch_idx, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(DEVICE)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        # Denormalize image
        mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1).to(DEVICE)
        std = torch.tensor(processor.image_std).view(1, 3, 1, 1).to(DEVICE)
        denorm_img = pixel_values * std + mean
        denorm_img = denorm_img.squeeze().cpu().numpy().transpose(1, 2, 0)
        denorm_img = (denorm_img * 255).astype(np.uint8)

        # Process depth prediction
        depth = outputs.predicted_depth.squeeze().cpu().numpy()
        depth_normalized = (
            (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255
        )

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(denorm_img)
        ax1.set_title("Original Image")
        ax1.axis("off")

        ax2.imshow(depth_normalized, cmap="viridis")
        ax2.set_title("Predicted Depth")
        ax2.axis("off")

        # Save plot
        img_name = dataloader.dataset.image_files[batch_idx].stem
        save_path = Path(save_dir) / save_model_name / f"{img_name}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), bbox_inches="tight")
        plt.close()


# Main execution block with visualization
if __name__ == "__main__":
    processor = AutoImageProcessor.from_pretrained(HF_MODEL, use_fast=True)
    train_loader = create_dataset(IMAGE_DIR, DEPTH_DIR, processor, BATCH_SIZE)
    model = load_model(PRETRAINED)

    train(
        model, train_loader, NUM_EPOCHS, LEARNING_RATE, SAVE_MODEL_PATH, SAVE_MODEL_NAME
    )

    # Load best model for visualization
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_PATH).to(DEVICE)

    # Create visualization dataloader without shuffling
    vis_loader = create_dataset(
        IMAGE_DIR, DEPTH_DIR, processor, BATCH_SIZE, shuffle=False
    )

    # Generate and save visualizations
    visualize_results(model, vis_loader, processor, RESULTS_DIR, SAVE_MODEL_NAME)
    print(f"Visualizations saved to {RESULTS_DIR} directory")
