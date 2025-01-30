from transformers import AutoModelForDepthEstimation, AutoImageProcessor
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os

# Hyperparameters and configurations
IMAGE_DIR = Path("depth_dataset/images")
DEPTH_DIR = Path("depth_dataset/depths")
MODEL_PATH = "save_models/depth_model_v1"  # Path to pretrained model or directory
HF_MODEL = "LiheYoung/depth-anything-small-hf"
SAVE_MODEL_PATH = "saved_models"  # Directory to save the best model
SAVE_MODEL_NAME = "depth_model_v1.pt"
PRETRAINED = False  # Set to False to train from scratch
BATCH_SIZE = 1
LEARNING_RATE = 5e-5
NUM_EPOCHS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure the save directory exists
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)


# Dataset class
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

        # Load image and depth
        image = Image.open(image_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")

        # Process image using the processor
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        # Resize depth to match the model's expected input size
        depth = depth.resize(pixel_values.shape[-2:][::-1])
        depth = transforms.ToTensor()(depth)

        return {"pixel_values": pixel_values, "labels": depth}


# Function to create dataset and DataLoader
def create_dataset(image_dir, depth_dir, processor, batch_size):
    dataset = DepthDataset(image_dir, depth_dir, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# Function to load the model
def load_model(pretrained=True):
    if pretrained:
        model = AutoModelForDepthEstimation.from_pretrained(MODEL_PATH)
    else:
        model = AutoModelForDepthEstimation.from_pretrained(HF_MODEL)
    model.to(DEVICE)
    return model


# Function to save the model
def save_model(model, save_path, save_model_name):
    model_save_path = os.path.join(save_path, f"{save_model_name}")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


# Function to train the model
def train(
    model, train_loader, num_epochs, learning_rate, save_model_path, save_model_name
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")  # Initialize best loss to a large value

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} started...")
        model.train()
        total_loss = 0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(pixel_values=pixel_values)
            predicted_depth = outputs.predicted_depth

            # Adjust output shape if necessary
            if predicted_depth.ndim == 3:
                predicted_depth = predicted_depth.unsqueeze(1)

            # Calculate loss
            loss = torch.nn.functional.mse_loss(predicted_depth, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Save the model if the current loss is the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, save_model_path, save_model_name)

    print("Training completed!")


# Main execution block
if __name__ == "__main__":
    # Load processor
    processor = AutoImageProcessor.from_pretrained(HF_MODEL)

    # Create dataset and DataLoader
    train_loader = create_dataset(IMAGE_DIR, DEPTH_DIR, processor, BATCH_SIZE)

    # Load model
    model = load_model(PRETRAINED)

    # Train the model
    train(
        model, train_loader, NUM_EPOCHS, LEARNING_RATE, SAVE_MODEL_PATH, SAVE_MODEL_NAME
    )
