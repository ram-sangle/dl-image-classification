import torch
import argparse
from utils import get_default_device
import data_loader
import model as model_module
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(dataset_path="data/beans/train", batch_size=32):
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load dataset using ImageFolder
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # Split into train and validation sets (e.g., 80/20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # No test set in this structure, so return None
    test_loader = None

    # Number of classes
    num_classes = len(full_dataset.classes)

    return train_loader, val_loader, test_loader, num_classes

def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained model on the test set")
    parser.add_argument("--dataset", type=str, default="beans", help="Dataset name (same as used in training)")
    parser.add_argument("--model_path", type=str, default="results/model.pth", help="Path to saved model weights")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--no_cuda", action="store_true", help="Use CPU only for evaluation")
    args = parser.parse_args()
    
    device = torch.device("cpu")
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Evaluation using device: {device}")
    
    # Load data (we will use test set; if no test, use validation)
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(dataset_path="data/beans/train", batch_size=args.batch_size)
    eval_loader = test_loader if test_loader is not None else val_loader
    if eval_loader is None:
        raise ValueError("No test or validation split available for evaluation.")
    
    # Initialize model and load weights
    if num_classes is None:
        # If num_classes couldn't be inferred, fall back to length of class names or labels in eval set
        sample_batch = next(iter(eval_loader))
        if isinstance(sample_batch, dict):
            sample_labels = sample_batch["labels"]
        else:
            sample_labels = sample_batch[1]
        num_classes = len(torch.unique(sample_labels))
    net = model_module.create_model(num_classes)
    net.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    net.to(device)
    net.eval()
    
    # Evaluation loop
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in eval_loader:
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                labels = batch["labels"].to(device)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
            outputs = net(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    accuracy = 100.0 * correct / total
    print(f"Evaluation accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
