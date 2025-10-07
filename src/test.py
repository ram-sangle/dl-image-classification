import os
import torch
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import create_model

def evaluate_model(model_path, test_data_path, batch_size, use_cuda):
    # Select device
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load test dataset
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    num_classes = len(test_dataset.classes)
    model = create_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Evaluation loop
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    overall_acc = 100.0 * correct / total
    print(f"\nOverall Test Accuracy: {overall_acc:.2f}% ({correct}/{total})")

    print("\nPer-Class Accuracy:")
    for i in range(num_classes):
        acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        print(f"  Class '{test_dataset.classes[i]}': {acc:.2f}% ({class_correct[i]}/{class_total[i]})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on test dataset")
    parser.add_argument("--model_path", type=str, default="results/model.pth", help="Path to trained model file")
    parser.add_argument("--test_data_path", type=str, default="data/beans/test", help="Path to test dataset folder")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        test_data_path=args.test_data_path,
        batch_size=args.batch_size,
        use_cuda=args.use_cuda
    )
