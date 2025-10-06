import os
import torch
import argparse
from datetime import datetime
from utils import get_default_device, set_seed
import data_loader
import model as model_module

def main():
    parser = argparse.ArgumentParser(description="Train a deep learning model on a dataset")
    parser.add_argument("--dataset", type=str, default="beans", help="HuggingFace dataset name to use")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save model and logs")
    parser.add_argument("--no_cuda", action="store_true", help="Force training on CPU even if GPU is available")
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Prepare device
    device = torch.device("cpu")
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    train_loader, val_loader, test_loader, num_classes = data_loader.get_dataloaders(args.dataset, args.batch_size)
    if num_classes is None:
        # Fallback: if unable to infer, get unique label count from train set
        # (This requires loading the entire train dataset into memory)
        labels = []
        for batch in train_loader:
            labels.extend(batch["labels"] if isinstance(batch, dict) else batch[1])
        num_classes = len(set(labels))
    print(f"Number of classes: {num_classes}")
    
    # Initialize model
    net = model_module.create_model(num_classes)
    net.to(device)
    
    # Set up loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    
    # Open a log file to record training progress
    log_path = os.path.join(args.output_dir, "training_log.txt")
    log_file = open(log_path, "w")
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            # Support both dict batch (from HuggingFace Datasets) or tuple (data, target)
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                labels = batch["labels"].to(device)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Track training loss and accuracy
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        # Compute average loss and accuracy for the epoch
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        # Validation evaluation (if validation set is available)
        val_loss = None
        val_acc = None
        if val_loader:
            net.eval()
            val_correct = 0
            val_total = 0
            val_running_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, dict):
                        images = batch["image"].to(device)
                        labels = batch["labels"].to(device)
                    else:
                        images, labels = batch
                        images = images.to(device)
                        labels = labels.to(device)
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
            val_loss = val_running_loss / val_total
            val_acc = 100.0 * val_correct / val_total
            net.train()  # back to training mode
        
        # Log and print epoch results
        if val_loss is not None:
            log_line = (f"Epoch {epoch}/{args.epochs}, "
                        f"Train Loss: {epoch_loss:.3f}, Train Acc: {epoch_acc:.1f}%, "
                        f"Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.1f}%\n")
        else:
            log_line = (f"Epoch {epoch}/{args.epochs}, "
                        f"Train Loss: {epoch_loss:.3f}, Train Acc: {epoch_acc:.1f}%\n")
        print(log_line.strip())
        log_file.write(log_line)
    
    # Save the trained model
    model_path = os.path.join(args.output_dir, "model.pth")
    torch.save(net.state_dict(), model_path)
    print(f"Training complete. Model saved to {model_path}")
    log_file.close()

if __name__ == "__main__":
    main()
