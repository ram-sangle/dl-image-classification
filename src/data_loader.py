import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

def get_dataloaders(dataset_name: str = "beans", batch_size: int = 32):
    """
    Load the specified dataset from Hugging Face and prepare PyTorch DataLoaders.
    Returns train_loader, val_loader, test_loader, and number of classes.
    """
    # Load dataset (automatically downloads if not already cached)
    ds = load_dataset(dataset_name)
    train_ds = ds["train"]
    # Use 'validation' split if available, otherwise use 'test' for validation
    if "validation" in ds:
        val_ds = ds["validation"]
    elif "test" in ds:
        val_ds = ds["test"]
    else:
        val_ds = None
    # Use 'test' split if it exists (some datasets have train/validation/test)
    test_ds = ds.get("test", None)
    
    # Define image transformations: resize to 224x224 and normalize
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Function to apply transform to a batch of examples
    def preprocess(batch):
        # 'batch["image"]' is a list of PIL images; apply transform to each
        batch["image"] = [transform(img) for img in batch["image"]]
        return batch
    
    # Set the transform for datasets (this will apply on-the-fly during iteration)
    train_ds.set_transform(preprocess)
    if val_ds is not None:
        val_ds.set_transform(preprocess)
    if test_ds is not None:
        test_ds.set_transform(preprocess)
    
    # Create DataLoader for each split
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds is not None else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False) if test_ds is not None else None
    
    # Determine number of classes from the dataset features (if available)
    num_classes = None
    try:
        # Hugging Face datasets with ClassLabel have this property
        if "labels" in train_ds.features:
            num_classes = train_ds.features["labels"].num_classes
        elif "label" in train_ds.features:
            num_classes = train_ds.features["label"].num_classes
    except AttributeError:
        pass
    
    return train_loader, val_loader, test_loader, num_classes
