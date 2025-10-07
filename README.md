# Deep Learning Project – Beans Image Classification

This project is a GPU-accelerated deep learning application that classifies coffee beans images into three categories (grades 1-3) using a PyTorch convolutional neural network. 
It uses an open-source dataset from Hugging Face (`coffee-beans` dataset), manually downloaded for ease and demonstrates effective use of GPU for training a transfer-learning model. 

## Project Structure

- **`src/`**: Source code modules.
  - `data_loader.py` – Functions to download/preprocess the dataset and create PyTorch DataLoaders.
  - `model.py` – Defines the neural network model (using a pre-trained ResNet18) and prepares it for fine-tuning.
  - `train.py` – Script to train the model; parses command-line arguments for configuration.
  - `evaluate.py` – Script to evaluate the trained model on the evaluation and output accuracy.
  - `utils.py` – Utility functions (e.g., for device selection and random seeding).
  - `test.py`- Test on a seperate dataset to check per class accuracy.
- **`scripts/`**: Helper shell scripts.
  - `run.sh` – Example script showing how to run training and evaluation with default parameters.
- **`results/`**: Output artifacts from running the code.
  - `training_log.txt` – Sample log of the training process (epoch losses and accuracies).
  - `installation_log.txt` – Sample log of dependency installation for project.
  - `evaluation_log.txt` – Sample log of evaluation script run.
  - `test_log.txt` – Sample log of test script run.
  - `model.pth` - Trained resnet model checkpoint using dataset under `data/train`
- **`requirements.txt`**: Python dependencies to install.
- **`README.md`**: Project documentation (you are reading it).

## Setup and Installation

**Prerequisites:**  
- Python 3.8+ (tested on Python 3.9)  
- An NVIDIA GPU with CUDA support (optional but recommended for training). The code will automatically use a GPU if available; otherwise, it will run on CPU.  
- PyTorch and related libraries.

**Steps:**

1. **Clone the repository** (or copy the files) to your local machine.
2. **Install dependencies:**  
   Use pip to install required packages:  
   ```bash
   pip install -r requirements.txt
   ```

**Execution:**
Run the training and evaluation via the provided shell script, or manually using Python. Both approaches utilize the command-line interface (CLI) of the scripts:
```bash
bash scripts/run.sh
```
This will execute training for a default configuration (5 epochs, batch size 32, learning rate 0.001 on the beans dataset) and then run the evaluation. 
You can open and modify run.sh to change these defaults or run individual steps.

***Running manually with Python:***
```bash
python src/train.py --dataset data/beans/train --epochs 5 --batch_size 32 --learning_rate 0.001 --output_dir results/
```
Options:

--dataset : Hugging Face dataset name (default "beans"). You can replace this with another dataset name if desired.
--epochs : Number of training epochs (default 5 for a quick demo; you can increase for better performance).
--batch_size : Batch size for DataLoader (default 32). Adjust based on your GPU memory.
--learning_rate : Learning rate for the optimizer (default 0.001).
--output_dir : Directory to save outputs (model and log). By default, it uses results/. If not existing, it will be created.
This script will print training progress to the console, including loss and accuracy for each epoch. 
It will also save a log of these metrics to results/training_log.txt and save the final model weights to results/model.pth. 
(You can change the output_dir or filename via arguments if needed.)

***Evaluate the model:***
After training, run the evaluation script:
```bash
python src/evaluate.py --dataset beans --model_path results/model.pth --batch_size 32
```
Options:

--model_path : Path to the saved model file (default results/model.pth as saved by train.py).
--dataset : Dataset name (should be the same dataset used for training, default "beans"). The script will load the test split of this dataset for evaluation.
--batch_size : Batch size for evaluation (default 32).

This will load the model and dataset, then print the overall accuracy on the test set to the console (e.g., "Evaluation accuracy: 92.30%"). 
You can modify the script to print more details or per-class accuracy if needed.

Note: Both scripts have additional arguments (e.g., --device to force CPU/GPU, etc.) – run python src/train.py -h or python src/evaluate.py -h to see all options.

***Testing the model:***
After training and evaluation run the test script for checking accuracy on unseen samples:
```bash
python src/test.py --model_path results/my_model.pth --test_data_path data/beans/test
```
Options:

--model_path : Path to the model file (default results/model.pth as saved by train.py).
--test_data_path : Path to test dataset files

## Project Overview


Data Loading: The data_loader.py module uses Hugging Face’s datasets library to fetch the dataset (by name) or manually downloaded dataset. 
It applies necessary transformations (resizing images, converting to PyTorch tensors, normalization, etc.) and prepares PyTorch DataLoader objects for the training, validation, and test splits. 
This encapsulation makes it easy to swap in a different dataset by changing one argument (--dataset). (For example, you could use --dataset cifar10 with slight modifications, as long as the dataset is available on Hugging Face.) 
The Beans dataset has three classes; the code automatically infers this from the dataset metadata using the ClassLabel feature.

Model Definition: The model.py defines create_model(num_classes) which loads a pre-trained ResNet-18 model from torchvision and replaces the final layer to output the correct number of classes for our dataset (3 in the beans example). 
By using a pre-trained model, we leverage transfer learning so the training can converge faster on a small dataset. (All layers are left trainable in this implementation, but you could freeze some early layers for possibly even faster training.)

Training Process: The train.py script ties everything together. It parses command-line arguments for configuration, sets up the device (GPU or CPU) for computation, and calls the data loader to get the train and validation DataLoaders. 
It then initializes the model via create_model, moves it to the GPU if available, and sets up the optimizer (Adam) and loss function (cross-entropy for multi-class classification). 
The training loop runs for the specified number of epochs: for each batch, it performs a forward pass, calculates loss, does backpropagation, and updates weights. 
It also accumulates statistics to compute average loss and accuracy for the epoch. After each epoch, the script runs a validation pass (with torch.no_grad() and model.eval() mode) to evaluate performance on the validation set. 
Progress is printed out and logged. For example:
```
Epoch 1/5, Train Loss: 1.05, Train Acc: 55.3%, Val Loss: 0.89, Val Acc: 66.7%
Epoch 2/5, Train Loss: 0.74, Train Acc: 70.1%, Val Loss: 0.58, Val Acc: 80.0%
...
Epoch 5/5, Train Loss: 0.29, Train Acc: 91.5%, Val Loss: 0.25, Val Acc: 92.3%
```
(Refer results/training_log.txt for the full log.) 
This shows the model improving each epoch, after final epoch, the trained model is saved to results/model.pth.

Evaluation: The evaluate.py script loads the saved model and uses the test set to measure final performance. It reports the overall accuracy. 
In our example, we expect around 90%+ accuracy on the beans test set (since the model learned to classify the beans images well). 
This separate evaluation step demonstrates how to use the trained model for inference on new data.

Throughout the code, we use GPU acceleration where possible. The model and data batches are moved to torch.device("cuda") when available, so matrix operations (forward pass, backward pass) execute on the GPU. 
This significantly speeds up training on image data. If no GPU is present, the code will automatically fall back to CPU to ensure usability in any environment. 
We also set a random seed for reproducibility in utils.py (so results are deterministic up to the inherent nondeterminism of CUDA operations).


Sample Coffee Bean Images and their Grades :

<img width="167" height="160" alt="image" src="https://github.com/user-attachments/assets/4a179944-0ef1-4d9a-b76b-27d7d8f787f0" />
Bean Grade 1

<img width="261" height="234" alt="image" src="https://github.com/user-attachments/assets/596a1a13-e785-4a7b-9116-09a23d5ab1d6" />
Bean Grade 2

<img width="259" height="276" alt="image" src="https://github.com/user-attachments/assets/03188a7c-1638-4fd8-9912-31f78da035aa" />
Bean Grade 3

Model gets trained on these 3 types and tries to bin/grade as per trained data.
From test data, we observed identifying grade 2 has lessor accuracy than other two grades.

```
$ python src/test.py --model_path results/model.pth --test_data_path data/beans/test
Using device: cpu

Overall Test Accuracy: 93.50% (187/200)

Per-Class Accuracy:
  Class '1': 100.00% (50/50)
  Class '2': 74.00% (37/50)
  Class '3': 100.00% (100/100)
```
