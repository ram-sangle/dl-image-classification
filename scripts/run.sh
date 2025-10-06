#!/bin/bash
# Run training for the deep learning capstone project

# Train the model on the beans dataset (5 epochs as an example)
python src/train.py --dataset beans --epochs 5 --batch_size 32 --learning_rate 0.001 --output_dir results/

# Evaluate the model on the test set
python src/evaluate.py --dataset beans --model_path results/model.pth --batch_size 32
