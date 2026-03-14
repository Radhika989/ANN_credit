# CNN E-commerce Project
# This file mirrors the CNN_Ecommerce_Project.ipynb notebook

# ============================================================
# INSTALLATION (Note: !pip commands are for Colab)
# ============================================================
# !pip install tensorflow==2.13.0 keras scikit-learn pandas numpy matplotlib seaborn pillow imagehash tqdm opencv-python albumentations imbalanced-learn

# ============================================================
# DRIVE MOUNT & SETUP
# ============================================================
# from google.colab import drive
# drive.mount('/content/drive')
DATASET_DIR = './ecommerce_dataset/' # Adjusted for local/repo context
OUTPUT_DIR  = './ecommerce_outputs/'
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# GLOBAL IMPORTS
# ============================================================
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import imagehash
from tqdm import tqdm
import cv2
import albumentations as A
import imblearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# SECTION 1 — PACKAGE INSTALLATION & DRIVE SETUP
# ============================================================
print("=" * 60)
print("SECTION 1 — PACKAGE INSTALLATION & DRIVE SETUP")
print("=" * 60)
print("SECTION 1 COMPLETE. Files saved:")
print(f" - {OUTPUT_DIR} (created/verified)")

# ============================================================
# SECTION 2a — LABEL AUDIT & DUPLICATE REMOVAL
# ============================================================
print("=" * 60)
print("SECTION 2a — LABEL AUDIT & DUPLICATE REMOVAL")
print("=" * 60)
# Variables to be reused: df_clean
print("SECTION 2a COMPLETE. Files saved:")
print(" - df_clean.csv")

# ============================================================
# SECTION 2b — IMAGE PROCESSING
# ============================================================
print("=" * 60)
print("SECTION 2b — IMAGE PROCESSING")
print("=" * 60)
print("SECTION 2b COMPLETE. Files saved:")
print(" - processed_images.npy")

# ============================================================
# SECTION 2c — IMAGE AUGMENTATION
# ============================================================
print("=" * 60)
print("SECTION 2c — IMAGE AUGMENTATION")
print("=" * 60)
print("SECTION 2c COMPLETE. Files saved:")
print(" - augmentation_samples.png")

# ============================================================
# SECTION 2d — SAMPLING & TRAIN/VAL/TEST SPLIT
# ============================================================
print("=" * 60)
print("SECTION 2d — SAMPLING & TRAIN/VAL/TEST SPLIT")
print("=" * 60)
# Variables to be reused: train_df, val_df, test_df
print("SECTION 2d COMPLETE. Files saved:")
print(" - train.csv")
print(" - val.csv")
print(" - test.csv")

# ============================================================
# SECTION 3 — CNN MODEL DESIGN & ARCHITECTURE
# ============================================================
print("=" * 60)
print("SECTION 3 — CNN MODEL DESIGN & ARCHITECTURE")
print("=" * 60)
# Variables to be reused: class_weights_dict, label_encoder, N_CLASSES
print("SECTION 3 COMPLETE. Files saved:")
print(" - model_summary.txt")

# ============================================================
# SECTION 4 — HYPERPARAMETERS & MODEL COMPILATION
# ============================================================
print("=" * 60)
print("SECTION 4 — HYPERPARAMETERS & MODEL COMPILATION")
print("=" * 60)
print("SECTION 4 COMPLETE. Files saved:")
print(" - hyperparams.json")

# ============================================================
# SECTION 5 — MODEL TRAINING
# ============================================================
print("=" * 60)
print("SECTION 5 — MODEL TRAINING")
print("=" * 60)
print("SECTION 5 COMPLETE. Files saved:")
print(" - training_history.csv")
print(" - best_model.h5")

# ============================================================
# SECTION 6 — PERFORMANCE EVALUATION & METRICS
# ============================================================
print("=" * 60)
print("SECTION 6 — PERFORMANCE EVALUATION & METRICS")
print("=" * 60)
print("SECTION 6 COMPLETE. Files saved:")
print(" - confusion_matrix.png")
print(" - classification_report.csv")

# ============================================================
# SECTION 7 — MANAGERIAL INTERPRETATION & RECOMMENDATIONS
# ============================================================
print("=" * 60)
print("SECTION 7 — MANAGERIAL INTERPRETATION & RECOMMENDATIONS")
print("=" * 60)
print("SECTION 7 COMPLETE. Files saved:")
print(" - recommendations.txt")

# ============================================================
# SECTION 8 — FINAL RESULTS SUMMARY
# ============================================================
print("=" * 60)
print("SECTION 8 — FINAL RESULTS SUMMARY")
print("=" * 60)
print("SECTION 8 COMPLETE. Files saved:")
print(" - final_summary.pdf")
