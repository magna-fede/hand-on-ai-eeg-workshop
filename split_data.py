# Splits data contained in all_epochs.pickle into train and test sets. 80% train, 20% test. Use torch where possible


import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


print('Loading data...')
with open('all_epochs.pickle', 'rb') as r:
    all_samples = pickle.load(r)
    data = all_samples.get_data() # X data
    labels = all_samples.events[:, -1] # y data

print(type(data))
print(data.shape)


print('Data loaded. Splitting data...')
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")  # Expected: (10728, 61, 500)
print(f"Test data shape: {X_test.shape}")  # Expected: (2682, 61, 500)



print('Data split. Converting to torch tensors...')
X_train = torch.tensor(X_train, dtype=torch.float32) # Shape: (10728, 61, 500)
X_test = torch.tensor(X_test, dtype=torch.float32)   # Shape: (2682, 61, 500)
y_train = torch.tensor(y_train, dtype=torch.int64)   # Shape: (10728,)
y_test = torch.tensor(y_test, dtype=torch.int64)     # Shape: (2682,)

print(f"X_train shape: {X_train.shape}")  # (10728, 61, 500)
print(f"X_test shape: {X_test.shape}")    # (2682, 61, 500)
print(f"y_train shape: {y_train.shape}")  # (10728,)
print(f"y_test shape: {y_test.shape}")    # (2682,)