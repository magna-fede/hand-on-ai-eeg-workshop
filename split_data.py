# Splits data contained in all_epochs.pickle into train and test sets for
# sequence-to-sequence regression: 80% train, 20% test

import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
# import sys


print('Loading data...')
with open('all_epochs.pickle', 'rb') as r:
    all_samples = pickle.load(r)
    data = all_samples.get_data()  # X data
    labels = all_samples.events[:, -1]  # labels, for information only

print(type(data))
print(data.shape)
print(all_samples.info)

print('Data loaded.')


X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")  # Expected: (10728, 61, 500)
print(f"Test data shape: {X_test.shape}")       # Expected: (2682, 61, 500)

# Reorder to be (batch, timepoints, channels)
X_train = np.transpose(X_train, (0, 2, 1))  # Shape: (10728, 500, 61)
X_test = np.transpose(X_test, (0, 2, 1))    # Shape: (2682, 500, 61)


print('Data split. Converting to torch tensors...')
X_train = torch.tensor(X_train, dtype=torch.float32)  # Shape: (10728, 61, 500)
X_test = torch.tensor(X_test, dtype=torch.float32)    # Shape: (2682, 61, 500)

print(f"X_train shape: {X_train.shape}")  # (10728, 61, 500)
print(f"X_test shape: {X_test.shape}")    # (2682, 61, 500)

print('Data converted. Creating DataLoader for batching...')
train_dataset = TensorDataset(X_train)
test_dataset = TensorDataset(X_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print dataset info
print(f"Training DataLoader: {len(train_loader)} batches")
print(f"Test DataLoader: {len(test_loader)} batches")

print('DataLoaders created. Saving data...')
torch.save(train_loader, 'train_loader.pth')
torch.save(test_loader, 'test_loader.pth')

print('Data saved.')
