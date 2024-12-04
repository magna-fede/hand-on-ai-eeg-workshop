# Splits data contained in all_epochs.pickle into train and test sets. 80% train, 20% test. Use torch where possible


import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import sys
print(np.__version__)
sys.exit()

print('Loading data...')
with open('all_epochs.pickle', 'rb') as r:
    all_samples = pickle.load(r)
    data = all_samples.get_data()
    labels = all_samples.events[:, -1]

print(type(data))
print(data.shape)


print('Data loaded. Splitting data...')

#X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
