import glob
import mne
import pickle
import torch
import torch.nn as nn


# Load epochs
if not glob.glob('./all_epochs.pickle'):
    all_epochs = None
    p_folders = glob.glob('./dataset_hackathon/P*/')
    for p_folder in p_folders:
        s_folders = glob.glob(f'{p_folder}S*')
        for s_folder in s_folders:
            files = glob.glob(f'{s_folder}/eeg/*MAT*.set')
            for f in files:
                epochs = mne.io.read_epochs_eeglab(f, verbose=False)
                if all_epochs is None:
                    all_epochs = epochs.copy()
                else:
                    all_epochs = mne.concatenate_epochs(
                        [all_epochs, epochs.copy()], verbose=False
                    )

    with open('all_epochs.pickle', 'wb') as w:
        pickle.dump(all_epochs, w)

else:
    with open('all_epochs.pickle', 'rb') as r:
        all_epochs = pickle.load(r)

all_epochs_numpy = all_epochs.get_data()
print(all_epochs_numpy)
print(all_epochs_numpy.shape)


# Try transformer
x = all_epochs_numpy.swapaxes(2, 1)
x = torch.from_numpy(x)

embedding = nn.Linear(61, 64).double()
transformer = nn.Transformer(d_model=64, nhead=4, batch_first=True).double()
out = transformer(embedding(x[:4, :450, :]), embedding(x[:4, 50:, :]))
