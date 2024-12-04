import mne

FILE_PATH = './dataset_hackathon/P01/S1/eeg/alldata_sbj01_sess1_MATBdiff.set'

epochs = mne.io.read_epochs_eeglab(FILE_PATH, verbose=False)  # FIR 1-40 Hz
epochs_numpy = epochs.get_data()
print(epochs_numpy)
print(epochs_numpy.shape)
