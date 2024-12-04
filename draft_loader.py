import glob
import mne

FILE_PATH = './dataset_hackathon/P01/S1/eeg/alldata_sbj01_sess1_MATBdiff.set'

all_epochs = None
p_folders = glob.glob('./dataset_hackathon/P*/')
for p_folder in p_folders:
    s_folders = glob.glob(f'{p_folder}S*')
    for s_folder in s_folders:
        files = glob.glob(f'{s_folder}/eeg/*MAT*.set')
        for f in files:
            epochs = mne.io.read_epochs_eeglab(f, verbose=False)  # FIR 1-40 Hz
            if all_epochs is None:
                all_epochs = epochs.copy()
            else:
                all_epochs = mne.concatenate_epochs([all_epochs, epochs.copy()], verbose=False)

all_epochs_numpy = all_epochs.get_data()
print(all_epochs_numpy)
print(all_epochs_numpy.shape)
