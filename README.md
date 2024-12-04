# Hands-on AI EEG workshop
https://zenodo.org/records/5055046

## Requirements
1. Locally create new virtual environment `python3 -m venv <env_name>`
2. Run `pip install -r requirements.txt` to install all dependencies
3. To add new libraries, run `pip freeze > requirements.txt` to update requirements.txt file

## Data
This repo runs on EEG dataset located at https://zenodo.org/records/5055046 [1]
In total, this dataset has 13,000 epochs with 61 channels. 
Data is loaded using the `draft_loader.py` program, saved as `all_epochs.pickle`. 


## Citations
[1] Hinss, M. F., Darmet, L., Somon, B., Jahanpour, E., Lotte, F., Ladouce, S., & Roy, R. N. (2021). An EEG dataset for cross-session mental workload estimation: Passive BCI competition of the Neuroergonomics Conference 2021 (Version 2) [Data set]. Neuroergonomics Conference, Munich, Germany. Zenodo. https://doi.org/10.5281/zenodo.5055046
