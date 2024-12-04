# Hands-on AI EEG workshop
https://zenodo.org/records/5055046

## Requirements
1. Locally create new virtual environment `python3 -m venv <env_name>`
2. Run `pip install -r requirements.txt` to install all dependencies
3. To add new libraries, run `pip freeze > requirements.txt` to update requirements.txt file

## Data
This repo runs on EEG dataset located at https://zenodo.org/records/5055046 [1]
In total, this dataset has ~13,000 epochs with 61 channels. 'Epoch' in this context is a segment of continuous EEG data or a sample. 
Data is loaded using the `draft_loader.py` program, saved as `all_epochs.pickle`. 
    - Data shape: $(13410, 61, 500)$
        - $13410$ = number of epochs/samples
        - $61$ = number of channels
        - $500$ = number of timepoints = $2$ seconds of $250$ Hz data. 

Data is normalised on a channel by channel basis per mini-batch (e.g. ~200 out of ~13000 samples).

## Problem Statement
Predict the last few datapoints on the basis of the first many datapoints.
This is a **sequence-to-sequence regression task**. 

## Process
1. Run `draft_loader.py` to generate `all_epochs.pickle`. 
2. Run `split_data.py` to rearrange data to be [batch_size, timepoints=500, channels=61] train and test split data in `torch` format (`train_loader.pth` and `test_loader.pth`). 
3. Run embedding in linear layer
4. Pass through transformer
5. Revert to output dimensions for comparison
6. Implement loss function + loop

## Test-Training Split
80% training and 20% test. Evaluated to see how similar model predictions are as a linear regression problem. 
The problem is a sequence-to-sequence regression task, in which the last 50 timepoints are predicted on the basis of the first 450. The dataset is split in accordance to that and saved as `train_loader.pth` and `test_loader.pth` with a batch size of 32.
Timepoints can both be size 450 - the test dataset will be from timepoints 50 to 500 and training is from 0 to 450. 

## Linear Embedding
61 is a prime number. Data passed through a linear layer for embedding the input to a hidden dimension using `self.embedding = nn.Linear(input_channels, hidden_dim)`. This embeds the data to a nicer number that is divisible by the number of heads. 
    - input_channels = 61
    - timepoints = c. 500
    - hidden_dim = hidden_dim = 64
    - num_heads = 4
    - num_layers = 4  (number of layers for transformer)
    - output_dim = 64

### Source and Target
`src: (batch_size, channels, timepoints) -> [batch_size, timepoints, channels]`
`src = src.permute(2, 0, 1)  # Shape: [timepoints, batch_size, channels]`
`tgt = tgt.permute(2, 0, 1)  # Shape: [timepoints, batch_size, channels]`


## Transformer
# Transformer model
`transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers)`
`transformer_out = self.transformer(src, tgt)`

## Comparing to True Value
Convert back from 64 to 61 channels, compare to true value using `output = fc_out(transformer_out)`. 


## Citations
[1] Hinss, M. F., Darmet, L., Somon, B., Jahanpour, E., Lotte, F., Ladouce, S., & Roy, R. N. (2021). An EEG dataset for cross-session mental workload estimation: Passive BCI competition of the Neuroergonomics Conference 2021 (Version 2) [Data set]. Neuroergonomics Conference, Munich, Germany. Zenodo. https://doi.org/10.5281/zenodo.5055046
