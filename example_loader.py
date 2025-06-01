import torch
from torch.utils.data import DataLoader
from config import *

# load dataset
test_set = torch.load('ofdm_dataset_4_8_sdr.pth', weights_only=False)

# Check the first sample in the dataset
pdsch_iq, labels, sinr = test_set[0]

# Print the shape of the first sample and its label
print(pdsch_iq.shape, labels.shape, sinr)


#### Create DataLoader
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# Check the first sample from the DataLoader
for sample in test_loader:
    pdsch_iq, labels, sinr = sample
    break 