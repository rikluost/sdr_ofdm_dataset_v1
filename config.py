import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self):
        self.pdsch_iq = [] # pdsch symbols
        self.labels = [] # original bitstream labels
        self.sinr = [] # SINR

    def __len__(self):
        return len(self.pdsch_iq)
    
    def __getitem__(self, index):
        x1 = self.pdsch_iq[index]
        y = self.labels[index]
        z = self.sinr[index]
        return x1, y, z
    
    def add_item(self, new_pdsch_iq,  new_label, new_sinr):
        self.pdsch_iq.append(new_pdsch_iq) 
        self.labels.append(new_label) 
        self.sinr.append(new_sinr) 
        