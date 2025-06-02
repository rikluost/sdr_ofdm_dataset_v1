# Simulated Radio Channel Dataset for NN-Based Receivers

## Abstract

We introduce a measured radio channel dataset designed for training and evaluating neural network-based wireless receivers, specifically optimized for OFDM systems operating at 435 MHz. This dataset includes IQ symbols after Discrete Fourier Transform (DFT), the corresponding transmitted bitstreams (labels), and per-transmission SINR metrics. Captured in realistic indoor/outdoor non-line-of-sight conditions using Software-Defined Radio (SDR) equipment, the dataset provides essential resources for validating neural network architectures against practical channel effects.

Keywords: Neural Networks, OFDM, IQ Data, Channel Measurement, SINR, Software-Defined Radio

## Introduction

Neural network-based receivers are increasingly important for robust signal demodulation and accurate channel estimation. However, practical datasets capturing authentic mobile channel dynamics, particularly under non-line-of-sight conditions, remain scarce. This dataset addresses this gap by providing realistically measured IQ samples collected via mobile SDR antennas moving at pedestrian speeds.

## Measurement Setup

Measurements were conducted using an SDR system operating at a center frequency of 435 MHz. Data acquisition involved a stationary transmitter antenna and a mobile receiver antenna moving indoors and outdoors in non-line-of-sight environments.

Hardware: SDR radio moving at speed < 3 m/s
Center frequency: 435 MHz, restricted by transmit license
Cyclic prefix: 6 samples, chosen experimentally
Channel: Real-world fading, measured under practical indoor/outdoor conditions
OFDM parameters: FFT size 128, 102 active subcarriers, 16-QAM modulation. These were selected experimentally to suit radio transmit license, computational restrictions and limited maximum transmit power.  

The OFDM parameters were chosen based on licensing constraints, computational feasibility, and transmit power limitations.

The dataset consists of approximately 1000 TTIs, each capturing realistic channel variations, and took around 5 minutes to generate and process using SDR equipment and data processing pipelines.



## Dataset Structure 

The dataset is structured as a PyTorch CustomDataset, facilitating seamless integration into deep learning workflows.

- **File format:** PyTorch `.pth` file containing a pickled instance of `CustomDataset`.
- **FFT size:** 128
- **Number of subcarriers:** 102 (active subcarriers)
- **DC:** is included in subcarriers, but transmitted with zeros. Needs to be removed from the IQ data
- **OFDM offsets:** 13 subcarriers on either side of the modulated subcarriers (13+102+13 = 128)
- **Data types:**
  - IQ data: `torch.complex64`
  - Labels: `torch.float32` (bitstream, typically 0 or 1)
  - SINR: `torch.float64` (in dB)



### Field Definitions
Sample structure (per TTI):

```

| Component       | Shape          | Description                              |

|-----------------|----------------|------------------------------------------|

|  pdsch_iq       |  [14, 128]     | Full DFT output (includes DC, offsets)   |

| ├─   Used       |  [14, 101]     | Active subcarriers (DC & offsets removed)|

| └─   Pilots     |  [3rd symbol]  | Pilot symbols location                   |

|-----------------|----------------|------------------------------------------|

|   labels        |  [1400, 4]     | QAM bits per symbol                      |

|-----------------|----------------|------------------------------------------|

|   sinr          |  [1]           | Signal-to-Interference-plus-Noise Ratio (dB) |

```


#### Calculations for number of elements in each field

- n_s is the number of symbols in TTI = 128 * 14 = 1792
- n_offset is the number of unmodulated subcarriers on both sides = 13
- n_subcarriers is the number of modulated subcarriers = n_s-(2*n_offset) = 102 
- n_DC is the number of DC symbols = 14
- n_pilots is the number of pilots is = 14
- n_mod_symbols is the number of modulated symbols in TTI = 102 * 14 - n_pilots - n_DC = 1400
- Qm is the number of bits per symbol = 4 (i.e. 16-QAM)
- n_bits is the number of bits per TTI i.e. labels in TTI = n_mod_symbols * Qm = 1400 * 4 = 5600 

#### Pilot Information
- Pilots are present only in the 3rd OFDM symbol (index 2).
- Pilots are placed on every 8th subcarrier, **plus the last subcarrier (index 101) is always included as a pilot** for full-band coverage.  
  - i.e. pilot indices are: 0, 8, 16, ..., 96, 101

## Dataset Visualizations

Figures illustrating transmitted and received signal PSD, as well as OFDM symbol allocation, are provided within the dataset documentation.

![alt text](https://github.com/rikluost/sdr_ofdm_dataset_v1/blob/main/pics/PSD_TX.png)
Fig 1. PSD of a transmitted TTI.

![alt text](https://github.com/rikluost/sdr_ofdm_dataset_v1/blob/main/pics/PSD_RX.png)
Fig 2. PSD of a received TTI.

![alt text](https://github.com/rikluost/sdr_ofdm_dataset_v1/blob/main/pics/OFDM_blockmask.png)
Fig 3. Visualization of the TTI structure.

![alt text](https://github.com/rikluost/sdr_ofdm_dataset_v1/blob/main/pics/OFDM_blockmod.png)
Fig 4. Visualization of modulated TTI transmitted.

![alt text](https://github.com/rikluost/sdr_ofdm_dataset_v1/blob/main/pics/OFDM_block_RX.png)
Fig 5. Visualization of received TTI.


## Challenges

Key challenges during dataset creation included precise synchronization of SDR equipment to ensure accurate IQ sampling, handling environmental variability between indoor and outdoor measurement sessions, and ensuring robust DC removal. Additional complexities arose from accurately placing and verifying pilot tones, necessary for reliable channel estimation and SINR calculation.

## Example Use

### Load Data

Below is a simple example showing how to load and access samples from the dataset, including extraction of the pilot subcarriers.

```python
import torch

# Load the dataset 
dataset = torch.load('custom_dataset.pth')

# Get the first sample
pdsch_iq, labels, sinr = dataset[0]

print(f'PDSCH IQ shape: {pdsch_iq.shape}')      
# (num_symbols=14, FFT_size=128)

print(f'Labels shape: {labels.shape}')          
# ( modulated resource elements=1400, bits_per_TTI=4)

print(f'SINR: {sinr}')                          
# scalar (dB)

# Example: Extract pilot subcarriers from 3rd OFDM symbol
third_symbol = pdsch_iq[2]                      # shape: (102,)
pilot_indices = list(range(0, 102, 8))
if pilot_indices[-1] != 101:
    pilot_indices.append(101)
pilot_values = third_symbol[pilot_indices]
print('Pilot indices:', pilot_indices)
print('Pilot values:', pilot_values)
```
### CustomDataset

A CustomDataset needs to be defined specifically in a file named `config.py`.

```
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
```