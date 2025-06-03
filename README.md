# Simulated Radio Channel Dataset for NN-Based Receivers

## Abstract

We introduce a measured radio channel dataset designed for training and evaluating neural network-based wireless receivers, tailored for OFDM systems. This dataset includes IQ symbols after Discrete Fourier Transform (DFT) in the receiver, the corresponding transmitted bitstreams (labels), and per-transmission SINR metrics. Captured in realistic indoor mostly non-line-of-sight conditions using Software-Defined Radio (SDR) equipment, the dataset provides resources for validating neural network architectures against practical channel effects.

Keywords: Neural Networks, OFDM, IQ Data, Channel Measurement, SINR, Software-Defined Radio

## Introduction

Neural network-based receivers are increasingly important research direction, enabling more robust signal demodulation and improving channel estimation. However, practical datasets capturing authentic mobile channel dynamics, particularly under non-line-of-sight conditions, remain scarce. This dataset addresses this gap by providing realistically measured IQ samples collected indoors via mobile SDR antennas moving at pedestrian speeds.

## Measurement Setup

Measurements were conducted using an SDR system operating at a center frequency of 435 MHz. Data acquisition involved a stationary transmitter antenna and a mobile receiver antenna moving indoors and outdoors in non-line-of-sight environments.

Hardware: SDR radio, antennas separated by coaxial cables
Center frequency: 435 MHz
Channel: Real-world indoor fading, measured under practical conditions
OFDM parameters: FFT size 128, 102 active subcarriers, 16-QAM modulation, cyclic prefix 6 samples. 

The OFDM parameters were chosen based on licensing constraints, computational feasibility, and transmit power limitations. The dataset consists of approximately 1000 TTIs, each capturing realistic channel variations, and took around 5 minutes to generate and process using SDR equipment and data processing pipelines.

## Dataset Structure 

The dataset is structured as a PyTorch CustomDataset, facilitating seamless integration into deep learning workflows.

- **File format:** PyTorch `.pth` file containing a pickled instance of `CustomDataset`.
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

| └─   Pilots     |  3rd symbol    | indices 0, 8, 16, ..., 96, 101           |

|-----------------|----------------|------------------------------------------|

|   labels        |  [1400, 4]     | transmitted bits                         |

|-----------------|----------------|------------------------------------------|

|   sinr          |  [1]           | SINR (dB)                                |

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

Figures 1 and 2 illustrate transmitted and received signal PSD. Figures 3, 4, and 5 illustrate the TTI mask, populated TTI, and received TTI after over-the-air transmission and synchronization correspondingly.

### Power Spectral Density

![alt text](https://github.com/rikluost/sdr_ofdm_dataset_v1/blob/main/pics/PSD_TX.png)

Fig 1. PSD of a transmitted TTI.

![alt text](https://github.com/rikluost/sdr_ofdm_dataset_v1/blob/main/pics/PSD_RX.png)

Fig 2. PSD of a received TTI.

### TTI Structure

![alt text](https://github.com/rikluost/sdr_ofdm_dataset_v1/blob/main/pics/OFDM_blockmask.png)

Fig 3. Visualization of the TTI structure. (Purple: unmodulated, green: pilots, yellow: DC)

![alt text](https://github.com/rikluost/sdr_ofdm_dataset_v1/blob/main/pics/OFDM_blockmod.png)

Fig 4. Visualization of fully populated TTI.

![alt text](https://github.com/rikluost/sdr_ofdm_dataset_v1/blob/main/pics/OFDM_block_RX.png)

Fig 5. Visualization of received TTI after over the air transmission and syncronization.


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
# (num_symbols=14, FFT_size=128), includes offsets and DC

print(f'Labels shape: {labels.shape}')          
# ( modulated resource elements=1400, bits_per_TTI=4)

print(f'SINR: {sinr}')                          
# scalar (dB)
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

## References

Paszke, Adam, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, et al. “PyTorch: An Imperative Style, High-Performance Deep Learning Library.” In Advances in Neural Information Processing Systems 32, edited by H. Wallach, H. Larochelle, A. Beygelzimer, F. dAlché-Buc, E. Fox, and R. Garnett, 8024–35. Curran Associates, Inc., 2019. http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf.



