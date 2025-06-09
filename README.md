# Measured Radio Channel Dataset for validation of OFDM receivers

## Abstract

This dataset provides realistic IQ samples from OFDM signals measured using Software-Defined Radio (SDR) equipment. Specifically designed for training and evaluating neural network-based wireless receivers, the data includes received IQ symbols post-Discrete Fourier Transform (DFT), associated transmitted bitstreams (labels), and corresponding SINR measurements. Captured in practical indoor and outdoor non-line-of-sight (NLOS) scenarios at pedestrian speeds, this dataset addresses a gap by providing real-world channel dynamics e.g. for robust receiver validation.

Keywords: OFDM, IQ Data, Channel Measurement, SINR, SDR

## Introduction

Neural networks have emerged as powerful tools for enhancing wireless receivers, significantly improving demodulation robustness and accuracy. However, publicly available datasets capturing real-world radio channel conditions, especially under NLOS, remain limited. This dataset fills that gap by offering measured IQ samples collected with mobile SDR equipment in realistic propagation environments.

## Measurement Setup

Data acquisition was conducted using an SDR platform at pedestrian speeds to simulate user mobility.

- Hardware: SDR radio, antennas separated by coaxial cables
- Center frequency: 435 MHz
- Channel: Real-world indoor fading at slow speed, measured under practical conditions.
- OFDM parameters: FFT size 128, 101 subcarriers including DC, 
- Modulation: 16-QAM modulation, 
- Cyclic prefix 6 samples

Each dataset instance captures channel variations across approximately 1000 Transmission Time Intervals (TTIs). The OFDM parameters were chosen based on licensing constraints, computational feasibility, and transmit power limitations. The dataset took around 5 minutes to generate and process using SDR equipment and data processing pipelines.

Time-domain synchronization was achieved by identifying the symbol index with maximum correlation as the starting point for each TTI. SINR values were estimated by comparing the average power of the modulated received symbols against the average power observed in unmodulated signal preluding each modulated TTI.

The selected center-frequency (435 MHz) is constrained by licensing requirements and is relatively low compared to typical real-world wireless communication systems. This lower frequency leads to a longer coherence time and wider coherence bandwidth compared to higher center-frequencies.


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
---------------------------------------------------------------------------------
| Component       | Shape          | Description                                |
|-----------------|----------------|--------------------------------------------|
|  pdsch_iq       |  [14, 128]     | Full DFT output (includes DC, offsets)     |
| ├─   Used       |  [14, 101]     | Active subcarriers (DC & offsets removed)  |
| └─   Pilots     |                | 3rd symbol, indices 0, 8, 16, ..., 96, 101 |
|-----------------|----------------|--------------------------------------------|
|   labels        |  [1400, 4]     | transmitted bits                           |
|-----------------|----------------|--------------------------------------------|
|   sinr          |  [1]           | SINR (dB)                                  |
---------------------------------------------------------------------------------
```

### Pilot Information
- Pilots are present only in the 3rd OFDM symbol (index 2).
- Pilots are placed on every 8th subcarrier, **plus the last subcarrier (index 101) is always included as a pilot** for full-band coverage.  
  - i.e. pilot indices are: 0, 8, 16, ..., 96, 101
- pilot signals: `[-0.7-0.7j, -0.7+0.7j,  0.7-0.7j,  0.7+0.7j,
        -0.7-0.7j, -0.7+0.7j,  0.7-0.7j,  0.7+0.7j,
        -0.7-0.7j, -0.7+0.7j,  0.7-0.7j,  0.7+0.7j,
        -0.7-0.7j, -0.7+0.7j]`

## Dataset Visualizations

### Transmission

Figure 1 illustrates the structure of the OFDM Transmission Time Interval (TTI) block, comprising 128 frequency-domain subcarriers, of which 101 are modulated. The remaining subcarriers—including the DC component (unmodulated)—are not used for data transmission. In the time-domain dimension, each TTI consists of 14 OFDM symbols, as shown in the visualization. Pilot subcarrier positions, essential for channel estimation, are highlighted in green. Within the provided code context, this representation is referred to as the "OFDM-mask."


![alt text](https://github.com/rikluost/sdr_ofdm_dataset_v1/blob/main/pics/OFDM_blockmask.png)

Fig 1. Visualization of the TTI structure. (Purple: unmodulated, green: pilots, yellow: DC)

Few calculations to help verify correct processing of the data:

- n_s is the number of symbols in TTI = 128 * 14 = 1792
- n_offset is the number of unmodulated subcarriers on both sides of the signal = 13
- n_subcarriers is the number of modulated subcarriers = n_s-(2*n_offset) = 102 
- n_DC is the number of DC symbols = 14
- n_pilots is the number of pilots = 14
- n_mod_symbols is the number of modulated symbols in TTI = 102 * 14 - n_pilots - n_DC = 1400
- Qm is the number of bits per symbol = 4 (i.e. 16-QAM)
- n_bits is the number of bits in a TTI = n_mod_symbols * Qm = 1400 * 4 = 5600 

The OFDM-mask is populated with known pilot symbols and the modulated bitstream. Figure 2 visualizes an OFDM block fully prepared and ready for transmission, indicating both modulated and unmodulated subcarriers, DC, and embedded pilots within the structure.

![alt text](https://github.com/rikluost/sdr_ofdm_dataset_v1/blob/main/pics/OFDM_blockmod.png)

Fig 2. Visualization of fully populated TTI.

After populating the TTI and applying the Inverse Fast Fourier Transform (IFFT), the resulting time-domain data is normalized to meet the requirements of the SDR hardware. Figure 3 displays the Power Spectral Density (PSD) of the normalized transmit signal, illustrating the spectral characteristics of the transmitted OFDM waveform.

![alt text](https://github.com/rikluost/sdr_ofdm_dataset_v1/blob/main/pics/PSD_TX.png)

Fig 3. PSD of a transmitted TTI.

### Reception

Upon reception, the received signal undergoes synchronization followed by a Discrete Fourier Transform (DFT). The resulting frequency-domain representation of the received TTI is illustrated in Figure 4, highlighting the effects of the radio channel and synchronization accuracy.

![alt text](https://github.com/rikluost/sdr_ofdm_dataset_v1/blob/main/pics/OFDM_block_RX.png)

Fig 4. Visualization of received TTI after over the air transmission and syncronization.

Figure 5 shows the Power Spectral Density (PSD) of the received time-domain signal, illustrating channel-induced distortions in frequency domain after over-the-air transmission.

![alt text](https://github.com/rikluost/sdr_ofdm_dataset_v1/blob/main/pics/PSD_RX.png)

Fig 5. PSD of a received TTI.

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
> PDSCH IQ shape: [14,128]      
# (num_symbols=14, FFT_size=128), includes offsets and DC

print(f'Labels shape: {labels.shape}')          
> Labels shape: [1400,4]
# ( modulated resource elements=1400, bits_per_TTI=4)

print(f'SINR: {sinr}')                          
> SINR: 23
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
