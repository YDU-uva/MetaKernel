# MetaKernel: Learning Variational Random Features with Limited Labels

This repository contains the implementation of **MetaKernel**, an enhanced version of Meta Variational Random Features (MetaVRF) that incorporates **conditional normalizing flows** for improved few-shot learning performance.

## Paper Reference

Based on: **"MetaKernel: Learning Variational Random Features with Limited Labels"** https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9722994

The MetaKernel approach extends the original MetaVRF framework by introducing conditional normalizing flows to generate richer and more informative posterior distributions over random Fourier feature bases.



## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Support Set   │    │   Query Set      │    │  LSTM Context   │
│   Features      │    │   Features       │    │  Encoding       │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          └──────────┬───────────┴───────────────────────┘
                     │
          ┌─────────────────────────────────┐
          │     MetaKernel Inference        │
          │  ┌─────────────────────────────┐│
          │  │  Base Gaussian Posterior    ││
          │  └─────────────┬───────────────┘│
          │                │                │
          │  ┌─────────────▼───────────────┐│
          │  │ Conditional Normalizing     ││
          │  │ Flow (Coupling Layers)      ││
          │  └─────────────┬───────────────┘│
          └────────────────┼────────────────┘
                          │
          ┌───────────────▼─────────────────┐
          │   Enhanced Random Features      │
          │   (Flow-transformed samples)    │
          └───────────────┬─────────────────┘
                          │
          ┌───────────────▼─────────────────┐
          │    Kernel Computation &         │
          │    Classification               │
          └─────────────────────────────────┘
```

## Installation and Setup

1. **Prerequisites**:
   ```bash
   python 3.6+
   tensorflow 1.x
   numpy
   matplotlib
   scikit-image
   ```

2. **Download the repository**:
   ```bash
   git clone https://github.com/YDU-uva/MetaKernel.git
   cd MetaKernel
   ```

3. **Prepare datasets** (see original MetaVRF instructions in `data/`)

## Usage

### Running MetaKernel with Conditional Normalizing Flows

```bash
# Train and test MetaKernel on miniImageNet (1-shot 5-way)
python src/classification/run_metakernel_classifier.py \
    --dataset miniImageNet \
    --mode train_test \
    --shot 1 \
    --way 5 \
    --use_flow True \
    --num_flow_layers 4 \
    --flow_hidden_size 128 \
    --iterations 60000
```

### Running Original MetaVRF (Baseline)

```bash
# Train and test original MetaVRF for comparison
python src/classification/run_classifier.py \
    --dataset miniImageNet \
    --mode train_test \
    --shot 1 \
    --way 5 \
    --iterations 60000
```

### Demo Script for Comparison

```bash
# Run comparison between MetaKernel and MetaVRF
python src/classification/demo_metakernel.py \
    --mode compare \
    --dataset miniImageNet \
    --iterations 1000

# Run ablation study on number of flow layers
python src/classification/demo_metakernel.py \
    --mode ablation \
    --dataset miniImageNet \
    --iterations 500
```

## Key Parameters

### MetaKernel-Specific Parameters

- `--use_flow`: Enable/disable conditional normalizing flows (default: True)
- `--num_flow_layers`: Number of coupling layers (default: 4)
- `--flow_hidden_size`: Hidden size for flow networks (default: 128)
- `--flow_weight`: Weight for flow regularization (default: 0.01)

### Standard Parameters (inherited from MetaVRF)

- `--dataset`: Choose from 'Omniglot', 'miniImageNet', 'tieredImageNet', 'cifarfs'
- `--shot`: Number of support examples per class
- `--way`: Number of classes in few-shot tasks
- `--d_theta`: Feature extractor output size (default: 256)
- `--d_rn_f`: Random feature base size (default: 512)

## File Structure

```
src/classification/
├── normalizing_flows.py          # Conditional normalizing flow implementation
├── inference.py                  # Enhanced inference networks (includes MetaKernel)
├── run_metakernel_classifier.py  # Main MetaKernel training/testing script
├── demo_metakernel.py            # Demo and comparison script
├── run_classifier.py             # Original MetaVRF script
├── utilities.py                  # Utility functions
├── features.py                   # Feature extraction networks
└── data.py                       # Data loading utilities
```

## Citation

If you use this MetaKernel implementation, please cite both the original MetaVRF paper and the MetaKernel extension:

```bibtex
@misc{du2021metakernel,
    title={MetaKernel: Learning Variational Random Features with Limited Labels},
    author={Yingjun Du and Haoliang Sun and Xiantong Zhen and Jun Xu and Yilong Yin and Ling Shao and Cees G. M. Snoek},
    year={2021},
    eprint={2105.03781},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}

@inproceedings{zhen2020learning,
    title={Learning to Learn Kernels with Variational Random Features},
    author={Xiantong Zhen and Haoliang Sun and Yingjun Du and Jun Xu and Yilong Yin and Ling Shao and Cees Snoek},
    booktitle={International Conference on Machine Learning},
    pages={11409--11419},
    year={2020},
    organization={PMLR}
}
```
