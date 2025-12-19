# HAR-Bidirectional-RC: Human Action Recognition via Bidirectional Reservoir Computing

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of the Bidirectional Reservoir Computing (Bi-RC) framework for high-efficiency Skeleton-based Human Action Recognition.

## 🚀 Overview

This repository provides a modular, computationally lightweight framework for human action recognition using skeleton data. By decoupling spatiotemporal feature extraction from non-linear mapping, the system achieves state-of-the-art efficiency (up to 95% reduction in training cost) while maintaining competitive accuracy on large-scale benchmarks.

### Key Features
- **Bidirectional Reservoir Computing**: Captures full temporal context without the vanish gradient issues of backpropagation-based RNNs.
- **Adaptive Dimension Reduction**: Hybrid approach using PCA and **Tucker Decomposition** for multilinear rank preservation.
- **Advanced Readout**: High-dimensional projection using **Maxout** neural layers for superior discriminative power.
- **Ultra-Efficient**: Designed for edge deployment with $\mathcal{O}(1)$ training complexity relative to sequence steps.

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/haythemghz/HAR-Bidirectional-RC.git
   cd HAR-Bidirectional-RC
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📂 Data Preparation

The framework is optimized for public skeleton datasets. Ensure your data is organized as per the following structure:

```text
/data
  /NTU-RGBD
  /UTD-MHAD
  /CZU-MHAD
```

Refer to `data_utils.py` for specific loading configurations for each dataset.

---

## 🧪 Usage

### Training
To train the model on a specific dataset:
```bash
python train.py --data_path ./path/to/dataset --reservoir_size 1000 --fusion_strategy concat
```

### Evaluation
Use the example script to test the model on a pre-trained checkpoint:
```bash
python example_usage.py --checkpoint ./checkpoints/best_model.pth
```

---

## 📊 Performance Benchmarks

| Dataset | Metric | Bi-RC Accuracy |
| :--- | :--- | :--- |
| **UTD-MHAD** | Accuracy | 98.9% |
| **CZU-MHAD** | Accuracy | 95.7% |
| **NTU-60 (X-Sub)** | Accuracy | 93.2% |
| **Kinetics-Skeleton** | Accuracy | 38.7% |

---

## 📄 Citation

If you use this code or framework in your research, please cite our paper:

```bibtex
@article{ghazouani2025bidirectional,
  title={Bidirectional Reservoir Computing for High-Efficiency Skeleton-Based Human Action Recognition},
  author={Ghazouani, Haythem and others},
  journal={Journal Name},
  year={2025}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
