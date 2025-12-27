# MambaTAD
MambaTAD: When State-Space Models Meet Long-Range Temporal Action Detection

This repository contains the official implementation of **MambaTAD**, our Temporal Action Detection (TAD) method accepted by **IEEE Transactions on Multimedia (TMM)**.  
MambaTAD is a state-space based TAD framework that brings **long-range temporal modeling** and **global feature detection** into a unified, efficient architecture.

---

## 🔑 Key Features

- **Diagonal-Masked Bidirectional State-Space (DMBSS)**
  - Builds on structured state-space models (e.g., Mamba) to capture **long-range temporal context** with **linear computational complexity**.
  - Mitigates **temporal context decay** and **self-element conflict** during global visual context modeling, especially for **long-span action instances**.

- **Global Feature Fusion Detection Head**
  - Progressively refines predictions with **multi-granularity temporal features** and **global awareness**.
  - Enhances the detection of **long-span actions**, where traditional TAD heads often fail due to local receptive fields and weak global context.

- **State-Space Temporal Adapter (SSTA)**
  - Enables **end-to-end, one-stage** TAD with a lightweight temporal adapter.
  - Reduces parameters and computation while retaining the benefits of state-space modeling, achieving **linear complexity** in sequence length.

- **Consistent Performance Across Benchmarks**
  - Extensive experiments demonstrate **superior and robust TAD performance** on multiple public datasets, particularly on videos with long-duration actions.

---

## 🛠 Installation

Please refer to **OpenTAD**’s installation guide for environment setup:

> 👉 [install.md](docs/en/install.md)

Follow that document to:
- Create the conda environment,
- Install dependencies,
- And verify the base TAD framework runs correctly before using MambaTAD.

---

## 🚀 Usage

Please refer to **`usage.md`** for details of training and evaluation scripts:

> 👉 [usage.md](docs/en/usage.md)

You will find:
- Example commands for training MambaTAD with different backbones,
- How to enable DMBSS, the global feature fusion head, and SSTA,
- And how to reproduce the main results reported in our TMM paper.

---

## 🙏 Acknowledgement

Our implementation is built upon the excellent open-source work from:

- **OpenTAD**: A modular and extensible Temporal Action Detection framework  
  🔗 <https://github.com/sming256/OpenTAD>

- **Mamba**: The original Mamba SSM architecture 
  🔗 <https://github.com/state-spaces/mamba>

- **Video Mamba Suite**: Video models based on structured state-space architectures (Mamba)  
  🔗 <https://github.com/OpenGVLab/video-mamba-suite>

We sincerely thank the authors and maintainers of these projects for making their code publicly available.


---

If you find this repository useful, please use the following BibTeX entry for citation.

```latex
@article{lu2025mambatad,
  title={MambaTAD: When State-Space Models Meet Long-Range Temporal Action Detection},
  author={Lu, Hui and Yu, Yi and Lu, Shijian and Rajan, Deepu and Ng, Boon Poh and Kot, Alex C and Jiang, Xudong},
  journal={arXiv preprint arXiv:2511.17929},
  year={2025}
}
```
