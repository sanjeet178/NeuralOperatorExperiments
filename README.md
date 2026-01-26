# NeuralOperatorExperiments

This repository contains experiments and utilities for studying **Neural Operators**, with a primary focus on **Fourier Neural Operators (FNOs)** and related architectures for learning solution operators of PDEs.

---

## 📁 Repository Structure

```
NeuralOperatorExperiments/
│
├── modules.py          # Core model definitions (includes FNO)
├── utils.py            # Helper functions (dataloaders....)
├── postprocessing.py   # Creation of result plots
├── main.py             # Training/Inferencing scripts 
├── data/               # prepared datasets
├── modelParams/        # model learnable parameters
├── results/            # results/plots stored here
└── README.md           # Project documentation
```

> **Note:** The `FNO` model implementation is located in **`modules.py`**.

---

## 🚀 Models Implemented

### Fourier Neural Operator (FNO)

The **Fourier Neural Operator** learns mappings between infinite-dimensional function spaces by:

* Lifting input functions to a higher-dimensional latent space
* Applying **spectral convolutions** in the Fourier domain
* Using inverse FFTs to return to the physical domain

This allows the model to efficiently capture **global dependencies** in PDE solutions.

Key features:

* FFT-based convolution layers
* Resolution-invariant learning
* Suitable for physics-based problems (Darcy flow, Navier–Stokes, heat equation, etc.)

---

## 🧠 Motivation

Traditional neural networks struggle to generalize across discretizations and resolutions. Neural operators address this by directly learning the **operator** mapping:

[
\mathcal{G}: a(x) \mapsto u(x)
]

where `a(x)` is an input function (e.g., coefficients, forcing terms) and `u(x)` is the solution field.

---

## 🤝 Contributions

Contributions, issues, and experiment ideas are welcome. Please open a pull request or issue for discussion.

---

**Author:** Sanjeet
