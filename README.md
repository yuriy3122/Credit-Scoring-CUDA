# CUDA MLP Binary Classifier for Mortgage Creditworthiness

This project implements a **CUDA-accelerated Multilayer Perceptron (MLP)** for **binary classification**, designed to estimate the **creditworthiness of loan applicants** using historical HMDA-like datasets and macroeconomic indicators.

It is a **pure C++/CUDA** implementation — no Python or PyTorch required — featuring a one‑hidden‑layer neural network trained directly on GPU with stochastic gradient descent (SGD).

---

## 📘 Table of Contents

1. [Overview](#overview)  
2. [Dataset & Assumptions](#dataset--assumptions)  
3. [Model Architecture](#model-architecture)  
4. [Implementation Details](#implementation-details)  
5. [Build Instructions](#build-instructions)  
6. [Running the Model](#running-the-model)  
7. [Command-Line Arguments](#command-line-arguments)  
8. [Output Example](#output-example)
---

## 🧾 Overview

The model estimates whether a loan application is **creditworthy (1)** or **non‑creditworthy (0)** based on historical applicant data and macroeconomic factors.

It automatically:
- Parses a CSV dataset with headers,
- Detects numeric feature columns (skipping `"Exempt"` / empty values),
- Applies **Min‑Max scaling** per feature,
- Splits into **70% training / 30% testing**,
- Trains an MLP (1 hidden layer, ReLU → Sigmoid),
- Prints final **accuracy** on the test set.

---

## 📊 Dataset & Assumptions

Default expectations (override via CLI):
| Item | Description |
|------|-------------|
| **Target column** | `action_taken` |
| **Positive class** | `1` → Approved / Creditworthy |
| **Negative class** | all other values |
| **Ignored values** | `"Exempt"`, empty cells |
| **Numeric detection** | Automatically selects up to 14 numeric columns |
| **Scaling** | Min‑Max normalization per feature |

> You can change `--target`, `--features`, and RNG `--seed`.

---

## 🧮 Model Architecture

### ASCII Diagram

```
               ┌───────────────────────────────────────────────┐
               │                 Input Features                │
               │      x ∈ ℝ^D  (scaled with Min‑Max)           │
               └───────────────┬───────────────────────────────┘
                               │
                               │  Linear:  Z₁ = W₁·x + b₁   (W₁ ∈ ℝ^{H×D})
                               ▼
                      ┌──────────────────────┐
                      │        ReLU          │
                      │  A₁ = max(0, Z₁)     │
                      └─────────┬────────────┘
                                │
                                │  Linear:  z₂ = W₂·A₁ + b₂  (W₂ ∈ ℝ^{1×H})
                                ▼
                      ┌──────────────────────┐
                      │       Sigmoid        │
                      │    ŷ = σ(z₂) ∈ [0,1] │
                      └─────────┬────────────┘
                                │
                                ▼
                         Prediction ∈ {0,1}
                    (threshold at ŷ ≥ 0.5 → 1)
```

## ⚙️ Implementation Details

All computations are implemented with **custom CUDA kernels**:
| Function | Kernel | Description |
|---------|--------|-------------|
| Forward pass | `matmul`, `add_bias_relu`, `sigmoid_inplace` | Dense layers + activations |
| Backprop | `bce_dz2`, `relu_backward`, `matmul_AT_B` | Gradients for BCE + ReLU |
| Optimizer | `sgd_update` | Parameter updates |
| Utilities | `reduce_cols_mean`, `outer_product`, `scale_inplace` | Reductions & helpers |

> For production speed, swap naive matmuls with cuBLAS GEMM; add mini‑batches and Adam.

---

## 🛠️ Build Instructions

### Requirements
- **CUDA Toolkit 11+**
- **CMake 3.18+**
- **C++17** compiler (GCC/Clang/MSVC)

### Steps
```bash
cd cuda_mlp_binary
mkdir build && cd build
cmake ..
cmake --build . -j
```

---

## 🚀 Running the Model

```bash
./mlp_cuda --csv /path/to/train-hmda-data.csv \
           --epochs 200 \
           --lr 0.05 \
           --hidden 32 \
           --features 14 \
           --target action_taken
```

**Output example**
```
Loaded N=45000 D=14 from train-hmda-data.csv
Epoch 1/200 done
Epoch 25/200 done
...
Test accuracy: 0.864
```

---

## ⚙️ Command-Line Arguments

| Argument | Default | Description |
|---------|---------|-------------|
| `--csv PATH` | *(required)* | Path to input CSV |
| `--target NAME` | `action_taken` | Target column name |
| `--features N` | `14` | Max numeric columns to include |
| `--epochs N` | `200` | Number of training epochs |
| `--hidden H` | `32` | Hidden layer width |
| `--lr LR` | `0.05` | Learning rate (SGD) |
| `--seed S` | `5` | Random seed |

---

## 🧠 Output Example

After training, the model performs inference on the test split and prints:

```
Test accuracy: 0.8653
```
