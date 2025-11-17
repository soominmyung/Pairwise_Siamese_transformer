# 🧠 Pairwise Preference Learning with a Siamese Transformer  
### Predicting Financial Position Superiority under Identical Market Conditions

This project explores a **pairwise preference learning** formulation for financial time-series data using a **Siamese Transformer architecture**.  
Rather than predicting labels from isolated samples, the model compares **two candidate positions under the same market snapshot** and learns which candidate performs better.

This approach leverages **comparative representation learning**, **Transformer-based sequence modelling**, and **systematic experimentation** to reach **Kaggle Test Accuracy: `0.81`**.

---

## ⚡ Key Highlights

- **Siamese Transformer Encoder**  
  Shared weights ensure both candidates are encoded using the *exact same representational geometry*, enabling fair, consistent comparison.

- **Comparative Representation Learning**  
  The model learns how each position behaves in the same market environment, relying on the **absolute difference** between embeddings to capture the preference signal.

- **Financial-Time-Series-Aware Input Encoding**  
  Transformer encoder learns complex interactions between OHLCV features and the position taken in that context.

- **Generalisation-Focused Model Selection**  
  The best-performing model did not have the lowest loss.  
  Instead, it produced the **strongest decision boundary**, demonstrating deeper models can generalise better despite higher variance.

- **Reproducible and Modular ML Pipeline**  
  Including feature preparation, model definition, training loops, early stopping, and experiment tracking.

---

## 📂 Dataset Summary

Each data sample contains:

- **Option A** — market features + position  
- **Option B** — market features + position  
- **Label** — which option is superior (`0 = A`, `1 = B`)

Because both options share the **same market conditions**, the problem is naturally framed as:

> **“Which position is better under identical market states?”**

This makes it an ideal case for **pairwise preference learning** using a **Siamese encoder**.

---

## 🧠 Approach Overview

### 1. Comparative Feature Engineering
Although raw OHLCV values are available, meaningful patterns arise from position–market interactions.  
Feature steps included:

- Log returns and normalised values  
- Rolling statistics (mean, volatility, gradients)  
- Temporal differences  
- Combined market + position input vectors  
- Optional differential features `(A - B)` to strengthen comparative signals  

These features enable the encoder to learn **how a position behaves relative to the market**, not just the raw numbers.

---

### 2. Siamese Transformer Architecture

Two branches share the **same Transformer encoder**, ensuring consistent representation:

```
A_input ──► Shared Transformer ──► rep_A
B_input ──► Shared Transformer ──► rep_B
```

A comparison head computes:

```
z = |rep_A - rep_B|
```


Absolute difference emphasises the **magnitude of preference** while ignoring irrelevant directional noise.

Finally, a fully connected layer outputs the probability that **B is superior**.

---

### 3. Hyperparameter Experiments

Investigated:

- Transformer depth (1–4 layers)  
- Embedding size (32–256)  
- Dropout levels  
- Optimisers (AdamW, Ranger)  
- Learning rate schedules  
- Pooling strategies (mean vs attention)

**Best overall configuration:**

- 2-layer Transformer  
- 128-d embeddings  
- AdamW (LR: `3e-4`)  
- Dropout 0.2  
- Absolute difference comparison  

Interestingly, deeper models (e.g., 8-layer variants) showed:

- **Higher loss** (variance ↑)  
- **Higher accuracy** (bias ↓ → better boundary)

Consistent with the **bias–variance trade-off** in deep Transformers.

---

## 📊 Results

| Model | Validation Loss | Public Accuracy | Notes |
|-------|------------------|------------------|--------|
| Model 6 | Lowest | 0.77 | High confidence but weaker boundary |
| **Model 8** | Slightly higher | **0.81** | **Best generalisation** |

The deeper model captured more expressive market–position interactions, producing a stronger decision boundary even with a slightly higher loss.

---

## ⚙️ Tech Stack

- **Python 3**  
- **PyTorch** (Transformer encoder, Siamese comparison)  
- **NumPy / Pandas**  
- **Scikit-learn**  
- **Matplotlib / Seaborn**  
- Experiment logging & Git version control  

---

## 🚀 Why This Project Matters

This project showcases:

### ✔ Real-World ML Problem Framing  
Reformulating a financial prediction problem into **pairwise preference learning** rather than simple classification.

### ✔ Custom Deep Architecture Design  
Building a **Siamese Transformer** tailored for comparative decisions.

### ✔ Deep Understanding of Representation Learning  
Learning robust market–position embeddings under identical contexts.

### ✔ Strong Experimentation Discipline  
Hyperparameter sweeps, multiple model checkpoints, and nuanced interpretation (boundary vs confidence).

### ✔ Focus on Generalisation  
Selecting models based on **boundary stability**, not minimum validation loss.

---

## 📈 Final Remarks

This project reflects the ability to:

- Identify the intrinsic structure of a problem  
- Build architectures that respect that structure  
- Engineer meaningful features for sequence models  
- Diagnose loss–accuracy divergence  
- Prioritise robust generalisation over superficial metrics  

It highlights a practical, research-minded approach to deep learning for financial sequence analysis.

---

If you'd like to see the full training notebook or architecture diagrams, they are included in the repository.
