# Pairwise Preference Learning with Siamese Transformers  
Behavioural Modelling for Financial Strategy Comparison

This project implements a **Siamese Transformer architecture** to model **behavioural differences between two financial time-series strategies under identical market conditions.**  
The goal is to learn which strategy performs better *given the same context and the same market environment*, using representation learning rather than classical performance metrics.

The final model achieved **0.81 accuracy** on the Kaggle test set.

<img width="2335" height="1084" alt="Siamese" src="https://github.com/user-attachments/assets/26644f8e-bf9f-453a-b4e0-ce70ed68e702" />

---

## Project Motivation

Financial strategies exhibit **behavioural patterns** rather than simple numeric relationships.  
Instead of predicting future prices, this task focuses on:

- **How two strategies behave under the same market conditions**
- **How their relative positioning and reactions differ**
- **Whether those behavioural differences lead to better performance**

This approach aligns with behaviour-driven investment modelling, where the target is not price prediction but **decision behaviour representation**.

---

## Model Architecture

### 1. **Sequence Encoder**  
Each input sequence (OHLCV + engineered features) is processed by an **8-layer Transformer Encoder**:

- `MODEL_DIM = 256`  
- `N_LAYERS = 8`  
- `N_HEADS = 4`  
- `DROPOUT = 0.1`  
- LayerNorm, residual connections  
- Mean pooling for sequence embedding  

The encoder produces a **behavioural embedding** for each strategy:

```
z_A = Encoder(seq_A)
z_B = Encoder(seq_B)
```


### 2. **Siamese Design**  
Both encoders share weights (true Siamese):

- Ensures both strategies are mapped into the *same behavioural space*
- Reduces parameter count  
- Encourages meaningful comparison  

### 3. **Comparison Head**

The comparison vector is:


```
[z_A , z_B , |z_A - z_B|]
```


This captures:

- Absolute behaviour of each strategy  
- Relative behaviour  
- Behavioural disagreement  

Then a 3-layer MLP predicts which strategy performs better:

```
Linear(768 → 256) → ReLU → Dropout
Linear(256 → 128) → ReLU
Linear(128 → 1) → Sigmoid
```


---

## Feature Engineering

A systematic feature engineering process was performed to understand which types of signals help the Transformer learn stable behavioural patterns for the pairwise comparison task.

Although the final model uses a minimal subset of features, I tested several categories of financial indicators during experimentation.

### Profitability & Duration Features
These features capture how the strategy behaves relative to market movement and holding time.

- **Log Return**  
  Standardised return measure, scale-independent.

- **Proxy Return**  
  A smoothed transformation approximating micro-momentum and directional tendency.

- **Time in Position**  
  Measures how long a strategy maintains exposure, reflecting behavioural persistence.

### Momentum & Trend Indicators  
*(Tested but later removed due to added noise with attention layers)*  
:contentReference[oaicite:0]{index=0}

- **MACD (Moving Average Convergence Divergence)**  
- **MACD Signal Line**  
- **RSI (Relative Strength Index)**  

These improved directional sensitivity but introduced instability when combined with limited sequence lengths.  
Transformers over-reacted to short-term swings when these indicators were included.

### Volatility Indicators  
*(ATR & Historical Volatility were also tested)*  
:contentReference[oaicite:1]{index=1}

- **ATR (Average True Range)**  
- **Historical Volatility**  

Useful for regime awareness, but in this pairwise setting they amplified noise and reduced validation accuracy.

### Final Selected Features
After evaluating multiple combinations, the final model uses a **lean, stable subset** that produced the strongest behavioural embeddings:

- **Open, High, Low, Close**
- **Volume**
- **Position**
- **Log Return**
- **Proxy Return**

This combination offered the best trade-off between signal richness and stability, avoiding overfitting and keeping attention patterns focused on core behaviour rather than noisy microstructure effects.

### Normalisation
All numerical features were **z-score normalised per sequence**, ensuring that attention compares relationships rather than absolute magnitudes.



---

## Training

- **Loss:** Binary Cross-Entropy (log-loss)  
- **Optimizer:** Adam (`lr = 1e-4`)  
- **Early Stopping:** patience 10  
- **Batch size:** 32  
- **Epochs:** up to 50  

The chosen learning rate is conservative and stable for attention-based models.

---

## Results

| Metric | Result |
|--------|--------|
| **Kaggle Test Accuracy** | **0.81** |
| Validation Loss | Smooth convergence |
| Embedding Separation | Clear behavioural clustering |

The model successfully captured:

- Regime-dependent behaviour  
- Different positioning reactions between strategies  
- Long-range dependency patterns across market contexts  

---

## Key Insights

### 1. Behavioural patterns matter  
Transformers could detect relationships between:

- Volatility  
- Returns  
- Position changes  
- Cross-time interactions  

This goes beyond simple technical indicators.

### 2. Siamese works because inputs share the same data structure  
Both strategies are **two versions of the same behavioural language**.  
Weight sharing forces a comparable embedding space.

### 3. Attention > recurrence  
Unlike LSTM/RNN:

- No noise accumulation  
- Non-local interactions are directly modelled  
- Multi-scale behaviour captured effectively  

### 4. Feature engineering improved embedding quality  
Adding log returns & proxy returns significantly stabilised learning.

---

## Repository Structure

```
.
├── train/                  # Excel files for training pairs
│   ├── sample_0_a.xlsx
│   ├── sample_0_b.xlsx
│   ├── sample_1_a.xlsx
│   ├── sample_1_b.xlsx
│   └── ...
├── test/                   # Excel files for test pairs
│   ├── sample_0_a.xlsx
│   ├── sample_0_b.xlsx
│   └── ...
├── train.csv               # id, file names, and pairwise labels (0/1)
├── test.csv                # id and file names only (no labels)
├── Siamese_transformer.ipynb
└── README.md
```

---

## Key Stack

**Deep Learning**
- PyTorch  
- Siamese Networks  
- Transformer Encoder (8 layers, Multi-Head Attention)  
- MLP Classifier  
- Dropout / LayerNorm / Residual Connections  

**Financial Time-Series Engineering**
- OHLCV processing  
- Log Returns  
- Proxy Returns  
- Time-in-Position  
- MACD / RSI / ATR / Historical Volatility (experimentation)

**Data Handling & Preprocessing**
- Pandas  
- NumPy  
- Z-score normalisation  
- Sliding windows / sequence batching  
- Excel ingestion + dynamic feature parsing  

**Training & Evaluation**
- Adam Optimiser (lr=1e-4)  
- Early Stopping  
- Binary Cross-Entropy  
- Validation loss tracking  
- Kaggle evaluation pipeline (0.81 accuracy)

**Environment**
- Python 3  
- Jupyter Notebook  

---

## Future Improvements

- Regime-aware encoder (volatility-conditioning)
- Contrastive learning pretraining  
- Behavioural clustering of strategies  
- Attention visualisation for interpretability  
- Multi-modal features (macro, sentiment, volume profile)  

---

## Notes

This project demonstrates that:

> **Representation learning is a powerful tool for comparing strategy behaviour,  
not just for prediction.**

It aligns with the idea that successful investing involves  
**understanding behavioural fingerprints**,  
not just forecasting numbers.
