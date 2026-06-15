# ConvLSTM Smoke Plume Forecasting Model
### `convlstm_functional_v1.pt` — Technical & Operational Guide

---

## What does this model do?

This AI model predicts **where airborne smoke (or any atmospheric pollutant) will be one hour from now**, given three consecutive hours of observations.

It processes a 64 × 64 km grid of sensor/simulation data and outputs a map of predicted smoke concentration for the next time step. The intended use is in emergency response, air quality forecasting, and environmental monitoring.

---

## At a glance

| Property | Value |
|---|---|
| File | `src/plume/convlstm_functional_v1.pt` |
| Model type | Functional ConvLSTM (PyTorch) |
| Trained on | 2,000 HYSPLIT plume simulation windows |
| Input | 3 hourly frames × 10 channels × 64 × 64 grid |
| Output | Predicted plume concentration map (64 × 64) |
| Parameters | 538,145 (~2.1 MB) |
| Training epochs | 15 |
| Final validation MSE | 0.097 |
| Final test MSE (all pixels) | 0.125 |
| Final test MSE (plume region only) | 0.321 |
| Framework | PyTorch ≥ 2.0 |

---

## How the model works (non-technical summary)

The model uses an architecture called a **Convolutional Long Short-Term Memory network (ConvLSTM)**. It combines two ideas:

- **Convolutional neural networks (CNN)** — the same technology used in image recognition. Here, the CNN reads each hourly frame and extracts spatial patterns: where is the plume concentrated, where are the edges, how is it shaped?

- **Long Short-Term Memory (LSTM)** — a type of AI memory cell that learns how things change over time. It remembers how the plume moved between hour 1, hour 2, and hour 3, and uses that to predict hour 4.

The result is a model that understands both the *shape* of the plume and its *direction of travel*, allowing it to make physically informed predictions.

```
3 observed hours (past) → AI model → 1 predicted hour (future)

  Hour 1 (64×64) ─┐
  Hour 2 (64×64) ──►  ConvLSTM  ──►  Hour 4 prediction (64×64)
  Hour 3 (64×64) ─┘
```

---

## Input data specification

### What goes in

Each input is a **sliding time window** from a `.npz` file containing:

| Array | Shape | Description |
|---|---|---|
| `input` | (3, 10, 64, 64) | 3 consecutive hourly frames, 10 channels, 64 × 64 km grid |
| `target` | (1, 10, 64, 64) | The ground-truth next hour (used for evaluation only) |

### The 10 input channels

| Channel | Name | Unit | Description |
|---|---|---|---|
| 0 | `plume_concentration` | — | Smoke / tracer density (the value being predicted) |
| 1 | `u10m_ms` | m/s | Eastward wind at 10 m height |
| 2 | `v10m_ms` | m/s | Northward wind at 10 m height |
| 3 | `wspd10_ms` | m/s | Wind speed at 10 m |
| 4 | `wdir_sin` | — | Wind direction — sine component |
| 5 | `wdir_cos` | — | Wind direction — cosine component |
| 6 | `pblh_m` | m | Planetary boundary-layer height |
| 7 | `sfcp_hpa` | hPa | Surface pressure |
| 8 | `rh2m_pct` | % | Relative humidity at 2 m |
| 9 | `t02m_k` | K | Air temperature at 2 m |

### Input normalisation (mandatory)

The model was trained on **z-score normalised** inputs. Raw values **must** be normalised before feeding the model or predictions will be wrong. The normalisation statistics are stored inside the checkpoint itself:

```python
ch_mean = checkpoint['normalisation']['ch_mean']
ch_std  = checkpoint['normalisation']['ch_std']
```

**Per-channel statistics used during training:**

| Channel | Mean | Std |
|---|---|---|
| plume_concentration | 0.332 | 0.810 |
| u10m_ms | 1.558 | 4.238 |
| v10m_ms | 1.303 | 4.539 |
| wspd10_ms | 5.582 | 3.394 |
| wdir_sin | -0.219 | 0.686 |
| wdir_cos | -0.158 | 0.676 |
| pblh_m | 581.2 | 489.5 |
| sfcp_hpa | 1012.4 | 10.27 |
| rh2m_pct | 77.6 | 14.66 |
| t02m_k | 284.3 | 6.34 |

### Output

- Shape: `(batch_size, 1, 64, 64)` — one plume concentration map per sample
- Values: always ≥ 0 (guaranteed by Softplus activation)
- Units: same scale as `plume_concentration` in the input (raw, un-normalised)

---

## Architecture details

The model is implemented as a pure functional forward pass — no class inheritance, just a parameter dictionary and two functions. This makes it easy to inspect, port, and audit.

```
Input  (B, 3, 10, 64, 64)   — B = batch size, 3 time steps, 10 channels
    │
    │  repeated for t = 0, 1, 2
    ▼
┌──────────────────────────────────────┐
│  Encoder (per frame)                 │
│  Conv2d(10 → 32, kernel 3×3)         │
│  + GroupNorm(8 groups)  + ReLU       │
└──────────────────┬───────────────────┘
                   │ (B, 32, 64, 64)
                   ▼
┌──────────────────────────────────────┐
│  ConvLSTM Layer 1  (hidden = 64)     │
│  Gate conv: (32+64) → 4×64, 3×3     │
│  221,184 parameters                  │
└──────────────────┬───────────────────┘
                   │ (B, 64, 64, 64)
                   ▼
┌──────────────────────────────────────┐
│  ConvLSTM Layer 2  (hidden = 64)     │
│  Gate conv: (64+64) → 4×64, 3×3     │
│  294,912 parameters                  │
└──────────────────┬───────────────────┘
                   │ final hidden state after 3 frames
                   ▼
┌──────────────────────────────────────┐
│  Decoder                             │
│  Conv2d(64 → 32, 3×3) + GroupNorm   │
│  Conv2d(32 → 1, 1×1) + Softplus     │
└──────────────────┬───────────────────┘
                   ▼
Output  (B, 1, 64, 64)  — plume concentration at t+1
```

**Parameter breakdown:**

| Layer | Parameters |
|---|---|
| Encoder (Conv + GroupNorm) | 2,976 |
| ConvLSTM Layer 1 | 221,440 |
| ConvLSTM Layer 2 | 295,168 |
| Decoder (Conv + GroupNorm + output) | 18,561 |
| **Total** | **538,145** |

---

## Performance metrics

Trained for 15 epochs on 2,000 windows; evaluated on 200 held-out test windows.

| Metric | Value | Notes |
|---|---|---|
| Train MSE (epoch 15) | 0.0955 | Loss on training data |
| Validation MSE (epoch 15) | 0.0973 | Loss on unseen validation data |
| Test MSE — all pixels | 0.1252 | Includes background (many zeros) |
| Test RMSE — all pixels | 0.3539 | In original concentration units |
| Test MAE — all pixels | 0.1623 | Average absolute error per pixel |
| Test MSE — plume pixels only | 0.3205 | Where actual smoke is present (harder) |

**Training convergence:**

| Epoch | Train MSE | Val MSE |
|---|---|---|
| 1 | 0.27469 | 0.18034 |
| 5 | 0.13508 | 0.12597 |
| 10 | 0.10566 | 0.10563 |
| 15 | 0.09553 | 0.09726 |

The close gap between training and validation loss shows the model is generalising well (not overfitting).

---

## How to load and run the model

### Requirements

```
python >= 3.10
torch  >= 2.0
numpy  >= 1.24
```

### Step 1 — Define the two functions

These functions are the entire model. Copy them into your script:

```python
import torch
import torch.nn.functional as F

PAD = 1   # padding for 3×3 convolutions

def convlstm_step(x, h, c, w, b):
    """One time step of a ConvLSTM cell."""
    combined = torch.cat([x, h], dim=1)
    gates    = F.conv2d(combined, w, b, padding=PAD)
    hid_ch   = gates.shape[1] // 4
    gi, gf, gg, go = gates.split(hid_ch, dim=1)
    new_c = torch.sigmoid(gf) * c + torch.sigmoid(gi) * torch.tanh(gg)
    new_h = torch.sigmoid(go) * torch.tanh(new_c)
    return new_h, new_c


def forward(x, p):
    """
    x : (B, 3, 10, 64, 64)  — normalised input
    p : dict of torch.nn.Parameter
    returns (B, 1, 64, 64)  — predicted plume concentration
    """
    B, T, C, H, W = x.shape
    cfg = {'GN_GRPS': 8, 'HID_CH': 64}

    h1 = torch.zeros(B, cfg['HID_CH'], H, W, device=x.device)
    c1 = torch.zeros(B, cfg['HID_CH'], H, W, device=x.device)
    h2 = torch.zeros(B, cfg['HID_CH'], H, W, device=x.device)
    c2 = torch.zeros(B, cfg['HID_CH'], H, W, device=x.device)

    for t in range(T):
        enc = F.conv2d(x[:, t], p['enc_w'], p['enc_b'], padding=PAD)
        enc = F.group_norm(enc, cfg['GN_GRPS'], p['enc_gn_w'], p['enc_gn_b'])
        enc = F.relu(enc)
        h1, c1 = convlstm_step(enc, h1, c1, p['lstm1_w'], p['lstm1_b'])
        h2, c2 = convlstm_step(h1,  h2, c2, p['lstm2_w'], p['lstm2_b'])

    out = F.conv2d(h2, p['dec1_w'], p['dec1_b'], padding=PAD)
    out = F.group_norm(out, cfg['GN_GRPS'], p['dec1_gn_w'], p['dec1_gn_b'])
    out = F.relu(out)
    out = F.conv2d(out, p['dec2_w'], p['dec2_b'])
    return F.softplus(out)
```

### Step 2 — Load the checkpoint

```python
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpt = torch.load('src/plume/convlstm_functional_v1.pt',
                  map_location=device, weights_only=False)

# All learned weights
p = {k: v.to(device) for k, v in ckpt['model_state'].items()}

# Normalisation statistics — MUST be used on every input
ch_mean = ckpt['normalisation']['ch_mean']   # shape (10,)
ch_std  = ckpt['normalisation']['ch_std']    # shape (10,)
```

### Step 3 — Run a prediction

```python
# Load one data window
data = np.load('EDA/data/windows/000003_000.npz')
x_raw = data['input'].astype('float32')   # (3, 10, 64, 64)

# Normalise (mandatory)
mean_bc = ch_mean[:, None, None]   # broadcast over H, W
std_bc  = ch_std[:, None, None]
x_norm  = (x_raw - mean_bc) / std_bc

# Run model
x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)  # (1, 3, 10, 64, 64)

with torch.no_grad():
    prediction = forward(x_tensor, p)   # (1, 1, 64, 64)

plume_next_hour = prediction[0, 0].cpu().numpy()   # (64, 64) — your result
print(f"Predicted concentration range: {plume_next_hour.min():.4f} – {plume_next_hour.max():.4f}")
```

---

## What is stored inside the `.pt` file

```python
checkpoint = {
    'model_state':    {...},   # 14 weight tensors — the trained model
    'config':         {...},   # architecture hyperparameters (IN_CH, HID_CH, etc.)
    'normalisation':  {...},   # ch_mean, ch_std — needed to normalise input
    'metrics':        {...},   # MSE, RMSE, MAE recorded at end of training
    'train_losses':   [...],   # MSE per epoch (15 values)
    'val_losses':     [...],   # validation MSE per epoch (15 values)
}
```

Everything needed to reproduce results or continue training is self-contained in this single file.

---

## Limitations and known issues

| Limitation | Impact |
|---|---|
| Trained on only 2,000 windows (of 40,000+ available) | Model has not seen the full dataset diversity — accuracy will improve with more training data |
| Predicts only 1 step ahead | For longer-range forecasts, use `best_full_checkpoint.pt` instead (predicts 4 steps) |
| Stage 1 training only (no physics loss) | No explicit conservation laws enforced — mass balance error ~27%, background false-positive rate is high (1.0) |
| 64 × 64 km fixed grid | Cannot be applied directly to different spatial resolutions without retraining |
| Plume-region MSE (0.321) is 2.5× higher than global MSE (0.125) | The model is more accurate on background (zero) pixels than on the active plume |

---

## How to continue training or improve the model

### Option 1 — Train on more data

The full dataset has 40,215 windows. The current model used only 2,000. Re-running the training notebook with a larger sample is the single easiest improvement:

- Open [EDA/EDA_final_model.ipynb](EDA/EDA_final_model.ipynb)
- In the cell that sets `N_TRAIN`, increase to 10,000–30,000
- Re-run all cells — training takes roughly 1–3 hours on a GPU

### Option 2 — Add more epochs

Set `N_EPOCHS = 30` or higher. The learning curves show the model was still improving at epoch 15.

### Option 3 — Add physics-aware loss terms

The companion model (`best_full_checkpoint.pt`) was built with loss terms for mass conservation, non-negativity, and spatial smoothness. Incorporating these into the functional model would reduce the background false-positive rate and improve physical plausibility.

### Option 4 — Multi-step prediction

Currently predicts only t+1. To predict t+2, t+3, t+4:
- Feed the prediction back as input (autoregressive rollout)
- Or adopt the autoregressive architecture from `best_full_checkpoint.pt`

### Option 5 — Improve the data pipeline

The current loader reads all windows into RAM (`load_windows`). For full-dataset training, switch to a `torch.utils.data.Dataset` that reads files on demand to avoid memory limits.

---

## File locations

```
Geospatial-Forecasting/
│
├── src/plume/
│   ├── convlstm_functional_v1.pt        ← this model
│   └── best_full_checkpoint.pt          ← advanced autoregressive model (4-step)
│
├── EDA/
│   ├── EDA_final_model.ipynb            ← inference & evaluation notebook
│   ├── ConvLSTM_Smoke_Prediction.ipynb  ← original architecture notebook
│   └── data/
│       └── windows/                     ← 40,215 .npz training windows
│
└── CONVLSTM_MODEL_GUIDE.md              ← this document
```

---

## Quick-reference card

```
LOAD:
  ckpt = torch.load('src/plume/convlstm_functional_v1.pt', map_location=device, weights_only=False)
  p    = {k: v.to(device) for k, v in ckpt['model_state'].items()}

NORMALISE INPUT (mandatory):
  x_norm = (x_raw - ckpt['normalisation']['ch_mean'][:, None, None]) \
                  / ckpt['normalisation']['ch_std'][:, None, None]

PREDICT:
  with torch.no_grad():
      pred = forward(x_tensor, p)   # input: (B, 3, 10, 64, 64) → output: (B, 1, 64, 64)

DATA FORMAT:
  np.load('windows/XXXXXX_XXX.npz')
  input  shape → (3, 10, 64, 64)
  target shape → (1, 10, 64, 64)
```

---

*Model trained by the Geospatial Forecasting team. For questions, refer to [EDA/EDA_final_model.ipynb](EDA/EDA_final_model.ipynb) or [EDA/ConvLSTM_Smoke_Prediction.ipynb](EDA/ConvLSTM_Smoke_Prediction.ipynb).*
