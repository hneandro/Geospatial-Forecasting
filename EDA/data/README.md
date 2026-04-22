# HYSPLIT Plume ConvLSTM Multiyear Dataset with GDAS Meteorology (2024-2026)

This dataset contains HYSPLIT-generated atmospheric dispersion outputs prepared for spatiotemporal forecasting experiments, enriched with broadcast GDAS meteorology channels for ConvLSTM-style next-frame prediction.

## Contents

- `windows_manifest_enriched.csv`
  - Window-level metadata for ConvLSTM-ready enriched training windows.
- `windows/`
  - Compressed `.npz` files containing fixed-shape ConvLSTM windows.
  - Each file contains:
    - `scenario_id`
    - `window_id`
    - `input` with shape `[T_in, C, H, W]`
    - `target` with shape `[T_out, C, H, W]`



## Generation summary

- Meteorology source years: 2024, 2025, and 2026
- Dispersion simulator: HYSPLIT
- Successful HYSPLIT scenarios: 10,000
- ConvLSTM-ready windows: 40,215
- Standardized spatial resolution: 64 x 64
- Window format: 3 input frames -> 1 target frame
- Total channels per frame: 10
- Concentration transform: `log1p(alpha * x)` with `alpha = 1e12`

## Channel order

Each frame contains the following channels in this exact order:

1. `plume_concentration`
2. `u10m_ms`
3. `v10m_ms`
4. `wspd10_ms`
5. `wdir_sin`
6. `wdir_cos`
7. `pblh_m`
8. `sfcp_hpa`
9. `rh2m_pct`
10. `t02m_k`

## Intended use

This dataset is intended for:
- ConvLSTM baseline training with meteorological context
- Spatiotemporal plume forecasting
- Environmental hazard spread modeling
- Next-frame concentration grid prediction
- Physics-informed surrogate modeling with weather-aware inputs

## Notes

- Concentration values are transformed for training stability.
- Meteorology channels are broadcast spatially across the grid for each frame.
- Wind direction is encoded as sine and cosine instead of raw degrees.
- Samples were derived from HYSPLIT concentration outputs and enriched with GDAS-derived meteorology using the HYSPLIT `profile` utility.
- Spatial extents vary in the original simulations; the ConvLSTM windows were standardized to a common grid.

## File format

Each file in `windows/` is a compressed NumPy archive (`.npz`).

Example loading code:

```python
import numpy as np

data = np.load("example_window.npz", allow_pickle=True)
print(data.files)
print(data["input"].shape)
print(data["target"].shape)
```

## Version notes

Added GDAS-enriched meteorology channels to the ConvLSTM dataset: u10m, v10m, wind speed, wind direction sin/cos, boundary layer height, surface pressure, relative humidity, and 2m temperature.
