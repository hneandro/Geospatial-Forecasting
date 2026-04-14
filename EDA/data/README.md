# HYSPLIT Plume ConvLSTM Multiyear Dataset (2024-2026)

This dataset contains HYSPLIT-generated atmospheric dispersion outputs prepared for spatiotemporal forecasting experiments, especially ConvLSTM-style next-frame prediction.

## Contents

- `windows_manifest.csv`
  - Window-level metadata for ConvLSTM-ready training windows.
- `windows/`
  - Compressed `.npz` files containing fixed-shape ConvLSTM windows.
  - Each file contains:
    - `scenario_id`
    - `window_id`
    - `input` with shape `[T_in, C, H, W]`
    - `target` with shape `[T_out, C, H, W]`

- `dataset_manifest.csv` - Scenario-level metadata for the full generated HYSPLIT dataset.

## Generation summary

- Meteorology source years: 2024, 2025, and January-March 2026
- Dispersion simulator: HYSPLIT
- Successful HYSPLIT scenarios: 10,000
- ConvLSTM-ready windows: 40,215
- Standardized spatial resolution: 64 x 64
- Window format: 3 input frames -> 1 target frame
- Concentration transform: `log1p(alpha * x)` with `alpha = 1e12`

## Intended use

This dataset is intended for:
- ConvLSTM baseline training
- Spatiotemporal plume forecasting
- Environmental hazard spread modeling
- Next-frame concentration grid prediction

## Notes

- Concentration values are transformed for training stability.
- Samples were derived from HYSPLIT concentration outputs converted to text and then standardized to fixed-size tensors.
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
## Version notes Initial release: 10,000 HYSPLIT multiyear scenarios with 40,215 ConvLSTM-ready windows.
