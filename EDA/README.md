# Exploratory Data Analysis (EDA) - README

## Quick Start

### View the Notebook
Open `EDA.ipynb` in Jupyter Notebook or JupyterLab to explore the complete analysis:

```bash
jupyter notebook EDA/EDA.ipynb
```

## Notebook Contents

The `EDA.ipynb` notebook contains 31 cells organized into the following sections:

### 1. **Data Loading & Setup**
   - Load NPZ files and CSV manifests
   - Verify data structures and dimensions

### 2. **NPZ File Inspection**
   - Validate dataset structure (shape, keys)
   - Confirm input/target frame dimensions: (3, 10, 64, 64) and (1, 10, 64, 64)

### 3. **Visualization**
   - Sample concentration grids with heatmaps
   - Frame-by-frame temporal visualization

### 4. **EDA Overview**
   - Dataset shape and structure summary
   - Sample path inspection

### 5. **Metadata Distributions**
   - Scenario parameters: geographic coordinates, run hours, emission rates, release heights
   - Concentration statistics across windows
   - Nonzero cell counts

### 6. **Sample Window Statistics**
   - Load and inspect sample NPZ files
   - Min/max/mean value analysis
   - Data integrity checks

### 7. **Temporal & Time-Series Analysis** ⭐ *New*
   - **Time-Series Properties**: Run hours, frame sequences
   - **Temporal Distribution**: Month/hour histograms for release timing
   - **Window Sequence Analysis**: Distribution of windows per scenario
   - **Data Continuity**: Consecutive vs. non-consecutive windows
   - **Data Quality Metrics**: Sparsity, dynamic range, tensor distributions
   - **Parameter Relationships**: Scatter plots and correlation analysis
   - **Summary & Insights**: 11+ key findings with model training implications

## Key Findings

| Finding | Value |
|---------|-------|
| Total Scenarios | 10,000 |
| Total Windows | 40,215 |
| Avg Windows/Scenario | 4-5 |
| Grid Size | 64×64 |
| Channels/Frame | 10 |
| Sparsity | ~98% zeros |
| Time Span | 2024-01-01 to 2026-03-31 |
| Run Hours Range | 3-12 hours |
| Concentration Range | 0 to 1200+ |
| Nonzero Cells | 19 to 20,667 |

## Data Files

### Core Data
- **`data/windows_manifest.csv`** - Catalog of 40k+ windows with metadata
- **`data/dataset_manifest.csv`** - 10k scenario records with parameters and statistics
- **`data/windows/`** - 40k+ NPZ files containing input/target frame pairs

### Documentation
- **`data/README.md`** - Dataset technical documentation
- **`ANALYSIS_SUMMARY.md`** - Comprehensive EDA findings and implications

## Dataset Structure

### Input/Target Format (3→1 Prediction)
```
Input:   3 frames × 10 channels × 64×64 grid
Target:  1 frame  × 10 channels × 64×64 grid
```

### Channels
Represents atmospheric concentration and auxiliary variables at each time step.

### Sparsity Pattern
- Highly sparse grids (~98% zeros)
- Concentration localized to regions near release point
- Exponential tail distribution in nonzero values

## For Model Development

### Data Preprocessing
1. **Normalization**: Apply log-scale transformation to handle exponential tail
2. **Sparse Tensors**: Consider COO format for computational efficiency
3. **Padding/Masking**: Handle variable sequence lengths (3-12 hours)

### Train/Validation/Test Split
- Stratify by scenario to ensure diverse spatial/temporal coverage
- Typical split: 70% train, 15% val, 15% test

### Model Architecture Considerations
- **Input**: Time-series of 3 frames (channels × 64×64)
- **Output**: Single future frame prediction
- **Options**: CNN → LSTM, 3D Convolution, or Transformer architectures

### Loss Functions
- MSE/MAE on log-transformed concentrations (handles skewed distribution)
- Spatially-weighted loss emphasizing nonzero regions
- Concentration-weighted loss for physical realism

## Execution Status

| Metric | Status |
|--------|--------|
| Notebook Cells | 31 total (15 code, 16 markdown) |
| Execution Status | ✅ All cells executed successfully |
| Runtime Errors | 0 |
| Generated Outputs | Histograms, scatter plots, statistics |

## Notes

- The dataset is **production-ready** and validated for model training
- Temporal patterns show seasonal and diurnal variations in release timing
- Spatial patterns follow exponential diffusion from release point
- 98% sparsity requires sparse tensor operations for efficiency

---

**For Questions or Extensions**: See `ANALYSIS_SUMMARY.md` for detailed findings and next steps.
