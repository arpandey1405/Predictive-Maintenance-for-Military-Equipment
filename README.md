# Predictive Maintenance — Notebook1: `pmme1.ipynb`

This repository contains a single exploratory notebook (`pmme1.ipynb`) and the sample dataset `train_FD001.txt` used to build a baseline Remaining Useful Life (RUL) regression model for equipment (engine) degradation. The README here has been updated to match what the notebook actually does.

## What this notebook does (summary)

- Loads the dataset `train_FD001.txt` (plain text, space-separated). The file has 26 columns per row: `engine_id`, `cycle`, 3 operation settings, and 21 sensor readings.
- Assigns descriptive column names and computes the target variable RUL (Remaining Useful Life) per row by subtracting the current cycle from the engine's maximum cycle.
- Performs basic EDA: descriptive statistics, RUL trend plots for engines, distribution of RUL, and a correlation heatmap for selected features.
- Performs simple feature selection (a hand-picked list of sensors and settings used in the notebook) and scales features with StandardScaler.
- Trains a RandomForestRegressor on the processed data and evaluates it on a hold-out test split. Reports MAE, MSE, RMSE, and R².
- Produces diagnostic plots: residuals vs predicted, actual vs predicted, and residual distribution.

## Files in this repo

- `pmme1.ipynb` — The main Jupyter notebook with the end-to-end exploratory workflow and model training/evaluation.
- `train_FD001.txt` — Training data file used by the notebook (space-separated values).
- `requirements.txt` — List of Python dependencies with version constraints for reproducibility.

## Notebook: step-by-step (what you'll find inside)

1. Dependencies are imported: pandas, numpy, matplotlib, seaborn, and scikit-learn.
2. The dataset is loaded using `pd.read_csv(..., sep=r"\\s+", header=None)` and column names are assigned:
   - `engine_id`, `cycle`, `op_setting_1..3`, `sensor_1..21`.
3. RUL target computation:
   - Compute max cycle per engine and set `RUL = max_cycle - cycle`.
4. EDA and visualization:
   - RUL degradation curves for the first engines, histograms of RUL, and correlation heatmap of selected features.
5. Feature selection:
   - The notebook filters to a predefined set of features (e.g., `cycle`, `op_setting_1..3`, `sensor_2,3,4,7,8,11,15,17,20,21`).
6. Train/test split and scaling:
   - 10% test split, features scaled with `StandardScaler`.
7. Modeling and evaluation:
   - Model: `RandomForestRegressor(n_estimators=100, random_state=42)`
   - Metrics printed: MAE, MSE, RMSE, and R². The notebook's summary reports an example R² ≈ 0.7002 and MAE ≈ 26.66 (these are from the run in the notebook and may vary if you re-run with different random states or after changing features).
8. Diagnostic plots: residuals, actual vs predicted, and residual distribution.

## How to run the notebook (Windows PowerShell)

1. (Optional) Create and activate a virtual environment (recommended):

```powershell
python -m venv venv; .\\venv\\Scripts\\Activate.ps1
```

2. Install dependencies using the requirements.txt file:

```powershell
pip install -r requirements.txt
```

3. Start Jupyter and open the notebook:

```powershell
jupyter notebook
```

4. In the notebook UI open `pmme1.ipynb` and run the cells in order. Make sure `train_FD001.txt` is in the same directory as the notebook (it already is in this repo).

## Important notes and suggestions

- The notebook is exploratory and intentionally concise. It uses a fixed, hand-picked feature list. For production or more robust experiments consider:
  - Adding cross-validation (time-series aware if necessary) and hyperparameter tuning.
  - Using pipeline objects (scikit-learn Pipeline) to bundle scaling and modeling.
  - Saving trained models (joblib/pickle) and test predictions for later analysis.
  - Separating preprocessing, feature engineering, training, and evaluation into modular scripts or notebooks.

- The dataset `train_FD001.txt` in this repo appears to be the well-known CMAPSS FD001 subset (or similar). If you use other files (e.g., FD002/FD003/FD004), update the notebook accordingly.

## Reproducibility

To make experiments reproducible, pin package versions and set random seeds consistently. The notebook uses `random_state=42` for train/test split and RandomForest initialization.

## Contact / Next steps

Potential enhancements for this project:

- Convert the notebook into a cleaner pipeline script (train.py / evaluate.py) and add saving/loading of trained models.
- Add unit tests for data-loading and basic preprocessing.
- Create a dashboard for visualizing model predictions and sensor data relationships.
- Implement additional models (e.g., XGBoost, neural networks) for comparison.
