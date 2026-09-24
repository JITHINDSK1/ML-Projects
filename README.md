# SmartCrop - Crop Yield Prediction System

SmartCrop is a machine learning tool designed for agricultural stakeholders. It abstracts away complex data structures (like historical environmental variables) to provide a simple, practical interface for predicting crop yields and maximizing production.

## User Workflow & Inputs
A user should not need to understand ML concepts, historical timelines, or state-wide environmental averages. The system requires only five basic inputs:
1. **State**
2. **Crop**
3. **Area** (Hectares)
4. **Fertilizer** (kg)
5. **Pesticide** (kg)

## How the Recommended Season is Determined
SmartCrop automatically evaluates every valid agricultural season for the selected Crop and State. 
For each season, it retrieves the representative historical environmental conditions (State-level average for Temperature, Rainfall, Humidity, and Soil metrics) and passes the data through the trained ML model. The system then compares the predicted yield across all seasons and recommends the season that maximizes production.

*Note: The "Recommended Season" is strictly the season with the highest predicted yield based on the trained model and historical data; it is not a universally scientifically proven "best" season.*

## Setup and Training

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost flask joblib
```

### 2. How to Train the Models
Before running the predictors, run the training pipeline.
```bash
python train.py
```
This script reads the raw datasets, applies a strict chronological train/test split (Train <= 2017, Test >= 2018) to prevent time-leakage, and saves the trained models along with the necessary metadata (valid seasons mapping and state-wide historical weather averages). 

## How to Run the Terminal Predictor (CLI)
A clean command-line interface is available for fast predictions without a browser.
```bash
python predict.py
```
You will be prompted to enter the 5 inputs sequentially. The tool will output the Season Comparison, Recommended Season, and the environmental variables that were automatically retrieved.

## How to Run the Website
To launch the minimalist graphical interface:
```bash
python app.py
```
Then navigate to `http://127.0.0.1:5000` in your browser. Enter your inputs and click "Predict Crop Yield" to view the detailed breakdown.

## Model Evaluation Metrics Explained
SmartCrop utilizes Support Vector Regression (SVR) as its primary predictor, chosen after robust evaluation against Random Forest and XGBoost.

- **MAE (Mean Absolute Error):** The average absolute difference between the predicted yield and actual historical yield. A lower MAE means the predictions are closer to reality.
- **RMSE (Root Mean Squared Error):** Similar to MAE, but heavily penalizes large errors. A lower RMSE indicates the model rarely makes huge mistakes.
- **R² (R-Squared):** The proportion of the variance in the crop yield that is predictable from the input features. A score closer to 1.0 indicates a perfect fit (our corrected SVR achieved 0.77).
