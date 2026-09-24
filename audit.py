import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def run_audit():
    print("=== 1. AUDIT FINAL MODELING DATASET ===")
    crop_df = pd.read_csv('dataset/crop_yield.csv')
    soil_df = pd.read_csv('dataset/state_soil_data.csv')
    weather_df = pd.read_csv('dataset/state_weather_data_1997_2020.csv')
    
    for df in [crop_df, soil_df, weather_df]:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
            
    merged_df = pd.merge(crop_df, soil_df, on='state', how='left')
    merged_df = pd.merge(merged_df, weather_df, on=['state', 'year'], how='left')
    
    print(f"Final shape: {merged_df.shape}")
    print("\nColumns and Data Types:")
    print(merged_df.dtypes)
    print(f"\nMissing Values:\n{merged_df.isnull().sum()}")
    print(f"\nDuplicate Rows: {merged_df.duplicated().sum()}")
    
    cat_cols = merged_df.select_dtypes(include=['object']).columns.tolist()
    num_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    print(f"\nCategorical columns: {cat_cols}")
    print(f"Numerical columns: {num_cols}")
    
    print("\n=== 2. AUDIT THE DATASET MERGE ===")
    # Check if there are weather records for every state+year in crop dataset
    missing_weather = merged_df['avg_temp_c'].isnull().sum()
    print(f"Rows missing weather data after state+year merge: {missing_weather}")
    
    print("\n=== 3. TARGET LEAKAGE AUDIT ===")
    # Check correlations
    correlations = merged_df[num_cols].corr()
    print("\nCorrelations with yield:")
    print(correlations['yield'].sort_values(ascending=False))
    
    print("\nCorrelations with production:")
    print(correlations['production'].sort_values(ascending=False))
    
    print("\nCheck if pesticide/fertilizer are deterministic based on area:")
    print("Fertilizer / Area variance:", (merged_df['fertilizer'] / merged_df['area']).var())
    print("Pesticide / Area variance:", (merged_df['pesticide'] / merged_df['area']).var())
    print("Fertilizer / Area mean:", (merged_df['fertilizer'] / merged_df['area']).mean())
    print("Pesticide / Area mean:", (merged_df['pesticide'] / merged_df['area']).mean())

    print("\n=== 4 & 5 & 6. TRAIN/TEST METHODOLOGY & METRICS & RF INVESTIGATION ===")
    X = merged_df.drop(columns=['production', 'year', 'yield'])
    y = merged_df['yield']
    
    # Train/test split exactly as in train.py
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = joblib.load('models/best_model_pipeline.joblib')
    
    y_pred = model.predict(X_test)
    print(f"Loaded RF R2: {r2_score(y_test, y_pred):.4f}")
    
    # Extract feature importances
    rf_model = model.named_steps['model']
    preprocessor = model.named_steps['preprocessor']
    
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
    num_features = ['area', 'fertilizer', 'pesticide', 'N', 'P', 'K', 'pH', 'avg_temp_c', 'total_rainfall_mm', 'avg_humidity_percent']
    all_feature_names = num_features + list(cat_feature_names)
    
    importances = rf_model.feature_importances_
    feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)
    print("\nTop 10 Feature Importances:")
    print(feat_imp.head(10))
    
if __name__ == "__main__":
    run_audit()
