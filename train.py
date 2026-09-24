import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

def main():
    print("=== Step 1 & 2: Loading Datasets ===")
    crop_df = pd.read_csv('dataset/crop_yield.csv')
    soil_df = pd.read_csv('dataset/state_soil_data.csv')
    weather_df = pd.read_csv('dataset/state_weather_data_1997_2020.csv')
    
    for df in [crop_df, soil_df, weather_df]:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
            
    print("=== Step 3: Merging Datasets ===")
    merged_df = pd.merge(crop_df, soil_df, on='state', how='left')
    merged_df = pd.merge(merged_df, weather_df, on=['state', 'year'], how='left')
    
    numeric_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        merged_df[col] = merged_df[col].fillna(merged_df[col].median())
        
    print("=== Step 4: Correcting Coconut Anomaly ===")
    merged_df = merged_df[merged_df['crop'] != 'Coconut']
        
    print("=== Step 5 & 6: Time-based Train/Test Split ===")
    train_df = merged_df[merged_df['year'] <= 2017]
    test_df = merged_df[merged_df['year'] >= 2018]
    
    columns_to_drop = ['production', 'year', 'yield']
    X_train = train_df.drop(columns=columns_to_drop)
    y_train = train_df['yield']
    X_test = test_df.drop(columns=columns_to_drop)
    y_test = test_df['yield']
    
    os.makedirs('processed_data', exist_ok=True)
    merged_df.drop(columns=['production']).to_csv('processed_data/final_dataset.csv', index=False)
    
    print("=== Step 7: Shared Preprocessing Pipeline ===")
    cat_features = ['crop', 'season', 'state']
    num_features = ['area', 'fertilizer', 'pesticide', 'N', 'P', 'K', 'pH', 'avg_temp_c', 'total_rainfall_mm', 'avg_humidity_percent']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ])
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1),
        'SVR': SVR(kernel='rbf', C=100, epsilon=0.1)
    }
    
    results = []
    trained_pipelines = {}
    
    print("=== Step 8, 9, 10: Training Models ===")
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
        trained_pipelines[name] = pipeline
        
    print("\n=== Step 11: Evaluation Comparison ===")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    results_df.to_csv('processed_data/model_metrics.csv', index=False)
    
    best_model_name = results_df.sort_values(by='R2', ascending=False).iloc[0]['Model']
    print(f"\nBest model identified: {best_model_name}")
    
    os.makedirs('models', exist_ok=True)
    best_pipeline = trained_pipelines[best_model_name]
    joblib.dump(best_pipeline, 'models/best_model_pipeline.joblib')
    
    print("=== Step 12: Generating Meta Info ===")
    # Calculate state-level weather averages across the entire dataset (1997-2020)
    weather_avg = weather_df.groupby('state')[['avg_temp_c', 'total_rainfall_mm', 'avg_humidity_percent']].mean().to_dict('index')
    
    # Store valid seasons for each state and crop
    valid_seasons_dict = {}
    for _, row in merged_df.iterrows():
        state = row['state']
        crop = row['crop']
        season = row['season']
        if state not in valid_seasons_dict:
            valid_seasons_dict[state] = {}
        if crop not in valid_seasons_dict[state]:
            valid_seasons_dict[state][crop] = set()
        valid_seasons_dict[state][crop].add(season)
        
    # Convert sets to lists
    for state in valid_seasons_dict:
        for crop in valid_seasons_dict[state]:
            valid_seasons_dict[state][crop] = list(valid_seasons_dict[state][crop])
    
    # Format soil lookup
    soil_dict = soil_df.set_index('state')[['N', 'P', 'K', 'pH']].to_dict('index')
            
    meta_info = {
        'crops': sorted(merged_df['crop'].unique().tolist()),
        'states': sorted(merged_df['state'].unique().tolist()),
        'valid_seasons': valid_seasons_dict,
        'weather_avg': weather_avg,
        'soil_data': soil_dict
    }
    joblib.dump(meta_info, 'models/meta_info.joblib')
    print("Metadata saved successfully.")

if __name__ == "__main__":
    main()
