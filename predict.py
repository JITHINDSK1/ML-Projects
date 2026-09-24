import sys
import joblib
import pandas as pd
import os

def main():
    if not os.path.exists('models/best_model_pipeline.joblib') or not os.path.exists('models/meta_info.joblib'):
        print("Error: Models and metadata not found. Please run train.py first.")
        sys.exit(1)
        
    model = joblib.load('models/best_model_pipeline.joblib')
    meta = joblib.load('models/meta_info.joblib')
    
    print("\n--- SMARTCROP ---")
    print("Crop Yield Prediction System\n")
    
    # Simple input prompts
    print("Please enter the following details:")
    state = input(f"State (e.g. {meta['states'][0]}): ").strip()
    if state not in meta['states']:
        print(f"Error: State '{state}' not recognized.")
        sys.exit(1)
        
    crop = input(f"Crop (e.g. {meta['crops'][0]}): ").strip()
    if crop not in meta['crops']:
        print(f"Error: Crop '{crop}' not recognized.")
        sys.exit(1)
        
    try:
        area = float(input("Area (hectares): ").strip())
        fertilizer = float(input("Fertilizer (kg): ").strip())
        pesticide = float(input("Pesticide (kg): ").strip())
    except ValueError:
        print("Error: Please enter valid numerical values for Area, Fertilizer, and Pesticide.")
        sys.exit(1)
        
    print("\nAnalyzing available seasons...")
    
    valid_seasons = meta['valid_seasons'].get(state, {}).get(crop, [])
    if not valid_seasons:
        print(f"Error: No historical season data found for {crop} in {state}.")
        sys.exit(1)
        
    weather = meta['weather_avg'][state]
    soil = meta['soil_data'][state]
    
    results = {}
    best_season = None
    max_yield = -1
    
    for season in valid_seasons:
        input_data = pd.DataFrame([{
            'crop': crop,
            'season': season,
            'state': state,
            'area': area,
            'fertilizer': fertilizer,
            'pesticide': pesticide,
            'N': soil['N'],
            'P': soil['P'],
            'K': soil['K'],
            'pH': soil['pH'],
            'avg_temp_c': weather['avg_temp_c'],
            'total_rainfall_mm': weather['total_rainfall_mm'],
            'avg_humidity_percent': weather['avg_humidity_percent']
        }])
        
        pred = model.predict(input_data)[0]
        pred = max(0, pred)
        results[season] = pred
        
        if pred > max_yield:
            max_yield = pred
            best_season = season
            
    # Print results
    print()
    for season, pred_yield in results.items():
        print(f"{season}: {pred_yield:.2f} tonnes/hectare")
        
    print(f"\nRecommended Season: {best_season}")
    print(f"Estimated Yield: {max_yield:.2f} tonnes/hectare")
    print(f"Estimated Production: {(max_yield * area):.2f} tonnes\n")
    
    # Read model metrics to get the best model name
    model_used = "Ensemble Model"
    if os.path.exists('processed_data/model_metrics.csv'):
        metrics_df = pd.read_csv('processed_data/model_metrics.csv')
        model_used = metrics_df.sort_values(by='R2', ascending=False).iloc[0]['Model']
        
    print(f"Model Used: {model_used}")
    print("\nAutomatically Retrieved Conditions (Historical Averages):")
    print(f"Soil -> N: {soil['N']}, P: {soil['P']}, K: {soil['K']}, pH: {soil['pH']}")
    print(f"Weather -> Temp: {weather['avg_temp_c']:.1f} °C, Rain: {weather['total_rainfall_mm']:.1f} mm, Humidity: {weather['avg_humidity_percent']:.1f} %")
    print("\n-----------------\n")

if __name__ == "__main__":
    main()
