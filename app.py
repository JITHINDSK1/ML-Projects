from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = 'models/best_model_pipeline.joblib'
META_PATH = 'models/meta_info.joblib'
METRICS_PATH = 'processed_data/model_metrics.csv'

@app.route('/')
def home():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        return "Model not trained yet. Please run train.py first.", 500
        
    meta_info = joblib.load(META_PATH)
    
    metrics_html = ""
    if os.path.exists(METRICS_PATH):
        metrics_df = pd.read_csv(METRICS_PATH)
        metrics_df[['MAE', 'RMSE', 'R2']] = metrics_df[['MAE', 'RMSE', 'R2']].round(4)
        metrics_html = metrics_df.to_html(classes='metrics-table', index=False)
        
    # The UI only needs States and Crops now.
    return render_template('index.html', 
                           crops=meta_info['crops'], 
                           states=meta_info['states'],
                           metrics_table=metrics_html)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        state = data['state'].strip()
        crop = data['crop'].strip()
        area = float(data['area'])
        fertilizer = float(data['fertilizer'])
        pesticide = float(data['pesticide'])
        
        meta = joblib.load(META_PATH)
        
        valid_seasons = meta['valid_seasons'].get(state, {}).get(crop, [])
        if not valid_seasons:
            return jsonify({'error': f'No historical season data found for {crop} in {state}.'}), 400
            
        weather = meta['weather_avg'][state]
        soil = meta['soil_data'][state]
        
        model = joblib.load(MODEL_PATH)
        
        season_results = []
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
            
            season_results.append({
                'season': season,
                'yield': round(pred, 2)
            })
            
            if pred > max_yield:
                max_yield = pred
                best_season = season
                
        # Sort season_results by yield descending
        season_results.sort(key=lambda x: x['yield'], reverse=True)
        
        production = max_yield * area
        
        best_model_name = "Ensemble Model"
        if os.path.exists(METRICS_PATH):
            metrics_df = pd.read_csv(METRICS_PATH)
            best_model_name = metrics_df.sort_values(by='R2', ascending=False).iloc[0]['Model']

        return jsonify({
            'success': True,
            'recommended_season': best_season,
            'yield': round(max_yield, 2),
            'production': round(production, 2),
            'season_comparison': season_results,
            'model_used': best_model_name,
            'soil': soil,
            'weather': weather
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
