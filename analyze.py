import pandas as pd
from pptx import Presentation
import sys

def extract_pptx_text(file_path):
    print("=== PRESENTATION TEXT ===")
    try:
        prs = Presentation(file_path)
        for i, slide in enumerate(prs.slides):
            print(f"\n--- Slide {i+1} ---")
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    print(shape.text)
    except Exception as e:
        print(f"Error reading pptx: {e}")
    print("\n" + "="*40 + "\n")

def analyze_csv(file_path):
    print(f"=== ANALYZING {file_path} ===")
    try:
        df = pd.read_csv(file_path)
        print("--- Info ---")
        df.info()
        print("\n--- Head ---")
        print(df.head())
        print("\n--- Missing Values ---")
        print(df.isnull().sum())
        print("\n--- Duplicates ---")
        print("Total duplicates:", df.duplicated().sum())
        
        print("\n--- Categorical Columns Unique Values (Top 5) ---")
        for col in df.select_dtypes(include=['object']).columns:
            unique_vals = df[col].unique()
            print(f"{col} ({len(unique_vals)} unique): {unique_vals[:5]}")
            
        print("\n--- Numerical Columns Describe ---")
        print(df.describe())
            
    except Exception as e:
        print(f"Error reading csv: {e}")
    print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    extract_pptx_text('SmartCrop_Presentation (1).pptx')
    analyze_csv('dataset/crop_yield.csv')
    analyze_csv('dataset/state_soil_data.csv')
    analyze_csv('dataset/state_weather_data_1997_2020.csv')
