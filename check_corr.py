import pandas as pd
df = pd.read_csv('dataset/crop_yield.csv')
print("Corr fertilizer & area:", df['fertilizer'].corr(df['area']))
print("Corr pesticide & area:", df['pesticide'].corr(df['area']))
