import pandas as pd

df = pd.read_csv('dataset/crop_yield.csv')
df['crop'] = df['crop'].str.strip()
yield_stats = df.groupby('crop')['yield'].agg(['mean', 'max', 'count']).sort_values('mean', ascending=False)
print("Top 10 crops by mean yield:")
print(yield_stats.head(10))
