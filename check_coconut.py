import pandas as pd

df = pd.read_csv('dataset/crop_yield.csv')
print("Coconut mean yield:", df[df['crop'].str.strip() == 'Coconut']['yield'].mean())
print("Other crops mean yield:", df[df['crop'].str.strip() != 'Coconut']['yield'].mean())
print("Coconut max yield:", df[df['crop'].str.strip() == 'Coconut']['yield'].max())
print("Other crops max yield:", df[df['crop'].str.strip() != 'Coconut']['yield'].max())
