import pandas as pd

df = pd.read_csv('data.csv')

df = df.dropna(subset=['sentiment', 'text'])

df = df[(df['text'].str.strip() != '') & (df['sentiment'].str.strip() != '')]

df.to_csv('cleaned_data.csv', index=False)

print(f"Numb rows: {df.shape[0]}")
