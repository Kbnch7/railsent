import pandas as pd

dataset_path = 'cleaned_data_V2.csv'

new_sentences_file = 'new_text.txt'

df = pd.read_csv(dataset_path)

with open(new_sentences_file, 'r', encoding='utf-8') as file:
    new_sentences = [line.strip().split('. ', 1)[-1] for line in file if line.strip()]

new_data = pd.DataFrame(new_sentences, columns=['text'])

new_data['sentiment'] = 'Positive'

df_updated = pd.concat([df, new_data], ignore_index=True)

df_updated.to_csv(dataset_path, index=False)

print("New data added successfully.")
