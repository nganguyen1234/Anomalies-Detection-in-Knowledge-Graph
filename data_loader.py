import pandas as pd

def load_entity_types(file_path):
    df = pd.read_csv(file_path, sep='\t', names=['entity', 'type'])
    df.dropna(inplace=True)
    entities = df['entity'].tolist()
    types = df['type'].tolist()
    return entities, types
