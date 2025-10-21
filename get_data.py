import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    """Récupérer chunk par chunk pour éviter les problèmes de mémoire."""
    chunk_size = 1000000
    data_chunks = []
    for chunk in pd.read_csv('./higgs/HIGGS.csv', chunksize=chunk_size):
        data_chunks.append(chunk)
        print("chunk loaded, current shape :", len(data_chunks))
    data = pd.concat(data_chunks, ignore_index=True)

    return data

def sample_data(file_name='HIGGS_sample', sample_size=1000000, random_state=42):
    """Échantillonner les données pour une analyse plus rapide."""
    data = load_data()
    data_samples = data.sample(n = sample_size, random_state=random_state)
    data_samples.to_csv(f'./higgs/{file_name}.csv', index=False)
