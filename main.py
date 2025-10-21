import get_data as get
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """A faire au propre si besoin plus tard. Le noteboolk est plus pratique pour l'instant."""
    # get.sample_data()
    df = pd.read_csv('./higgs/HIGGS_sample.csv')
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df)

    # Afficher la variance expliquée par chaque composante principale dans un graphique
    explained_variance = pca.explained_variance_ratio_
    print("Explained variance ratio:", explained_variance)
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center')
    plt.xlabel('Composante principale')
    plt.ylabel('Variance expliquée')
    plt.title('Variance expliquée par chaque composante principale')
    plt.show()
    # Créer un DataFrame pour les résultats PCA
    pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])

    
