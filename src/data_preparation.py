import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

def load_data(directory_path):
    """
    Charge les données à partir des fichiers CSV.
    
    Args:
        directory_path (str): Chemin du répertoire contenant les fichiers CSV.
    
    Returns:
        pd.DataFrame: DataFrame contenant les données fusionnées.
    """
    caracteristiques_path = os.path.join(directory_path, 'data', 'carcteristiques-2021.csv')
    lieux_path = os.path.join(directory_path, 'data', 'lieux-2021.csv')
    vehicules_path = os.path.join(directory_path, 'data', 'vehicules-2021.csv')
    usagers_path = os.path.join(directory_path, 'data', 'usagers-2021.csv')

    caracteristiques = pd.read_csv(caracteristiques_path)
    lieux = pd.read_csv(lieux_path)
    vehicules = pd.read_csv(vehicules_path)
    usagers = pd.read_csv(usagers_path)

    data = pd.merge(caracteristiques, lieux, on='Num_Acc')
    data = pd.merge(data, vehicules, on='Num_Acc')
    data = pd.merge(data, usagers, on='Num_Acc')

    return data

def process_data(data):
    """
    Prépare les données pour l'entraînement du modèle.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les données brutes.
    
    Returns:
        tuple: X_resampled, y_resampled (données préparées et rééchantillonnées)
    """
    features = [
        'lum', 'agg', 'int', 'atm', 'col', 'catr', 'circ', 'nbv', 'prof', 
        'plan', 'surf', 'infra', 'situ', 'vma', 'catv', 'obs', 'obsm', 
        'choc', 'manv', 'motor', 'place', 'catu', 'sexe', 'an_nais'
    ]
    target = 'grav'

    data = data[features + [target]]
    data = data[data[target].isin([1, 2, 3, 4])]
    data[target] = data[target] - 1
    data = data.fillna(0)
    data = pd.get_dummies(data, columns=features, drop_first=True)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.drop(columns=[target]))

    pca = PCA(n_components=0.95)
    data_pca = pca.fit_transform(data_scaled)

    data_pca = pd.DataFrame(data_pca)
    data_pca[target] = data[target].values

    X, y = data_pca.drop(columns=[target]), data_pca[target]
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled

