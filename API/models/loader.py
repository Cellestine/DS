"""
Ce module contient les fonctions permettant de load les models pour lancer l'API.
"""

import joblib


# Chargement des modèles XGBoost
def load_model_freq(path=r"models_pkls/frequence/new_full_model_pipeline.pkl"):
    """
    Charge le modèle de prédiction de la fréquence des sinistres depuis le fichier spécifié.

    Parameters:
        path (str): Chemin vers le fichier .pkl contenant le modèle XGBoost.

    Returns:
        model: Modèle entraîné chargé avec joblib.
    """
    return joblib.load(path)


def load_model_montant(path=r"models_pkls/montant/xgb_model (2).pkl"):
    """
    Charge le modèle de prédiction du montant des sinistres depuis le fichier spécifié.

    Parameters:
        path (str): Chemin vers le fichier .pkl contenant le modèle XGBoost.

    Returns:
        model: Modèle entraîné chargé avec joblib.
    """
    model = joblib.load(path)
    # si c'est un XGBRegressor sklearn, on lui dit qu'il doit traiter
    # les catégories nativement :
    try:
        model.enable_categorical = True
    except AttributeError:
        pass
    return model
