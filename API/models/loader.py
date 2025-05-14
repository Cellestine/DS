import joblib


# Chargement des modèles XGBoost
def load_model_freq(path=r"new_full_model_pipeline.pkl"):
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
    return joblib.load(path)
