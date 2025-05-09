import joblib

# Chargement des modèles XGBoost
def load_model_freq(path=r"models_pkls/frequence/xgb_regressor_model.pkl"):
    return joblib.load(path)

def load_model_montant(path=r"models_pkls/montant/xgb_model (2).pkl"):
    return joblib.load(path)