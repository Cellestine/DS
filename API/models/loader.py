import joblib

def load_model(path=r"models_pkls/frequence/xgb_regressor_model.pkl"):
    return joblib.load(path)
