import joblib

def load_model(path=r"C:\Users\CYTech Student\projet\indus_ia\DS\models_pkls\frequence\xgb_regressor_model.pkl"):
    return joblib.load(path)
