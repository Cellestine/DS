# prediction_pipeline_pickle.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import unittest

# === Étape de transformation personnalisée ===
class NanRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.columns_to_remove = []

    def fit(self, X, y=None):
        self.columns_to_remove = [
            col for col in X.columns if X[col].isna().mean() > self.threshold
        ]
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_remove, errors='ignore')

# === Classe pour entraîner et sauvegarder un pipeline sklearn ===
class ModelTrainer:
    def __init__(self, threshold=0.4):
        self.pipeline = Pipeline([
            ('nan_remover', NanRemover(threshold=threshold)),
            ('regressor', LinearRegression())
        ])

    def train(self, X, y):
        self.pipeline.fit(X, y)
        return self.pipeline

    def save(self, path="pipeline_frequence.pkl"):
        joblib.dump(self.pipeline, path)

    def load(self, path="pipeline_frequence.pkl"):
        self.pipeline = joblib.load(path)
        return self.pipeline

    def predict(self, X):
        return self.pipeline.predict(X)

# === Exemple d'utilisation (à commenter en prod) ===
if __name__ == "__main__":
    # Données fictives
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [1, 1, 1, 1, 1],
        'C': [np.nan, np.nan, np.nan, np.nan, 5],
        'target': [10, 15, 10, 20, 25]
    })

    X = df.drop(columns='target')
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    trainer = ModelTrainer(threshold=0.8)
    trainer.train(X_train, y_train)
    trainer.save("pipeline_frequence.pkl")
    print("✅ Pipeline entraîné et sauvegardé avec succès.")

    # Chargement et prédiction
    loaded_pipeline = trainer.load("pipeline_frequence.pkl")
    predictions = trainer.predict(X_test)
    print("🔍 Prédictions :", predictions)

# === TESTS UNITAIRES ===
class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [1, 1, 1, 1, 1],
            'C': [np.nan, np.nan, np.nan, np.nan, 5],
            'target': [10, 15, 10, 20, 25]
        })
        self.X = self.df.drop(columns='target')
        self.y = self.df['target']
        self.trainer = ModelTrainer(threshold=0.8)

    def test_pipeline_training_and_prediction(self):
        self.trainer.train(self.X, self.y)
        preds = self.trainer.predict(self.X)
        self.assertEqual(len(preds), len(self.X))

    def test_model_saving_and_loading(self):
        self.trainer.train(self.X, self.y)
        self.trainer.save("test_pipeline.pkl")
        loaded = self.trainer.load("test_pipeline.pkl")
        preds_loaded = self.trainer.predict(self.X)
        self.assertEqual(len(preds_loaded), len(self.X))

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)
