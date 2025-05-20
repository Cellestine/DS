"""
Refactoring du code de prétraitement pour le rendre plus modulaire et réutilisable.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ---------------------------------------------
# Chargement des fichiers
# ---------------------------------------------
X_train = pd.read_csv("train_input_Z61KlZo 1.csv", low_memory=False)
y_train = pd.read_csv("train_output_DzPxaPY 1.csv")["FREQ"]
X_test = pd.read_csv("test_input_5qJzHrr 1.csv", low_memory=False)

# ---------------------------------------------
# Colonnes (issues de config)
# ---------------------------------------------
NUMERICAL_COLUMNS = [
    "ID",
    "TYPERS",
    "ANCIENNETE",
    "DUREE_REQANEUF",
    "TYPBAT2",
    "KAPITAL12",
    "KAPITAL25",
    "KAPITAL32",
    "SURFACE1",
    "SURFACE4",
    "SURFACE10",
    "NBBAT1",
    "RISK1",
    "RISK7",
    "EQUIPEMENT4",
    "EQUIPEMENT6",
    "ZONE_VENT",
    "ANNEE_ASSURANCE",
    "ZONE",
    "surface_totale",
    "capital_total",
    "surface_par_batiment",
    "capital_par_surface",
    "capital_moyen_par_batiment",
]

CATEGORIAL_COLUMNS = [
    "ACTIVIT2",
    "VOCATION",
    "CARACT1",
    "CARACT3",
    "CARACT4",
    "TYPBAT1",
    "INDEM2",
    "FRCH1",
    "FRCH2",
    "DEROG12",
    "DEROG13",
    "DEROG14",
    "DEROG16",
    "TAILLE1",
    "TAILLE2",
    "COEFASS",
    "RISK6",
    "RISK8",
    "RISK9",
    "RISK10",
    "RISK11",
    "RISK12",
    "RISK13",
    "EQUIPEMENT2",
    "EQUIPEMENT5",
]

ALL_COLUMNS = NUMERICAL_COLUMNS + CATEGORIAL_COLUMNS

# Création de variables dérivées
for df in [X_train, X_test]:
    df["surface_totale"] = (
        df[["SURFACE1", "SURFACE4", "SURFACE10"]].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    )
    df["capital_total"] = (
        df[["KAPITAL12", "KAPITAL25", "KAPITAL32"]]
        .apply(pd.to_numeric, errors="coerce")
        .sum(axis=1)
    )
    df["surface_par_batiment"] = df["surface_totale"] / df["NBBAT1"].replace(0, np.nan)
    df["capital_par_surface"] = df["capital_total"] / df["surface_totale"].replace(0, np.nan)
    df["capital_moyen_par_batiment"] = df["capital_total"] / df["NBBAT1"].replace(0, np.nan)

# Forcer la conversion des colonnes numériques
for col in NUMERICAL_COLUMNS:
    if col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    if col in X_test.columns:
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")


# ---------------------------------------------
# Définition des classes custom
# ---------------------------------------------
class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Sélectionne un sous-ensemble de colonnes d'un DataFrame.

    Paramètres
    ----------
    selected_columns : list of str
        Liste des noms de colonnes à conserver dans le DataFrame.

    Méthodes
    --------
    fit(X, y=None)
        Méthode d'ajustement (inutile ici, renvoie self).
    transform(X)
        Retourne un DataFrame ne contenant que les colonnes sélectionnées.
    """

    def __init__(self, selected_columns):
        self.selected_columns = selected_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.selected_columns].copy()


class MissingValueFiller(BaseEstimator, TransformerMixin):
    """
    Remplit les valeurs manquantes :
    - avec 0 pour les colonnes numériques,
    - avec 'Inconnu' pour les colonnes catégorielles.

    Paramètres
    ----------
    num_cols : list of str, optional
        Noms des colonnes numériques.
    cat_cols : list of str, optional
        Noms des colonnes catégorielles.

    Méthodes
    --------
    fit(X, y=None)
        Renvoie self sans modification.
    transform(X)
        Remplit les NaN selon le type des colonnes.
    """

    def __init__(self, num_cols=None, cat_cols=None):
        self.num_cols = num_cols
        self.cat_cols = cat_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.num_cols:
            for col in self.num_cols:
                if col in X_copy.columns:
                    X_copy[col] = X_copy[col].fillna(0)
        if self.cat_cols:
            for col in self.cat_cols:
                if col in X_copy.columns:
                    X_copy[col] = X_copy[col].fillna("Inconnu")
        return X_copy


class ManualCountEncoder(BaseEstimator, TransformerMixin):
    """
    Encode les variables catégorielles avec leur fréquence d'apparition (count encoding).

    Paramètres
    ----------
    cat_cols : list of str
        Liste des colonnes catégorielles à encoder.

    Méthodes
    --------
    fit(X, y=None)
        Calcule les fréquences des catégories dans chaque colonne.
    transform(X)
        Applique le mapping de fréquence à chaque colonne catégorielle.
    """

    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols
        self.count_maps = {}

    def fit(self, X, y=None):
        for col in self.cat_cols:
            counts = X[col].value_counts()
            self.count_maps[col] = counts.to_dict()
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.cat_cols:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].map(self.count_maps.get(col, {})).fillna(0)
            else:
                X_copy[col] = 0
        return X_copy


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Supprime les colonnes peu informatives :
    - trop de valeurs manquantes,
    - faible variance,
    - forte corrélation avec d'autres colonnes.

    Paramètres
    ----------
    num_cols : list of str
        Colonnes numériques à vérifier pour la variance/corrélation.
    missing_thresh : float, default=0.4
        Seuil au-delà duquel une colonne est supprimée pour taux de valeurs manquantes.
    var_thresh : float, default=0.01
        Seuil minimum de variance.
    corr_thresh : float, default=0.95
        Seuil maximum de corrélation autorisée entre colonnes numériques.

    Méthodes
    --------
    fit(X, y=None)
        Identifie les colonnes à supprimer.
    transform(X)
        Supprime les colonnes identifiées.
    """

    def __init__(self, num_cols=None, missing_thresh=0.4, var_thresh=0.01, corr_thresh=0.95):
        self.num_cols = num_cols
        self.missing_thresh = missing_thresh
        self.var_thresh = var_thresh
        self.corr_thresh = corr_thresh
        self.columns_to_drop_ = []

    def fit(self, X, y=None):
        X_copy = X.copy()
        drop_cols = []

        for col in X_copy.columns:
            if col in self.num_cols:
                missing_ratio = (X_copy[col] == 0).sum() / len(X_copy)
            else:
                missing_ratio = (X_copy[col] == "Inconnu").sum() / len(X_copy)
            if missing_ratio > self.missing_thresh:
                drop_cols.append(col)

        var_series = X_copy.var(numeric_only=True)
        low_var_cols = var_series[var_series < self.var_thresh].index.tolist()
        drop_cols += low_var_cols

        corr_matrix = X_copy.select_dtypes(include=["number"]).corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_cols = [
            col for col in upper_tri.columns if any(upper_tri[col] > self.corr_thresh)
        ]
        drop_cols += high_corr_cols

        self.columns_to_drop_ = list(set(drop_cols))
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop_, errors="ignore")


class ScalerWrapper(BaseEstimator, TransformerMixin):
    """
    Applique une standardisation (z-score) aux colonnes numériques sélectionnées.

    Paramètres
    ----------
    num_cols : list of str
        Colonnes à normaliser.

    Méthodes
    --------
    fit(X, y=None)
        Calcule les statistiques de normalisation.
    transform(X)
        Applique la transformation standardisée.
    """

    def __init__(self, num_cols=None):
        self.num_cols = num_cols
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        if self.num_cols:
            self.scaler.fit(X[self.num_cols])
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.num_cols:
            X_copy[self.num_cols] = self.scaler.transform(X_copy[self.num_cols])
        return X_copy


# ---------------------------------------------
# Création pipeline complet
# ---------------------------------------------
preprocessing_pipeline = Pipeline(
    [
        ("select", ColumnSelector(ALL_COLUMNS)),
        ("missing", MissingValueFiller(NUMERICAL_COLUMNS, CATEGORIAL_COLUMNS)),
        ("encoding", ManualCountEncoder(CATEGORIAL_COLUMNS)),
        ("drop", ColumnDropper(NUMERICAL_COLUMNS)),
        ("scaling", ScalerWrapper(NUMERICAL_COLUMNS)),
    ]
)

X_train_processed = preprocessing_pipeline.fit_transform(X_train)
X_test_processed = preprocessing_pipeline.transform(X_test)

model = XGBRegressor(objective="count:poisson", random_state=42)
model.fit(X_train_processed, y_train)


# ---------------------------------------------
# Pipeline final avec modèle
# ---------------------------------------------
full_pipeline = Pipeline([("preprocessing", preprocessing_pipeline), ("model", model)])


# 🔁 Prédiction sur les 10 premières lignes du test
print("✅ Prédictions effectuées avec succès sur les 10 premières lignes du test.")
for i, x in enumerate(X_test.iloc[:10].to_dict(orient="records"), 1):
    input_df = pd.DataFrame([x])
    prediction = full_pipeline.predict(input_df)[0]
    print(f"Prédiction {i}: {round(prediction, 2)}")


joblib.dump(full_pipeline, r"models_pkls/frequence/new_full_model_pipeline.pkl")
print("✅ Nouveau pipeline sauvegardé sous new_full_model_pipeline.pkl")
