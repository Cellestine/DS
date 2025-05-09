# prediction_frequence_assurance
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from category_encoders import CountEncoder
import os


def identify_column_types(df):
    """Identifie les colonnes numériques et catégorielles."""
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    return num_cols, cat_cols

def fill_missing_values(train_df, test_df, num_cols, cat_cols):
    """Remplace les valeurs manquantes dans les colonnes numériques par 0 et les colonnes catégorielles par 'Inconnu'."""
    for col in num_cols:
        train_df[col] = train_df[col].fillna(0)
        test_df[col] = test_df[col].fillna(0)
    for col in cat_cols:
        train_df[col] = train_df[col].fillna("Inconnu")
        test_df[col] = test_df[col].fillna("Inconnu")
    return train_df, test_df

def encode_categorical_features(train_df, test_df, cat_cols):
    """Encode les variables catégorielles avec CountEncoder."""
    encoder = CountEncoder(cols=cat_cols)
    train_df = encoder.fit_transform(train_df)
    test_df = encoder.transform(test_df)
    return train_df, test_df

def drop_columns_with_high_missing_or_low_variance(train_df, test_df, num_cols, threshold=0.4, var_threshold=0.01, corr_threshold=0.95):
    """
    Supprime les colonnes avec :
    - Trop de valeurs manquantes simulées (0 ou 'Inconnu')
    - Faible variance
    - Forte corrélation
    """
    columns_to_drop = []

    for col in train_df.columns:
        if col in num_cols:
            missing_ratio = (train_df[col] == 0).sum() / len(train_df)
        else:
            missing_ratio = (train_df[col] == "Inconnu").sum() / len(train_df)
        if missing_ratio > threshold:
            columns_to_drop.append(col)

    train_df.drop(columns=columns_to_drop, inplace=True)
    test_df.drop(columns=columns_to_drop, inplace=True)

    # Faible variance
    low_variance = train_df.var()
    low_var_cols = low_variance[low_variance < var_threshold].index.tolist()
    train_df.drop(columns=low_var_cols, inplace=True)
    test_df.drop(columns=low_var_cols, inplace=True)

    # Forte corrélation
    corr_matrix = train_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]
    train_df.drop(columns=high_corr_cols, inplace=True)
    test_df.drop(columns=high_corr_cols, inplace=True)

    return train_df, test_df, columns_to_drop + low_var_cols + high_corr_cols

def save_cleaned_data(train_df, test_df, train_path='train_cleaned.csv', test_path='test_cleaned.csv'):
    """Sauvegarde les fichiers nettoyés."""
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)