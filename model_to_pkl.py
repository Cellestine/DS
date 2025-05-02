# Importations nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import numpy as np
import joblib

# Chargement des données préparées (déjà nettoyées)
X_train_full = pd.read_csv('/train_features.csv')
y_train_full = pd.read_csv('/train_output_DzPxaPY.csv')['FREQ']
X_test = pd.read_csv('/test_features.csv')

pd.set_option('display.max_columns', 375)
display(X_train_full.head())
display(y_train_full.head())
display(X_test.head())


# Définir la grille des hyperparamètres à tester
param_grid = {
    'n_estimators': [100,200],
    'max_depth': [3, 5],
    'learning_rate': [0.1,0.05],
    'subsample': [0.6,0.8]
}

# Initialiser le modèle de base
xgb = XGBRegressor(objective="count:poisson", random_state=42)

# GridSearch avec validation croisée
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)

# Lancer la recherche
grid_search.fit(X_train_full, y_train_full)

# Meilleurs hyperparamètres
best_params = grid_search.best_params_
print("✅ Meilleurs paramètres trouvés :", best_params)

# Remplacer le modèle par le meilleur trouvé
model = XGBRegressor(objective="count:poisson", random_state=42, **best_params)


#-------------------------------------------------#


# 1. Entraîner le modèle avec les meilleurs paramètres
model.fit(X_train_full, y_train_full)

# 2. Évaluer (optionnel)
train_preds = model.predict(X_train_full)
rmse = mean_squared_error(y_train_full, train_preds, squared=False)
print(f"📉 RMSE sur train : {rmse:.4f}")

# 3. Sauvegarder le modèle entraîné
joblib.dump(model, "xgb_regressor_model.pkl")
print("✅ Modèle sauvegardé dans xgb_regressor_model.pkl")
