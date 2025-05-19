from flask_restx import fields
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask
from flask_restx import Api, Resource
import pandas as pd
from models.loader import load_model_freq, load_model_montant
from models.input_schema import get_input_model_charge, get_input_model_freq, get_input_model_montant
from models_pkls.frequence.model_to_pkl import ColumnSelector, MissingValueFiller, ManualCountEncoder, ColumnDropper, ScalerWrapper
from models.config import CATEGORIAL_COLUMNS
from models.config_montant import CATEGORICAL_COLUMNS_MONTANT, ORDINAL_COLUMNS_MONTANT
from flask_cors import CORS

# Init app
app = Flask(__name__)
CORS(app)
api = Api(
    app,
    version="1.0",
    title="API FREQ",
    description="API de prédiction de fréquence et de montant d'incendie",
)
ns = api.namespace("predict", description="Opérations de prédiction")

# Charger les modèles
model_freq = load_model_freq()
model_montant = load_model_montant()

# Charger schémas Swagger
input_model_freq = get_input_model_freq(api)
input_model_montant = get_input_model_montant(api)
input_model_charge = get_input_model_charge(api)
input_model_charge_bis = get_input_model_charge(api)


@api.route("/health")
class HealthCheck(Resource):
    """Endpoint pour vérifier que l'API est opérationnelle."""

    def get(self):
        """Renvoie un message de confirmation que l'API fonctionne.

        Returns
        -------
        dict
            Statut et message de confirmation.
        int
            Code HTTP 200.
        """
        return {"status": "ok", "message": "API is up and running!"}, 200


@ns.route("/freq")
class PredictFreq(Resource):
    """Endpoint pour prédire la fréquence d'incendie."""

    @ns.expect(input_model_freq)
    def post(self):
        """Reçoit les données d'entrée, applique le modèle de fréquence et renvoie une prédiction.
        
        Returns
        -------
        dict
            La prédiction de fréquence.
        """
        payload = api.payload
        df = pd.DataFrame([payload])

        # S'assurer que toutes les colonnes catégorielles sont là
        from models.config import CATEGORIAL_COLUMNS  # <- déjà importé

        for col in CATEGORIAL_COLUMNS:
            if col not in df.columns:
                df[col] = "Inconnu"
            else:
                df[col] = df[col].astype(str)


        for col in df.columns:
            if col not in CATEGORIAL_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "surface_totale" not in df:
            df["surface_totale"] = df[["SURFACE1", "SURFACE4", "SURFACE10"]].sum(axis=1)

        if "capital_total" not in df:
            df["capital_total"] = df[["KAPITAL12", "KAPITAL25", "KAPITAL32"]].sum(axis=1)

        if "surface_par_batiment" not in df:
            df["surface_par_batiment"] = df["surface_totale"] / df["NBBAT1"].replace(0, pd.NA)

        if "capital_par_surface" not in df:
            df["capital_par_surface"] = df["capital_total"] / df["surface_totale"].replace(0, pd.NA)

        if "capital_moyen_par_batiment" not in df:
            df["capital_moyen_par_batiment"] = df["capital_total"] / df["NBBAT1"].replace(0, pd.NA)

        prediction = model_freq.predict(df)[0]
        return {"prediction": float(prediction)}



@ns.route("/montant")
class PredictMontant(Resource):
    """Endpoint pour prédire le montant."""

    @ns.expect(input_model_montant)
    def post(self):
        """Reçoit les données d'entrée, applique le modèle de montant et renvoie une prédiction.
        
        Returns
        -------
        dict
            La prédiction du montant.
        """
        payload = api.payload
        df = pd.DataFrame([payload])

        # 1) on crée ou on complète TOUTES les colonnes catégorielles
        for col in CATEGORICAL_COLUMNS_MONTANT:
            if col not in df.columns:
                # broadcast : df[col] devient une Series de "Inconnu"
                df[col] = "Inconnu"
            # ensuite on caste la Series complète en catégorie
            df[col] = df[col].astype("category")

        # 1.2) ordinales
        for col in ORDINAL_COLUMNS_MONTANT:
            df[col] = pd.to_numeric(df.get(col, -1), errors="coerce")

        # 1.3) tout le reste en numérique
        for col in df.columns:
            if col not in CATEGORICAL_COLUMNS_MONTANT and col not in ORDINAL_COLUMNS_MONTANT:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 2) on récupère la liste exacte des features que le booster attend
        booster = model_montant.get_booster()  # si c’est un XGBRegressor sklearn
        expected_feats = booster.feature_names

        # 3) on ajoute toutes les colonnes manquantes avec une valeur par défaut
        for feat in expected_feats:
            if feat not in df.columns:
                if feat in CATEGORICAL_COLUMNS_MONTANT:
                    df[feat] = "Inconnu"
                else:
                    # pour les ordinales ou numériques, on peut mettre -1 ou 0
                    df[feat] = -1

        # 4) enfin, on réordonne le df pour coller strictement à expected_feats
        df = df[expected_feats]

        # 5) prédiction sans plus d’erreur de feature_names
        pred = model_montant.predict(df)[0]
        return {"prediction": float(pred)}


@ns.route("/charge")
class PredictChargeBis(Resource):
    """Endpoint pour prédire la charge."""

    @ns.expect(input_model_charge)
    def post(self):
        """Reçoit les données d'entrée, applique les modèles de montant, de fréquence et renvoie une prédiction.
        
        Returns
        -------
        dict
            La prédiction de la charge.
        """
        payload = api.payload 
        df = pd.DataFrame([payload])

        # --- 1) Prépa pour freq (caté + num) ---
        for col in CATEGORIAL_COLUMNS:
            if col not in df.columns:
                df[col] = "Inconnu"
            # maintenant df[col] est une Series, on peut caster
            df[col] = df[col].astype(str)
        for col in df.columns:
            if col not in CATEGORIAL_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # mêmes calculs dérivés que dans /freq
        if "surface_totale" not in df:
            df["surface_totale"] = df[["SURFACE1","SURFACE4","SURFACE10"]].sum(axis=1)
        if "capital_total" not in df:
            df["capital_total"] = df[["KAPITAL12","KAPITAL25","KAPITAL32"]].sum(axis=1)
        if "surface_par_batiment" not in df:
            df["surface_par_batiment"] = df["surface_totale"]/df["NBBAT1"].replace(0,pd.NA)
        if "capital_par_surface" not in df:
            df["capital_par_surface"] = df["capital_total"]/df["surface_totale"].replace(0,pd.NA)
        if "capital_moyen_par_batiment" not in df:
            df["capital_moyen_par_batiment"] = df["capital_total"]/df["NBBAT1"].replace(0,pd.NA)

        # clone pour montant
        df_mont = df.copy()

        # --- 2) Prépa pour montant (caté en category + ord + num) ---
        for col in CATEGORICAL_COLUMNS_MONTANT:
            if col not in df_mont.columns:
                df_mont[col] = "Inconnu"
            df_mont[col] = df_mont[col].astype("category")
        for col in ORDINAL_COLUMNS_MONTANT:
            df_mont[col] = pd.to_numeric(df_mont.get(col, -1), errors="coerce")
        for col in df_mont.columns:
            if col not in CATEGORICAL_COLUMNS_MONTANT and col not in ORDINAL_COLUMNS_MONTANT:
                df_mont[col] = pd.to_numeric(df_mont[col], errors="coerce")

        # réaligner sur les features XGBoost du montant
        booster = model_montant.get_booster()
        feats = booster.feature_names
        for feat in feats:
            if feat not in df_mont.columns:
                df_mont[feat] = -1
        df_mont = df_mont[feats]

        # --- 3) Prédictions & calcul de la charge ---
        freq_pred    = model_freq.predict(df)[0]
        montant_pred = model_montant.predict(df_mont)[0]
        annee        = float(df["annee_survenance"].iat[0])

        charge = freq_pred * montant_pred * annee

        return {
            "frequence":        float(freq_pred),
            "montant":         float(montant_pred),
            "annee_survenance":  annee,
            "charge":          float(charge)
        }

if __name__ == "__main__":
    app.run(debug=True)