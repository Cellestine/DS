from flask_restx import fields
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask
from flask_restx import Api, Resource
import pandas as pd
from models.loader import load_model_freq, load_model_montant
from models.input_schema import get_input_model_charge, get_input_model_charge_bis, get_input_model_freq, get_input_model_montant
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
input_model_charge_bis = get_input_model_charge_bis(api)


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
        """Reçoit les données d'entrée, applique le modèle de montant et renvoie une prédiction.
        
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

    @ns.expect(input_model_montant)
    def post(self):
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
class PredictCharges(Resource):
    """Endpoint pour prédire la charge."""

    @ns.expect(input_model_charge)
    def post(self):
        """Reçoit les données d'entrée, applique la formule de calcul pour la charge et renvoie le résultat.
        
        Returns
        -------
        dict
            La charge.
        """
        data = api.payload
        charges = data["frequence"] * data["montant"] * data["annee_survenance"]
        return {"charge": float(charges)}

@ns.route("/charge_bis")
class PredictCharges(Resource):
    """Endpoint pour prédire la charge."""

    @ns.expect(input_model_charge_bis)
    def post(self):
        """Reçoit les données d'entrée, applique les modèles de prédictions freq et montant. Puis applique la formule de calcul de la charge et renvoie le résultat.
        
        Returns
        -------
        dict
            La charge.
        """
        payload = api.payload
        df = pd.DataFrame([payload])

        # Ajout des colonnes manquantes pour les 2 modèles
        for col in CATEGORIAL_COLUMNS:
            if col not in df.columns:
                df[col] = "Inconnu"

        # Prédictions
        freq = model_freq.predict(df)[0]
        montant = model_montant.predict(df)[0]
        annee = df["annee_survenance"].values[0]

        charge = freq * montant * annee

        return {
            "frequence": float(freq),
            "montant": float(montant),
            "annee_survenance": float(annee),
            "charge": float(charge)
        }


if __name__ == "__main__":
    app.run(debug=True)
