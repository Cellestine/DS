from flask import Flask
from flask_restx import Api, Resource
import pandas as pd
from models.loader import load_model_freq, load_model_montant
from models.input_schema import get_input_model_freq, get_input_model_montant
from models_pkls.frequence.model_to_pkl import CATEGORIAL_COLUMNS

# Init app
app = Flask(__name__)
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


@ns.route("/health")
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

        # Ajout des colonnes catégorielles manquantes avec valeur "Inconnu"
        for col in CATEGORIAL_COLUMNS:
            if col not in df.columns:
                df[col] = "Inconnu"

        prediction = model_freq.predict(df)[0]
        return {"prediction": float(prediction)}


@ns.route("/montant")
class PredictMontant(Resource):
    """Endpoint pour prédire le montant d'indemnisation."""

    @ns.expect(input_model_montant)
    def post(self):
        """Reçoit les données d'entrée, applique le modèle de montant et renvoie une prédiction.

        Returns
        -------
        dict
            La prédiction de montant.
        """
        payload = api.payload
        df = pd.DataFrame([payload])

        # Ajout des colonnes catégorielles manquantes avec valeur "Inconnu"
        for col in CATEGORIAL_COLUMNS:
            if col not in df.columns:
                df[col] = "Inconnu"

        prediction = model_montant.predict(df)[0]
        return {"prediction": float(prediction)}


if __name__ == "__main__":
    app.run(debug=True)
