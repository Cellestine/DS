from flask import Flask
from flask_restx import Api, Resource
import pandas as pd
from models.loader import load_model_freq, load_model_montant
from models.input_schema import get_input_model_freq, get_input_model_montant
#from models_pkls.frequence.model_to_pkl import CATEGORIAL_COLUMNS
from models.config import CATEGORIAL_COLUMNS
from models.config_montant import CATEGORICAL_COLUMNS_MONTANT, ORDINAL_COLUMNS_MONTANT

# Init app
app = Flask(__name__)
api = Api(app, version="1.0", title="API FREQ", description="API prédiction de fréquence incendie")
ns = api.namespace("predict", description="Opérations de prédiction")

# Charger les modèles
model_freq = load_model_freq()
model_montant = load_model_montant()

# Charger schema Swagger
input_model_freq = get_input_model_freq(api)
input_model_montant = get_input_model_montant(api)


@ns.route("/health")
class HealthCheck(Resource):
    def get(self):
        return {"status": "ok", "message": "API is up and running!"}, 200

@ns.route("/freq")
class PredictFreq(Resource):
    @ns.expect(input_model_freq)
    def post(self):
        payload = api.payload
        df = pd.DataFrame([payload])

        # Ajout des colonnes catégorielles manquantes avec valeur "Inconnu"
        for col in CATEGORIAL_COLUMNS:
            df[col] = "Inconnu"  # ou une valeur par défaut raisonnable

        prediction = model_freq.predict(df)[0]
        return {"prediction": float(prediction)}

    
@ns.route("/montant")
class PredictMontant(Resource):
    @ns.expect(input_model_montant)
    def post(self):
        payload = api.payload
        df = pd.DataFrame([payload])

        # Ajout des colonnes catégorielles manquantes avec valeur "Inconnu"
        for col in CATEGORICAL_COLUMNS_MONTANT:
            df[col] = "Inconnu"  # ou une valeur par défaut raisonnable

        # Colonnes ordinales manquantes : -1
        for col in ORDINAL_COLUMNS_MONTANT:
            if col not in df.columns:
                df[col] = -1

        prediction = model_montant.predict(df)[0]
        return {"prediction": float(prediction)}
    
if __name__ == "__main__":
    app.run(debug=True)
