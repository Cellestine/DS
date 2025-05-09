from flask import Flask
from flask_restx import Api, Resource
import pandas as pd
from models.loader import load_model_freq, load_model_montant
from models.input_schema import get_input_model

# Init app
app = Flask(__name__)
api = Api(app, version="1.0", title="API FREQ", description="API prédiction de fréquence incendie")
ns = api.namespace("predict", description="Opérations de prédiction")

# Charger les modèles
model_freq = load_model_freq()
model_montant = load_model_montant()

# Charger schema Swagger
input_model = get_input_model(api)


@ns.route("/health")
class HealthCheck(Resource):
    def get(self):
        return {"status": "ok", "message": "API is up and running!"}, 200

@ns.route("/freq")
class Predict(Resource):
    @ns.expect(input_model)
    def post(self):
        payload = api.payload
        df = pd.DataFrame([payload])
        prediction = model_freq.predict(df)[0]
        return {"prediction": float(prediction)}
    
@ns.route("/montant")
class Predict(Resource):
    @ns.expect(input_model)
    def post(self):
        payload = api.payload
        df = pd.DataFrame([payload])
        prediction = model_montant.predict(df)[0]
        return {"prediction": float(prediction)}

if __name__ == "__main__":
    app.run(debug=True)
