from flask_restx import fields

def get_input_model(api):
    return api.model("Input", {
        "ID": fields.Integer(required=True),
        "TYPERS": fields.Integer(required=True),
        "ANCIENNETE": fields.Float(required=True),
        "DUREE_REQANEUF": fields.Float(required=True),
        "TYPBAT2": fields.Integer(required=True),
        "KAPITAL12": fields.Float(required=True),
        "KAPITAL25": fields.Float(required=True),
        "KAPITAL32": fields.Float(required=True),
        "SURFACE1": fields.Float(required=True),
        "SURFACE4": fields.Float(required=True),
        "SURFACE10": fields.Float(required=True),
        "NBBAT1": fields.Integer(required=True),
        "RISK1": fields.Integer(required=True),
        "RISK7": fields.Integer(required=True),
        "EQUIPEMENT4": fields.Integer(required=True),
        "EQUIPEMENT6": fields.Integer(required=True),
        "ZONE_VENT": fields.Float(required=True),
        "ANNEE_ASSURANCE": fields.Float(required=True),
        "AN_EXERC": fields.Integer(required=True),
        "ZONE": fields.Float(required=True),
        "surface_totale": fields.Float(required=True),
        "capital_total": fields.Float(required=True),
        "surface_par_batiment": fields.Float(required=True),
        "capital_par_surface": fields.Float(required=True),
        "capital_moyen_par_batiment": fields.Float(required=True),
        # Ajoute les autres champs ici si besoin
    })
