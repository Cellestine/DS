"""
Ce module contient les fonctions permettant d'inserer les inputs JSON.
"""

from flask_restx import fields


def get_input_model_freq(api):
    """
    Déclare le modèle d'entrée Swagger pour la prédiction de fréquence d'incendie.

    Ce modèle est utilisé pour documenter les paramètres requis de l'endpoint /freq dans Swagger UI.
    Toutes les variables numériques nécessaires sont déclarées ici.
    Les variables catégorielles sont gérées directement dans le pipeline, et donc omises ici.

    Parameters:
        api (flask_restx.Api): L'objet API Flask-RESTX utilisé pour attacher le modèle.

    Returns:
        fields.Model: Modèle Swagger décrivant les paramètres attendus.
    """
    return api.model(
        "InputFreq",
        {
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
            # ↓↓↓ Catégorielles ↓↓↓
            "ACTIVIT2": fields.String(required=True),
            "VOCATION": fields.String(required=True),
            "CARACT1": fields.String(required=True),
            "CARACT3": fields.String(required=True),
            "CARACT4": fields.String(required=True),
            "TYPBAT1": fields.String(required=True),
            "INDEM2": fields.String(required=True),
            "FRCH1": fields.String(required=True),
            "FRCH2": fields.String(required=True),
            "DEROG12": fields.String(required=True),
            "DEROG13": fields.String(required=True),
            "DEROG14": fields.String(required=True),
            "DEROG16": fields.String(required=True),
            "TAILLE1": fields.String(required=True),
            "TAILLE2": fields.String(required=True),
            "COEFASS": fields.String(required=True),
            "RISK6": fields.String(required=True),
            "RISK8": fields.String(required=True),
            "RISK9": fields.String(required=True),
            "RISK10": fields.String(required=True),
            "RISK11": fields.String(required=True),
            "RISK12": fields.String(required=True),
            "RISK13": fields.String(required=True),
            "EQUIPEMENT2": fields.String(required=True),
            "EQUIPEMENT5": fields.String(required=True),
        },
    )


def get_input_model_montant(api):
    """
    Déclare le modèle d'entrée Swagger pour la prédiction du montant des sinistres.

    Ce modèle est utilisé pour documenter les paramètres requis de l'endpoint /montant dans Swagger UI.
    Toutes les variables nécessaires à la prédiction du montant sont définies ici.

    Parameters:
        api (flask_restx.Api): L'objet API Flask-RESTX utilisé pour attacher le modèle.

    Returns:
        fields.Model: Modèle Swagger décrivant les paramètres attendus.
    """
    return api.model(
        "InputMontant",
        {
            "DEROG13": fields.Float(required=True),
            "DEROG14": fields.Float(required=True),
            "DEROG16": fields.Float(required=True),
            "ANCIENNETE": fields.Float(required=True),
            "CARACT2": fields.Float(required=True),
            "DUREE_REQANEUF": fields.Float(required=True),
            "CARACT5": fields.Float(required=True),
            "TYPBAT2": fields.Float(required=True),
            "DEROG1": fields.Float(required=True),
            "DEROG6": fields.Float(required=True),
            "DEROG7": fields.Float(required=True),
            "DEROG9": fields.Float(required=True),
            "DEROG10": fields.Float(required=True),
            "DEROG11": fields.Float(required=True),
            "DEROG15": fields.Float(required=True),
            "CA1": fields.Float(required=True),
            "CA2": fields.Float(required=True),
            "CA3": fields.Float(required=True),
            "KAPITAL1": fields.Float(required=True),
            "KAPITAL2": fields.Float(required=True),
            "KAPITAL3": fields.Float(required=True),
            "KAPITAL4": fields.Float(required=True),
            "KAPITAL5": fields.Float(required=True),
            "KAPITAL6": fields.Float(required=True),
            "KAPITAL7": fields.Float(required=True),
            "KAPITAL8": fields.Float(required=True),
            "KAPITAL9": fields.Float(required=True),
            "KAPITAL10": fields.Float(required=True),
            "KAPITAL11": fields.Float(required=True),
            "KAPITAL12": fields.Float(required=True),
            "KAPITAL13": fields.Float(required=True),
            "KAPITAL14": fields.Float(required=True),
            "KAPITAL15": fields.Float(required=True),
            "KAPITAL16": fields.Float(required=True),
            "KAPITAL17": fields.Float(required=True),
            "KAPITAL18": fields.Float(required=True),
            "KAPITAL19": fields.Float(required=True),
            "KAPITAL20": fields.Float(required=True),
            "KAPITAL21": fields.Float(required=True),
            "KAPITAL22": fields.Float(required=True),
            "KAPITAL23": fields.Float(required=True),
            "KAPITAL24": fields.Float(required=True),
            "KAPITAL25": fields.Float(required=True),
            "KAPITAL26": fields.Float(required=True),
            "KAPITAL27": fields.Float(required=True),
            "KAPITAL28": fields.Float(required=True),
            "KAPITAL29": fields.Float(required=True),
            "KAPITAL30": fields.Float(required=True),
            "KAPITAL31": fields.Float(required=True),
            "KAPITAL32": fields.Float(required=True),
            "KAPITAL33": fields.Float(required=True),
            "KAPITAL36": fields.Float(required=True),
            "KAPITAL38": fields.Float(required=True),
            "KAPITAL39": fields.Float(required=True),
            "SURFACE1": fields.Float(required=True),
            "SURFACE2": fields.Float(required=True),
            "SURFACE3": fields.Float(required=True),
            "SURFACE5": fields.Float(required=True),
            "SURFACE7": fields.Float(required=True),
            "SURFACE8": fields.Float(required=True),
            "SURFACE9": fields.Float(required=True),
            "SURFACE10": fields.Float(required=True),
            "SURFACE11": fields.Float(required=True),
            "SURFACE12": fields.Float(required=True),
            "SURFACE13": fields.Float(required=True),
            "SURFACE14": fields.Float(required=True),
            "SURFACE15": fields.Float(required=True),
            "SURFACE16": fields.Float(required=True),
            "SURFACE17": fields.Float(required=True),
            "SURFACE18": fields.Float(required=True),
            "SURFACE19": fields.Float(required=True),
            "SURFACE20": fields.Float(required=True),
            "SURFACE21": fields.Float(required=True),
            "NBBAT1": fields.Float(required=True),
            "NBBAT2": fields.Float(required=True),
            "NBBAT3": fields.Float(required=True),
            "NBBAT4": fields.Float(required=True),
            "NBBAT5": fields.Float(required=True),
            "NBBAT6": fields.Float(required=True),
            "NBBAT7": fields.Float(required=True),
            "NBBAT8": fields.Float(required=True),
            "NBBAT9": fields.Float(required=True),
            "NBBAT10": fields.Float(required=True),
            "NBBAT11": fields.Float(required=True),
            "NBBAT13": fields.Float(required=True),
            "NBBAT14": fields.Float(required=True),
            "TAILLE3": fields.Float(required=True),
            "TAILLE4": fields.Float(required=True),
            "NBSINCONJ": fields.Float(required=True),
            "NBSINSTRT": fields.Float(required=True),
            "RISK1": fields.Float(required=True),
            "RISK2": fields.Float(required=True),
            "RISK3": fields.Float(required=True),
            "RISK4": fields.Float(required=True),
            "RISK5": fields.Float(required=True),
            "RISK7": fields.Float(required=True),
            "EQUIPEMENT1": fields.Float(required=True),
            "EQUIPEMENT3": fields.Float(required=True),
            "EQUIPEMENT4": fields.Float(required=True),
            "EQUIPEMENT6": fields.Float(required=True),
            "EQUIPEMENT7": fields.Float(required=True),
            "ZONE_VENT": fields.Float(required=True),
        },
    )


def get_input_model_charge(api):
    """
    Construit et retourne le modèle Swagger (OpenAPI) pour l'endpoint de prédiction de la charge.

    Ce modèle fusionne les champs requis pour les prédictions de fréquence et de montant,
    et y ajoute un champ supplémentaire spécifique à la charge : `annee_survenance`.

    Paramètres
    ----------
    api : flask_restx.Api
        L'instance de l'API Flask-RestX utilisée pour déclarer le modèle.

    Retours
    -------
    Model
        Un modèle Flask-RestX combiné, nommé "InputCharge", contenant :
            - les champs requis pour la prédiction de fréquence,
            - les champs requis pour la prédiction de montant,
            - le champ `annee_survenance` de type float.
    """
    input_model_freq = get_input_model_freq(api)
    input_model_montant = get_input_model_montant(api)

    freq_fields = dict(api.models[input_model_freq.name].clone("freq_clone").items())
    montant_fields = dict(api.models[input_model_montant.name].clone("montant_clone").items())

    return api.model(
        "InputCharge",
        {
            "annee_survenance": fields.Float(required=True),
            **freq_fields,
            **montant_fields,
        },
    )
