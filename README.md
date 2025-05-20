
---

## Prérequis

- Python **3.8+**  
- Virtualenv ou Conda  
- (optionnel) MongoDB pour la persistance  

---

## 1. Installation

1. **Cloner le dépôt**  
    ```bash
    git clone https://github.com/ton-orga/api-incendie.git
    cd api-incendie
    ```

2. **Créer un environnement virtuel**  
    ```bash
    python -m venv venv
    source venv/bin/activate    # Unix/macOS
    venv\Scripts\activate       # Windows
    ```

3. **Installer les dépendances**  
    ```bash
    pip install -r requirements.txt
    ```

---

## 2. Configuration

1. **Chemins modèles**  
   - `models/loader.py` → ajuster `path` des .pkl

2. **Configurations des colonnes**  
   - Fréquence : `models/config.py` (`CATEGORIAL_COLUMNS`)  
   - Montant   : `models/config_montant.py` (`CATEGORICAL_COLUMNS_MONTANT`, `ORDINAL_COLUMNS_MONTANT`)

---

## 3. Démarrage de l’API

    ```bash
    python API/app.py
    ```

---

## 4. Documentation Swagger
    Visiter :
    http://localhost:5000/swagger/
    pour tester les endpoints et consulter les schémas d’entrée/sortie.

---

## 5. Endpoints
1. **Route Health**    
    - GET /health
    - Vérifie que l’API fonctionne.
    Réponse 200
    
    ```bash
    { "status": "ok", "message": "API is up and running!" }
    ```
    
2. **Route predict/freq** 
    - POST /predict/freq
    - Prédit la fréquence annuelle d’incendie.

    Payload : selon InputFreq (Swagger)

    Réponse 200 :
    ```bash
    { "prediction": 0.05 }
    ```

3. **Route predict/montant** 
    - POST /predict/montant
    - Prédit le montant moyen d’indemnisation.
    
    Payload : selon InputMontant (Swagger)

    Réponse 200 :
    ```bash
    { "prediction": 120000.0 }
    ```

4. **Route predict/charge** 
    POST /predict/charge
    Pipeline complet + calcul de la charge.
    Calcule la charge = fréquence × montant × années.

    Payload : 
    ```bash
    { "frequence": 0.05, "montant": 120000.0, "annee_survenance": 5 }
    ```
    
    Réponse 200 :
    ```bash
    {
    "frequence": 0.05,
    "montant": 120000.0,
    "annee_survenance": 5.0,
    "charge": 30000.0
    }
    ```
