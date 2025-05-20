
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
export FLASK_APP=app.py        # Unix/macOS
set FLASK_APP=app.py           # Windows
flask run --debug
