# maintenance-predictive

Système intelligent de maintenance prédictive industrielle — projet M1 Data Science (EFREI).
Prédiction d'une panne dans les 24h (`failure_within_24h`) à partir de données capteurs
(vibration, température, pression, RPM, mode opératoire...).

Le projet est découpé en deux blocs :

- **Bloc 1 — Data Science** : EDA, nettoyage/preprocessing, entraînement de plusieurs modèles
  ML, évaluation (SHAP / feature importance).
- **Bloc 2 — Industrialisation** : structure du projet, API FastAPI, dashboard Streamlit,
  rapport.

L'interface entre les deux blocs (features attendues, format du modèle, fichiers de
métadonnées) est décrite dans [CONTRACT.md](CONTRACT.md).

## Structure

```
api/        API FastAPI (/health, /predict, /model-info)
dashboard/  Dashboard décisionnel Streamlit (appelle l'API)
data/       Dataset (industrial_machine_maintenance.csv)
model/      Artefacts du modèle entraîné (model.pkl, metrics.json, feature_importance.json)
notebooks/  Notebooks d'EDA et d'expérimentation (Bloc 1)
src/        Code source partagé (preprocessing, training) (Bloc 1)
```

## Installation

```bash
pip install -r requirements.txt
```

## Lancer l'API

```bash
uvicorn api.main:app --reload
```

Documentation interactive : http://127.0.0.1:8000/docs

## Lancer le dashboard

L'API doit être démarrée au préalable — le dashboard l'appelle pour obtenir les prédictions
et les métriques des modèles.

```bash
streamlit run dashboard/app.py
```
