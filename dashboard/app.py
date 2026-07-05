from pathlib import Path

import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "industrial_machine_maintenance.csv"

NUMERIC_FEATURES = [
    "vibration_rms",
    "temperature_motor",
    "current_phase_avg",
    "pressure_level",
    "rpm",
    "hours_since_maintenance",
    "ambient_temp",
]
MACHINE_TYPES = ["CNC", "Pump", "Compressor", "Robotic Arm"]
OPERATING_MODES = ["idle", "normal", "peak"]

st.set_page_config(page_title="Maintenance Prédictive", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def call_api(api_url: str, endpoint: str, method: str = "get", payload: dict | None = None):
    url = api_url.rstrip("/") + endpoint
    try:
        response = (
            requests.post(url, json=payload, timeout=5)
            if method == "post"
            else requests.get(url, timeout=5)
        )
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


st.sidebar.title("Maintenance Prédictive")
api_url = st.sidebar.text_input("URL de l'API", value="http://127.0.0.1:8000")

health, health_error = call_api(api_url, "/health")
if health_error:
    st.sidebar.error(f"API inaccessible : {health_error}")
elif health.get("model_loaded"):
    st.sidebar.success("API connectée — modèle chargé")
else:
    st.sidebar.warning("API connectée — modèle non chargé")

df = load_data()

tab_overview, tab_predict, tab_models, tab_importance = st.tabs(
    ["Vue d'ensemble des données", "Prédiction en temps réel", "Comparaison des modèles", "Importance des variables"]
)

with tab_overview:
    st.header("Vue d'ensemble des données capteurs")

    col1, col2, col3 = st.columns(3)
    col1.metric("Machines", df["machine_id"].nunique())
    col2.metric("Enregistrements", len(df))
    col3.metric("Taux de panne (24h)", f"{df['failure_within_24h'].mean():.1%}")

    st.subheader("Distribution des capteurs")
    feature = st.selectbox("Variable", NUMERIC_FEATURES)
    fig, ax = plt.subplots()
    sns.histplot(df, x=feature, hue="failure_within_24h", bins=40, ax=ax, multiple="stack")
    st.pyplot(fig)

    st.subheader("Taux de panne par type de machine et mode opératoire")
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(df.groupby("machine_type")["failure_within_24h"].mean())
    with col2:
        st.bar_chart(df.groupby("operating_mode")["failure_within_24h"].mean())

    st.subheader("Corrélations entre capteurs")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[NUMERIC_FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with tab_predict:
    st.header("Simuler un scénario machine")
    st.caption("Renseignez les paramètres machine pour obtenir une prédiction en temps réel via l'API.")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            machine_type = st.selectbox("Type de machine", MACHINE_TYPES)
            operating_mode = st.selectbox("Mode opératoire", OPERATING_MODES)
            hours_since_maintenance = st.number_input("Heures depuis la dernière maintenance", min_value=0.0, value=200.0)
        with col2:
            vibration_rms = st.number_input("Vibration RMS", min_value=0.0, value=float(df["vibration_rms"].median()))
            temperature_motor = st.number_input("Température moteur (°C)", value=float(df["temperature_motor"].median()))
            ambient_temp = st.number_input("Température ambiante (°C)", value=float(df["ambient_temp"].median()))
        with col3:
            current_phase_avg = st.number_input("Courant phase moyen", value=float(df["current_phase_avg"].median()))
            pressure_level = st.number_input("Niveau de pression", min_value=0.0, value=float(df["pressure_level"].median()))
            rpm = st.number_input("RPM", min_value=0.0, value=float(df["rpm"].median()))

        submitted = st.form_submit_button("Prédire")

    if submitted:
        payload = {
            "machine_type": machine_type,
            "vibration_rms": vibration_rms,
            "temperature_motor": temperature_motor,
            "current_phase_avg": current_phase_avg,
            "pressure_level": pressure_level,
            "rpm": rpm,
            "operating_mode": operating_mode,
            "hours_since_maintenance": hours_since_maintenance,
            "ambient_temp": ambient_temp,
        }
        result, error = call_api(api_url, "/predict", method="post", payload=payload)

        if error:
            st.error(f"Erreur lors de l'appel à l'API : {error}")
        else:
            probability = result["probability"]
            if result["failure_within_24h"] == 1:
                st.error(f"⚠️ {result['message']} — probabilité de panne : {probability:.1%}")
            else:
                st.success(f"✅ {result['message']} — probabilité de panne : {probability:.1%}")
            st.progress(min(probability, 1.0))

with tab_models:
    st.header("Comparaison des modèles")
    model_info, error = call_api(api_url, "/model-info")

    if error:
        st.warning(f"Impossible de récupérer les métriques des modèles : {error}")
    else:
        metrics = model_info.get("metrics")
        if not metrics:
            st.info("Aucune métrique disponible pour le moment.")
        else:
            models_df = pd.DataFrame(metrics["models"]).set_index("name")
            st.dataframe(models_df, use_container_width=True)
            st.bar_chart(models_df[["accuracy", "precision", "recall", "f1", "roc_auc"]])
            st.caption(f"Modèle retenu : **{metrics.get('selected_model', 'n/a')}**")

with tab_importance:
    st.header("Importance des variables")
    model_info, error = call_api(api_url, "/model-info")

    if error:
        st.warning(f"Impossible de récupérer l'importance des variables : {error}")
    else:
        feature_importance = model_info.get("feature_importance")
        if not feature_importance:
            st.info("Aucune importance de variable disponible pour le moment.")
        else:
            st.caption(f"Méthode : {feature_importance.get('method', 'n/a')}")
            importance_df = pd.DataFrame(feature_importance["importances"]).set_index("feature")
            st.bar_chart(importance_df["importance"])
