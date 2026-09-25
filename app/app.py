import json

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "artifacts/flight_price_model.joblib"
META_PATH = "artifacts/flight_price_model.meta.json"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_metadata():
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


model = load_model()
meta = load_metadata()

st.title("Prédiction du prix d'un billet d'avion")
st.write("Renseignez les caractéristiques du vol pour estimer son prix.")

inputs = {}

st.subheader("Caractéristiques catégorielles")
for col in meta["discrete_variables"]:
    options = meta["categories"][col]
    inputs[col] = st.selectbox(col, options)

st.subheader("Caractéristiques numériques")
for col in meta["continuous_variables"]:
    low, high = meta["ranges"][col]
    default = (low + high) / 2
    inputs[col] = st.number_input(
        col, min_value=float(low), max_value=float(high), value=float(default)
    )

if st.button("Prédire le prix"):
    input_df = pd.DataFrame([inputs])[meta["features"]]
    prediction = model.predict(input_df)[0]

    st.success(f"Prix estimé : {prediction:,.2f}")

    st.subheader("Informations utilisées pour la prédiction")
    st.table(input_df.T.rename(columns={0: "Valeur"}))