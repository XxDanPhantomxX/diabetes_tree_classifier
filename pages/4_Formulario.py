import streamlit as st
import pandas as pd
from Diabetes import clf, feature_columns

st.title("Formulario de Diagnóstico Interactivo")

st.write("Selecciona los síntomas del paciente para predecir el diagnóstico de diabetes.")

# Mapeo consistente con el preprocesamiento (Yes->1, No->2)
map_YN = {"Si": 1, "No": 2}
class_names = ["Negativo", "Positivo"]  # 0 y 1 en el mapeo

# --- Widgets de Streamlit ---
col1, col2 = st.columns(2)
with col1:
    urin_val = st.selectbox("¿Orina con frecuencia? (Urinating often)", ["Si", "No"])
    weight_val = st.selectbox("¿Pérdida de peso? (Weight Loss)", ["Si", "No"])
with col2:
    slow_val = st.selectbox("¿Cicatrización lenta? (Slow Healing)", ["Si", "No"])
    fat_val = st.selectbox("¿Fatiga extrema? (Extreme Fatigue)", ["Si", "No"])

# Botón para predecir
if st.button("Predecir diagnóstico", type="primary"):
    # Construir el DataFrame de una fila con el mismo orden de columnas
    values_raw = {
        "Urinating often": urin_val,
        "Slow Healing": slow_val,
        "Weight Loss": weight_val,
        "Extreme Fatigue": fat_val
    }
    
    # Codificar Yes/No como 1/2
    row = {k: map_YN[v] for k, v in values_raw.items()}
    X_new = pd.DataFrame([row], columns=feature_columns)

    # Predicción
    y_pred = clf.predict(X_new)[0]
    
    # Probabilidades
    p = clf.predict_proba(X_new)[0]
    
    # Render del resultado
    label = class_names[y_pred]
    
    if y_pred == 1: # Positivo
        st.success(f"**Diagnóstico: {label}**")
        st.write(f"Probabilidad de ser positivo: {p[1]:.2%}")
    else: # Negativo
        st.error(f"**Diagnóstico: {label}**")
        st.write(f"Probabilidad de ser negativo: {p[0]:.2%}")

    st.info("Este es un diagnóstico preliminar basado en un modelo de Árbol de Decisión y no reemplaza la consulta médica profesional.")
