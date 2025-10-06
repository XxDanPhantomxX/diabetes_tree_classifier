from Diabetes import df_diabetes, clfz, featurez_columns, X1_train, X1_test, y1_train, y1_test
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

st.title("Arbol de Decisión para Predicción de Diabetes")


# Visualize the Decision Tree
fig, ax = plt.subplots(figsize=(15, 10))
plot_tree(clfz, filled=True, feature_names=featurez_columns, class_names=["Negative", "Positive"], ax=ax)
st.pyplot(fig)

st.write("## Importancia de Características")

# Serie ordenada con las importancias (suman 1 en árboles tipo CART)
importances = pd.Series(clfz.feature_importances_, index=featurez_columns)

# Gráfico (horizontal) de importancias
imp_sorted = importances.sort_values(ascending=True)  # para que la barra más importante quede arriba
plt.figure(figsize=(8, 5))
plt.barh(imp_sorted.index, imp_sorted.values)  # type: ignore # no especificamos colores ni estilos
plt.xlabel("Importancia (disminución de impureza de Gini)")
plt.title("Importancia de características – Árbol de Decisión")
plt.tight_layout()
plt.show()

# Tabla de apoyo (importancia y porcentaje)
tabla_imp = (
    importances.sort_values(ascending=False)
    .to_frame("Importance")
    .assign(Percent=lambda d: (d["Importance"]*100).round(1))
)
st.bar_chart(tabla_imp, horizontal=True)

st.write("## Matriz de Confusión")

# ---- Predicciones
y_pred_train = clfz.predict(X1_train)
y_pred_test  = clfz.predict(X1_test)

# ---- Matrices de confusión (conteos)
cm_train = confusion_matrix(y1_train, y_pred_train, labels=[0, 1])
cm_test  = confusion_matrix(y1_test,  y_pred_test,  labels=[0, 1])

# ---- Graficar matrices (conteos)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(cm_train, display_labels=["Negative", "Positive"]).plot(ax=axes[0], values_format='d', colorbar=False)
axes[0].set_title("Confusión (Train)")

ConfusionMatrixDisplay(cm_test, display_labels=["Negative", "Positive"]).plot(ax=axes[1], values_format='d', colorbar=False)
axes[1].set_title("Confusión (Test)")

st.pyplot(fig)

# ---- Matrices normalizadas por clase verdadera (filas suman 1)
cm_train_norm = confusion_matrix(y1_train, y_pred_train, labels=[0, 1], normalize='true')
cm_test_norm  = confusion_matrix(y1_test,  y_pred_test,  labels=[0, 1], normalize='true')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(cm_train_norm, display_labels=["Negative", "Positive"]).plot(ax=axes[0], values_format='.2f', colorbar=False)
axes[0].set_title("Confusión Normalizada (Train)")

ConfusionMatrixDisplay(cm_test_norm, display_labels=["Negative", "Positive"]).plot(ax=axes[1], values_format='.2f', colorbar=False)
axes[1].set_title("Confusión Normalizada (Test)")

st.pyplot(fig)
