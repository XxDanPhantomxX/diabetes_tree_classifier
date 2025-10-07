import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Diabetes Prediction", layout="wide")
st.title("Diabetes Prediction")
st.write("Esta aplicación utiliza un Clasificador de Árbol de Decisión para predecir la diabetes basándose en ciertos síntomas.")
st.write("## Tabla de Datos 25 Pacientes")
st.write("Los síntomas considerados son: 'Orinar con frecuencia', 'Cicatrización lenta', 'Pérdida de peso' y 'Fatiga extrema'.")

# Define the data as per the image provided
data = {
    "Patient": [f"ø{i}" for i in range(1, 26)],
    "Urinating often": ["No", "Yes", "No", "No", "No", "Yes", "No", "No", "Yes", "Yes",
                        "Yes", "No", "No", "No", "Yes", "Yes", "Yes", "Yes", "No", "Yes",
                        "No", "Yes", "No", "Yes", "No"],
    "Slow Healing": ["Yes", "No", "No", "Yes", "No", "Yes", "No", "No", "Yes", "No",
                     "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "No", "Yes",
                     "Yes", "No", "Yes", "Yes", "Yes"],
    "Weight Loss": ["Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "No", "No", "Yes",
                    "Yes", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes",
                    "Yes", "Yes", "Yes", "No", "Yes"],
    "Extreme Fatigue": ["Yes", "No", "Yes", "No", "No", "Yes", "Yes", "No", "No", "Yes",
                        "No", "Yes", "Yes", "Yes", "No", "No", "Yes", "Yes", "Yes", "Yes",
                        "Yes", "Yes", "Yes", "Yes", "Yes"],
    "Result": ["Positive", "Positive", "Positive", "Negative", "Negative", "Positive", "Negative",
               "Negative", "Positive", "Positive", "Positive", "Positive", "Positive", "Positive",
               "Negative", "Negative", "Positive", "Positive", "Negative", "Positive", "Positive",
               "Negative", "Positive", "Positive", "Positive"]
}



# Create DataFrame
df_diabetes = pd.DataFrame(data)
#df_diabetes = pd.read_csv("/contents/datos_diabetes.csv")
if st.checkbox("Mostrar datos"):
  st.dataframe(df_diabetes)
# Save to CSV
#df.to_csv("datos_diabetes.csv", index=False)

def indiscernibility(attr, table):
    u_ind = {}  # un diccionario vacío para almacenar los elementos de la relación de indiscernibilidad (U/IND({conjunto de atributos}))
    attr_values = []  # una lista vacía para almacenar los valores de los atributos

    for i in table.index:
        attr_values = []
        for j in attr:
            attr_values.append(table.loc[i, j])  # encontrar el valor de la tabla en la fila correspondiente y el atributo deseado y agregarlo a la lista attr_values

        # convertir la lista en una cadena y verificar si ya es una clave en el diccionario
        key = ''.join(str(k) for k in attr_values)

        if key in u_ind:  # si la clave ya existe en el diccionario
            u_ind[key].add(i)
        else:  # si la clave aún no existe en el diccionario
            u_ind[key] = set()
            u_ind[key].add(i)

    # Ordenar la relación de indiscernibilidad por la longitud de cada conjunto
    u_ind_sorted = sorted(u_ind.values(), key=len, reverse=True)
    return u_ind_sorted

################################################################
# Agrupar pacientes por síntomas (relación de indiscernibilidad)

# Permitir al usuario seleccionar qué síntomas usar para agrupar (checkboxes)
st.write("## Selección de síntomas para agrupar pacientes")
st.write("Marca los síntomas que quieres considerar al agrupar pacientes por indiscernibilidad:")

col_a, col_b = st.columns(2)
options = ["Urinating often", "Slow Healing", "Weight Loss", "Extreme Fatigue"]
selected_attrs = []
with col_a:
  if st.checkbox("Urinating often", value=True):
    selected_attrs.append("Urinating often")
  if st.checkbox("Slow Healing", value=True):
    selected_attrs.append("Slow Healing")
with col_b:
  if st.checkbox("Weight Loss", value=True):
    selected_attrs.append("Weight Loss")
  if st.checkbox("Extreme Fatigue", value=True):
    selected_attrs.append("Extreme Fatigue")

if not selected_attrs:
  st.warning("Debe seleccionar al menos un síntoma. Se usarán todos por defecto.")
  selected_attrs = options.copy()

st.write(f"Sintomas seleccionados: {selected_attrs}")

# Calcular los grupos usando los atributos seleccionados
patient_groups = indiscernibility(selected_attrs, df_diabetes)

grouped_dataframes = [df_diabetes.iloc[list(group)] for group in patient_groups]

result_df = pd.concat(grouped_dataframes, keys=[f"Group {i+1}" for i in range(len(patient_groups))])

st.write("### Agrupados por síntomas similares (Relación de Indiscernibilidad)")
st.write("Los pacientes se agrupan según la similitud en sus síntomas. Cada grupo representa un conjunto de pacientes que comparten las mismas respuestas a los síntomas considerados.")
st.dataframe(result_df)

# Arbol de decisión

df=df_diabetes
# Encode the Yes/No values as 1 and 2
df_encoded = df.replace({"Yes": 1, "No": 2})

# Define feature columns and target
feature_columns = selected_attrs
X = df_encoded[feature_columns]
y = df_encoded["Result"].replace({"Positive": 1, "Negative": 0})  # Encode 'Result' for binary classification

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)

# Create and train the Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

###################   Aproximación Superior   #############################

def upper_approximation(R, X):  #We have to try to describe the knowledge in X with respect to the knowledge in R; both are LISTS OS SETS [{},{}]

  u_approx = set()  #change to [] if you want the result to be a list of sets

  #print("X : " + str(len(X)))
  #print("R : " + str(len(R)))

  for i in range(len(X)):
    for j in range(len(R)):

      if(R[j].intersection(X[i])):
        u_approx.update(R[j]) #change to .append() if you want the result to be a list of sets

  return u_approx


###################   Aproximación Inferior   #############################

def lower_approximation(R, X):  #We have to try to describe the knowledge in X with respect to the knowledge in R; both are LISTS OS SETS [{},{}]

  l_approx = set()  #change to [] if you want the result to be a list of sets

  #print("X : " + str(len(X)))
  #print("R : " + str(len(R)))

  for i in range(len(X)):
    for j in range(len(R)):

      if(R[j].issubset(X[i])):
        l_approx.update(R[j]) #change to .append() if you want the result to be a list of sets

  return l_approx

##################################   Resultados   #######################################

Lista = df_diabetes.columns
R = indiscernibility(['Urinating often', 'Slow Healing', 'Weight Loss', 'Extreme Fatigue'], df_diabetes)
X_diabetes_indices = [set(df_diabetes[df_diabetes['Result'] == 'Positive'].index.tolist())]
L=lower_approximation(R, X_diabetes_indices)
U=upper_approximation(R, X_diabetes_indices)
P = (U - L)
P = [P]

groupeds_dataframes = [df_diabetes.iloc[list(group)] for group in P]

results_df = pd.concat(groupeds_dataframes, keys=[f"Group {i+1}" for i in range(len(P))])

st.write("### Zona Frontera Lista Completa (Posible diagnóstico incierto)")
st.write("Los pacientes en esta sección presentan síntomas que los colocan en una zona de incertidumbre diagnóstica. Estos casos pueden requerir una evaluación médica más detallada para un diagnóstico preciso.")
st.dataframe(results_df)

#################################### Lista Reducto #######################################4

Lista = df_diabetes.columns
R = indiscernibility(selected_attrs, df_diabetes)
X_diabetes_indices = [set(df_diabetes[df_diabetes['Result'] == 'Positive'].index.tolist())]
L=lower_approximation(R, X_diabetes_indices)
U=upper_approximation(R, X_diabetes_indices)
P = (U - L)
P = [P]

groupeds_dataframes = [df_diabetes.iloc[list(group)] for group in P]

results_df = pd.concat(groupeds_dataframes, keys=[f"Group {i+1}" for i in range(len(P))])

st.write("### Zona Frontera Lista Reducto (Posible diagnóstico incierto)")
st.write("Los pacientes en esta sección presentan síntomas que los colocan en una zona de incertidumbre diagnóstica. Estos casos pueden requerir una evaluación médica más detallada para un diagnóstico preciso.")
st.dataframe(results_df)

################################ Link for the code in Google Colab #######################################
st.link_button("Open in Google Colab", "https://colab.research.google.com/drive/1zyWU-_bq86NlaodqM7Mb2rHUQNRhQaqN?usp=sharing")

################################ Quitando Variable de Perdida de Peso #######################################

df=df_diabetes
# Encode the Yes/No values as 1 and 2
df_encoded = df.replace({"Yes": 1, "No": 2})

# Define feature columns and target
featurez_columns = ["Urinating often", "Slow Healing", "Extreme Fatigue"]
X = df_encoded[featurez_columns]
y = df_encoded["Result"].replace({"Positive": 1, "Negative": 0})  # Encode 'Result' for binary classification

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Create and train the Decision Tree classifier
clfz = DecisionTreeClassifier(random_state=42)
clfz.fit(X1_train, y1_train)
