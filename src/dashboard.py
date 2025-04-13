import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Configuración de la página
st.set_page_config(page_title="Dashboard Energético", layout="wide")

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("databproduction_clean.csv")
    df_predictions = pd.read_csv("predicciones_energia.csv")
    return df, df_predictions

df, df_predictions = load_data()

# Cargar modelo entrenado
@st.cache_resource
def load_model():
    with open('/home/gerardo/proyecto_energia/notebooks/best_gb_model.pkl', "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Sidebar para filtros
st.sidebar.header("Filtros")
selected_country = st.sidebar.selectbox("Selecciona un país", ["Colombia", "Brazil", "United States"])
selected_product = st.sidebar.selectbox("Selecciona un producto", df["PRODUCT"].unique())

# Filtrar datos
filtered_df = df[(df["COUNTRY"] == selected_country) & (df["PRODUCT"] == selected_product)]
filtered_predictions = df_predictions[(df_predictions["COUNTRY"] == selected_country) & (df_predictions["PRODUCT"] == selected_product)]

# Visualización de datos históricos
st.subheader(f"Datos históricos de {selected_product} en {selected_country}")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=filtered_df, x="YEAR", y="VALUE", marker="o", ax=ax)
plt.xlabel("Año")
plt.ylabel("Producción Energética")
st.pyplot(fig)

# Visualización de predicciones
st.subheader(f"Predicciones de {selected_product} en {selected_country}")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=filtered_predictions, x="MONTH", y="PREDICTED_VALUE", hue="YEAR", marker="o", ax=ax)
plt.xlabel("Mes")
plt.ylabel("Producción Predicha (GWh)")
plt.legend(title="Año")
st.pyplot(fig)

# Predicción personalizada
st.sidebar.subheader("Predicción personalizada")
year_input = st.sidebar.number_input("Año", min_value=2010, max_value=2026, step=1)
month_input = st.sidebar.number_input("Mes", min_value=1, max_value=12, step=1)

if st.sidebar.button("Predecir"):
    country_encoded = df_predictions[df_predictions["COUNTRY"] == selected_country]["COUNTRY_encoded"].iloc[0]
    product_encoded = df_predictions[df_predictions["PRODUCT"] == selected_product]["PRODUCT_encoded"].iloc[0]
    input_data = pd.DataFrame([[country_encoded, product_encoded, year_input, month_input]], columns=["COUNTRY_encoded", "PRODUCT_encoded", "YEAR", "MONTH"])
    prediction = model.predict(input_data)[0]
    st.sidebar.success(f"Predicción: {prediction:.2f} GWh")
