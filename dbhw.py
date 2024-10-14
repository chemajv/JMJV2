import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd

# Cargar la base de datos del Titanic
@st.cache_data
def load_data():
    return sns.load_dataset("titanic")

titanic = load_data()

# Configuración de la página
st.set_page_config(page_title="Titanic Dashboard", layout="wide")

# Definir las opciones de la barra lateral
page = st.sidebar.selectbox("Seleccione una opción", ["Inicio", "Gráficos Seaborn", "Gráficos Plotly", "Estadísticas"])

st.title(f"Bienvenido a la página {page}!")

# Página de inicio
if page == "Inicio":
    st.write("Esta es la página de inicio. Seleccione una página desde la barra lateral para ver los gráficos.")

    st.subheader("Estadísticas Generales del Titanic")
    st.metric(label="Promedio de Edad", value=round(titanic['age'].mean(), 2))
    st.metric(label="Porcentaje de Supervivientes", value=round(titanic['survived'].mean() * 100, 2))
    st.metric(label="Total de Pasajeros", value=len(titanic))

# Gráficos Seaborn
if page == "Gráficos Seaborn":
    st.header("Gráficos con Seaborn")

    col1, col2 = st.columns(2)

    # Gráfico de dispersión
    with col1:
        st.subheader("Gráfico de Dispersión (Edad vs Tarifa)")
        fig, ax = plt.subplots()
        sns.scatterplot(data=titanic, x="age", y="fare", hue="survived", ax=ax)
        st.pyplot(fig)

    # Histograma
    with col2:
        st.subheader("Histograma de Edades")
        fig, ax = plt.subplots()
        sns.histplot(data=titanic, x="age", bins=20, kde=True, ax=ax)
        st.pyplot(fig)

    # Heatmap
    st.subheader("Mapa de calor (Matriz de correlación)")
    corr_matrix = titanic[['age', 'fare', 'survived']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)