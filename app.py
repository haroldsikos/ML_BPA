import streamlit as st
import joblib
import pandas as pd
import numpy as np
# import os # Ya no es estrictamente necesario si no vas a usar os.getcwd()/os.listdir()

# --- Configuración de la página ---
st.set_page_config(
    page_title="G1 - 3997 - Predicción de Ventas de Avisos Publicitarios",
    page_icon="📈",
    layout="centered"
)

# --- Título de la aplicación ---
st.title('📈 Predicción de Ventas por Avisos Publicitarios')
st.markdown("---")
st.write('Ingresa la inversión en TV, Radio y Periódico para estimar las ventas de tu tienda.')

# --- Cargar el modelo y el escalador ---
# Asegúrate de que 'modelo_regresion_lineal.pkl' y 'scaler.pkl'
# estén en el mismo directorio que este script.
try:
    modelo_rl = joblib.load('modelo_regresion_lineal.pkl')
    scaler = joblib.load('scaler.pkl')
    # Si la carga es exitosa, no imprimimos nada, simplemente continuamos.
except FileNotFoundError:
    # Este error se mantendrá visible si los archivos no se encuentran,
    # ya que es un error crítico para el funcionamiento de la app.
    st.error("❌ Error: Archivos de modelo o escalador no encontrados.")
    st.info("Por favor, asegúrate de que 'modelo_regresion_lineal.pkl' y 'scaler.pkl' estén en el mismo directorio que 'app.py' o ajusta la ruta si es necesario.")
    st.stop() # Detiene la ejecución de la app si los archivos no están.

# --- Definir las características de entrada ---
# Deben ser las mismas que usaste para entrenar el modelo
num_features = ['TV', 'Radio', 'Newspaper']

# --- Crear widgets de entrada para el usuario ---
st.header('Monto de Inversión ($)')

# Los rangos de los sliders pueden ajustarse según tus datos históricos
tv_input = st.slider('Inversión en **TV**', min_value=0.0, max_value=300.0, value=150.0, step=1.0)
radio_input = st.slider('Inversión en **Radio**', min_value=0.0, max_value=50.0, value=25.0, step=0.1)
newspaper_input = st.slider('Inversión en **Periódico**', min_value=0.0, max_value=120.0, value=60.0, step=0.1)

st.markdown("---")

# --- Botón para hacer la predicción ---
if st.button('🚀 Predecir Ventas'):
    # Preparar los datos de entrada en un DataFrame, como lo hace tu notebook
    datos_nuevos = pd.DataFrame([[tv_input, radio_input, newspaper_input]], columns=num_features)

    # Realizar la transformación con el escalador cargado
    datos_nuevos_scaled = scaler.transform(datos_nuevos)

    # Hacer la predicción con el modelo cargado
    prediccion = modelo_rl.predict(datos_nuevos_scaled)

    # Mostrar el resultado de la predicción
    st.success(f'✨ La predicción de las ventas esperadas es: **${prediccion[0]:,.2f}**')
    st.markdown("""
    <style>
    .stSuccess {
        background-color: #e6ffe6;
        border-left: 8px solid #4CAF50;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True) # Pequeño estilo para el mensaje de éxito

st.markdown("---")
st.info("Este modelo predice las ventas basándose en la inversión en diferentes canales publicitarios.")