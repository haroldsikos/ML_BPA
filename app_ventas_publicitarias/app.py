import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os # ¡IMPORTANTE: Añade esta línea para usar os.getcwd() y os.listdir()!

# --- Configuración de la página ---
st.set_page_config(
    page_title="Predicción de Ventas de Avisos Publicitarios",
    page_icon="📈",
    layout="centered"
)

# --- Título de la aplicación ---
st.title('📈 Predicción de Ventas por Avisos Publicitarios')
st.markdown("---")
st.write('Ingresa la inversión en TV, Radio y Periódico para estimar las ventas de tu tienda.')

# --- INICIO DE SECCIÓN DE DEPURACIÓN ---
st.subheader("Información de Depuración (Eliminar después de resolver el error)")
current_dir = os.getcwd()
st.write(f"Directorio de trabajo actual (os.getcwd()): `{current_dir}`")

try:
    files_in_dir = os.listdir(current_dir)
    st.write("Archivos visibles en el directorio actual:")
    st.code(files_in_dir) # Usamos st.code para mejor visualización de la lista
    if 'modelo_regresion_lineal.pkl' in files_in_dir and 'scaler.pkl' in files_in_dir:
        st.success("¡Los archivos .pkl SÍ son visibles en el directorio actual!")
    else:
        st.warning("Advertencia: Los archivos .pkl NO son visibles en el directorio actual.")
        # Opcional: Si los archivos no se ven, podrías listar también el directorio padre si existiera
        # parent_dir = os.path.dirname(current_dir)
        # st.write(f"Archivos en el directorio padre ({parent_dir}):")
        # st.code(os.listdir(parent_dir))

except Exception as e:
    st.error(f"Error al intentar listar archivos en el directorio actual: {e}")
st.markdown("---") # Separador para la depuración
# --- FIN DE SECCIÓN DE DEPURACIÓN ---


# --- Cargar el modelo y el escalador ---
# Asegúrate de que 'modelo_regresion_lineal.pkl' y 'scaler.pkl'
# estén en el mismo directorio que este script.
try:
    modelo_rl = joblib.load('modelo_regresion_lineal.pkl')
    scaler = joblib.load('scaler.pkl')
    st.success("✅ Modelo y escalador cargados exitosamente por el código principal.") # Nuevo mensaje de éxito si carga
except FileNotFoundError:
    st.error("❌ Error: Archivos de modelo o escalador no encontrados por joblib.load().")
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