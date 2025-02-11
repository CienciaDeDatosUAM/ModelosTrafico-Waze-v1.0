import joblib
import pandas as pd
import numpy as np

def predecir_nivel_alerta(location_x, location_y, modelo_archivo, scaler_archivo='scaler.pkl', encoder_archivo='encoder.pkl'):
    """
    Predice el nivel de alerta basado en las coordenadas de ubicación.

    Parámetros:
    location_x (float): Coordenada de longitud de la ubicación.
    location_y (float): Coordenada de latitud de la ubicación.
    modelo_archivo (str): Ruta al archivo del modelo guardado (e.g., un clasificador entrenado).
    scaler_archivo (str, opcional): Ruta al archivo del escalador guardado (por defecto 'scaler.pkl').
    encoder_archivo (str, opcional): Ruta al archivo del codificador guardado (por defecto 'encoder.pkl').

    Retorna:
    str: Nivel de alerta predicho para la ubicación especificada.

    Proceso:
    1. Carga el modelo de predicción, el escalador y el codificador desde archivos.
    2. Escala las coordenadas de entrada usando el escalador guardado.
    3. Realiza la predicción usando el modelo cargado.
    4. Decodifica el nivel de alerta predicho si es necesario.
    """
    # Cargar el modelo, el escalador y el codificador guardado
    modelo = joblib.load(modelo_archivo)
    scaler = joblib.load(scaler_archivo)
    encoder = joblib.load(encoder_archivo)

    # Crear un DataFrame con las nuevas coordenadas
    nuevas_coordenadas = pd.DataFrame([[location_x, location_y]], columns=['longitud', 'latitud'])

    # Escalar las coordenadas
    nuevas_coordenadas_scaled = scaler.transform(nuevas_coordenadas)

    # Hacer la predicción
    prediccion = modelo.predict(nuevas_coordenadas_scaled)

    # Decodificar el nivel de alerta (si se codificó durante el preprocesamiento)
    nivel_alerta = encoder.inverse_transform(prediccion)

    return nivel_alerta[0]

# Pedir al usuario que ingrese las coordenadas
location_x = float(input("Introduce la coordenada de longitud: "))
location_y = float(input("Introduce la coordenada de latitud: "))

# Seleccionar el archivo del modelo a usar
modelo_archivo = "modelo_knn_2024-12-19.pkl"

# Realizar la predicción
resultado = predecir_nivel_alerta(location_x, location_y, modelo_archivo)

# Mostrar el resultado
print(f"Resultado de la predicción para la ubicación ({location_x}, {location_y}): {resultado}")

