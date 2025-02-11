from datetime import datetime
import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

fecha_actual = datetime.now().strftime("%Y-%m-%d")

# Verificar disponibilidad de GPU
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPUs disponibles para TensorFlow:", gpus)
    else:
        print("No se detectó GPU para TensorFlow. Usando CPU.")
except ImportError:
    print("TensorFlow no está instalado. Red Neuronal usará CPU si se elige.")

# Cargar datos
train = pd.read_csv('datos_finales_2019.csv')

# Preprocesar datos
le = LabelEncoder()
train['nivel_alertas'] = le.fit_transform(train['nivel_alertas'])
X = train[['longitud', 'latitud']]
y = train['nivel_alertas']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)
with open("encoder.pkl", "wb") as encoder_file:
    pickle.dump(le, encoder_file)
print("scaler.pkl y encoder.pkl guardados.\n")

def crear_red_neuronal(input_dim):
    """
    Crea y configura una red neuronal secuencial para clasificación multiclase.

    Parámetros:
    input_dim (int): Dimensión de entrada para las características (número de columnas de entrada).

    Retorna:
    tensorflow.keras.models.Sequential: Modelo de red neuronal configurado y compilado.
    """
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(len(y.unique()), activation='softmax')  # Salida para clasificación multiclase
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Modelos disponibles
modelos = {
    "Redes Neuronales": lambda: crear_red_neuronal(X_train.shape[1]),
    "Random Forest": lambda: RandomForestClassifier(),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=1),
    "XGBoost": lambda: XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', tree_method='gpu_hist' if gpus else 'auto')
}

# Elegir modelo
print("Modelos disponibles:")
for i, modelo in enumerate(modelos.keys(), 1):
    print(f"{i}. {modelo}")
opcion = int(input("Elige el modelo (1-4): "))

if opcion not in range(1, len(modelos) + 1):
    print("Opción no válida. Saliendo.")
    exit()

modelo_elegido = list(modelos.keys())[opcion - 1]
modelo = modelos[modelo_elegido]()

# Entrenar el modelo
if modelo_elegido == "Redes Neuronales":
    print(f"Entrenando {modelo_elegido}...")
    checkpoint = ModelCheckpoint('modelo_mejor.h5', save_best_only=True, monitor='val_loss', mode='min')
    modelo.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[checkpoint])
    print(f"Entrenamiento de {modelo_elegido} completado. Modelo guardado como 'modelo_mejor.h5'.")
else:
    print(f"Entrenando {modelo_elegido}...")
    modelo.fit(X_train, y_train)
    # Guardar el modelo entrenado
    nombre_archivo = f"modelo_{modelo_elegido.replace(' ', '_').lower()}_{fecha_actual}.pkl"
    with open(nombre_archivo, 'wb') as f:
        pickle.dump(modelo, f)
    print(f"Entrenamiento de {modelo_elegido} completado. Modelo guardado como '{nombre_archivo}'.")

