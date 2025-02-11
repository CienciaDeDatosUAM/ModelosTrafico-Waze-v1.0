import pandas as pd
import numpy as np
from cuml.model_selection import train_test_split
from cuml.ensemble import RandomForestClassifier
from cuml.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from cuml.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

# Cargar los datos
def cargar_datos(ruta):
    """
    Carga los datos desde un archivo CSV y los retorna como un DataFrame.

    Parámetros:
    ruta (str): Ruta del archivo CSV.

    Retorna:
    DataFrame: Los datos cargados.
    """
    return pd.read_csv(ruta)

# Codificar etiquetas
def codificar_etiquetas(datos, columna_objetivo):
    """
    Codifica las etiquetas de una columna usando LabelEncoder.

    Parámetros:
    datos (DataFrame): DataFrame que contiene la columna objetivo.
    columna_objetivo (str): Nombre de la columna objetivo.

    Retorna:
    tuple: DataFrame actualizado y un arreglo con las clases codificadas.
    """
    le = LabelEncoder()
    datos[columna_objetivo] = le.fit_transform(datos[columna_objetivo])
    clases = le.classes_
    return datos, clases

# Preparar características y variable objetivo
def preparar_datos(datos, columnas_caracteristicas, columna_objetivo):
    """
    Prepara las características (X) y la variable objetivo (y) de un DataFrame.

    Parámetros:
    datos (DataFrame): DataFrame con los datos.
    columnas_caracteristicas (list): Lista de columnas a usar como características.
    columna_objetivo (str): Nombre de la columna objetivo.

    Retorna:
    tuple: Arreglos de características escaladas y variable objetivo.
    """
    X = datos[columnas_caracteristicas].values
    y = datos[columna_objetivo].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Validación cruzada con cuML
def cross_validate_gpu_model(model, X, y, n_splits=5):
    """
    Realiza validación cruzada con GPU utilizando KFold y calcula la métrica F1.

    Parámetros:
    model (object): Modelo a evaluar.
    X (array-like): Características.
    y (array-like): Variable objetivo.
    n_splits (int): Número de divisiones para KFold.

    Retorna:
    float: Promedio de las puntuaciones F1.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        scores.append(f1_score(y_val, y_pred, average='macro'))

    return np.mean(scores)

# Main Script
if __name__ == "__main__":
    # Cargar los datos
    train = cargar_datos('datos_finales_2019_medias.csv')

    # Codificar etiquetas
    train, clases = codificar_etiquetas(train, 'nivel_alertas1')
    print("Clases codificadas: ", clases)
    print(train.groupby("nivel_alertas1").count())

    # Preparar X y y
    X_scaled, y = preparar_datos(train, ['longitud', 'latitud'], 'nivel_alertas1')

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Modelos para cuML
    modelos = {
        "Random Forest (GPU)": RandomForestClassifier(n_estimators=200),
        "KNN (GPU)": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(tree_method='gpu_hist', eval_metric='mlogloss', use_label_encoder=False),
    }

    # Validar modelos
    resultados = {}
    for nombre, modelo in modelos.items():
        print(f"Validando {nombre}...")
        score = cross_validate_gpu_model(modelo, X_scaled, y)
        resultados[nombre] = score

    # Mostrar resultados
    print("\nResultados de validación cruzada con GPU:")
    for modelo, score in resultados.items():
        print(f"{modelo}: {score:.4f}")




