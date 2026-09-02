from pathlib import Path

import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
except ModuleNotFoundError:
    RandomForestRegressor = None
    mean_absolute_error = None


RUTA_ENTRADA = Path("demanda_predictiva_takeshi.csv")
CARPETA_RESULTADOS = Path("resultados")
RUTA_PREDICCIONES = CARPETA_RESULTADOS / "random_forest_inicial_jesus.csv"
RUTA_METRICAS = CARPETA_RESULTADOS / "metricas_modelos_jesus.csv"

VARIABLES_MODELO = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "promedio_movil_4_correcto",
    "promedio_movil_8_correcto",
    "mes",
    "semana_del_anio",
    "trimestre",
]
VARIABLE_OBJETIVO = "demanda_real"


def calcular_wape(real, prediccion):
    """Calcula WAPE evitando division entre cero."""
    suma_real = real.sum()
    if suma_real == 0:
        raise ValueError("No se puede calcular WAPE porque la suma de demanda_real es 0.")

    return (real - prediccion).abs().sum() / suma_real * 100


def calcular_tqs(real, prediccion):
    """Calcula la tasa de quiebre de stock."""
    quiebre = (real > prediccion).astype(int)
    return quiebre.mean() * 100


def main():
    # 0. Verificar que scikit-learn este instalado antes de entrenar.
    if RandomForestRegressor is None or mean_absolute_error is None:
        print("Falta la dependencia: scikit-learn")
        print("No se entreno el modelo ni se generaron resultados.")
        return

    # 1. Leer el archivo preparado por Takeshi.
    df = pd.read_csv(RUTA_ENTRADA)

    # 2. Validar que existan las columnas necesarias.
    columnas_necesarias = [
        "semana",
        VARIABLE_OBJETIVO,
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4",
        "mes",
        "semana_del_anio",
        "trimestre",
    ]
    columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas necesarias: {columnas_faltantes}")

    # 3. Convertir semana a datetime y ordenar cronologicamente.
    df["semana"] = pd.to_datetime(df["semana"], errors="coerce")
    df = df.sort_values("semana").reset_index(drop=True)

    # 4. Validar que las semanas esten ordenadas cronologicamente.
    if not df["semana"].is_monotonic_increasing:
        raise ValueError("Las semanas no quedaron ordenadas cronologicamente.")

    # 5. Recalcular promedios moviles sin fuga de informacion.
    #    Se usa shift(1) para considerar solo semanas anteriores.
    df["promedio_movil_4_correcto"] = (
        df[VARIABLE_OBJETIVO].shift(1).rolling(window=4).mean()
    )
    df["promedio_movil_8_correcto"] = (
        df[VARIABLE_OBJETIVO].shift(1).rolling(window=8).mean()
    )

    # 6. Eliminar unicamente filas con nulos en variables necesarias para el modelo.
    columnas_evaluacion = ["semana", VARIABLE_OBJETIVO] + VARIABLES_MODELO
    df_modelo = df.dropna(subset=columnas_evaluacion).copy()

    if df_modelo.empty:
        raise ValueError("No hay filas validas para entrenar despues de eliminar nulos.")

    # 7. Dividir cronologicamente: primeras 80% semanas train, ultimas 20% test.
    indice_corte = int(len(df_modelo) * 0.8)
    if indice_corte == 0 or indice_corte == len(df_modelo):
        raise ValueError("No hay suficientes filas para separar entrenamiento y prueba.")

    train = df_modelo.iloc[:indice_corte].copy()
    test = df_modelo.iloc[indice_corte:].copy()

    # 8. Separar variables predictivas y variable objetivo.
    x_train = train[VARIABLES_MODELO]
    y_train = train[VARIABLE_OBJETIVO]
    x_test = test[VARIABLES_MODELO]
    y_test = test[VARIABLE_OBJETIVO]

    # 9. Entrenar Random Forest inicial sin optimizacion de hiperparametros.
    modelo = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(x_train, y_train)

    # 10. Predecir solo sobre test y evitar predicciones negativas.
    test["prediccion_random_forest"] = modelo.predict(x_test)
    test["prediccion_random_forest"] = test["prediccion_random_forest"].clip(lower=0)

    # 11. Calcular linea base sobre exactamente las mismas semanas de test.
    test["prediccion_baseline"] = test["lag_1"]

    # 12. Calcular errores absolutos.
    test["error_baseline"] = (
        test[VARIABLE_OBJETIVO] - test["prediccion_baseline"]
    ).abs()
    test["error_random_forest"] = (
        test[VARIABLE_OBJETIVO] - test["prediccion_random_forest"]
    ).abs()

    # 13. Calcular quiebre de stock para baseline y modelo.
    test["quiebre_baseline"] = (
        test[VARIABLE_OBJETIVO] > test["prediccion_baseline"]
    ).astype(int)
    test["quiebre_modelo"] = (
        test[VARIABLE_OBJETIVO] > test["prediccion_random_forest"]
    ).astype(int)

    # 14. Calcular metricas comparables sobre el mismo conjunto de prueba.
    mae_baseline = mean_absolute_error(y_test, test["prediccion_baseline"])
    wape_baseline = calcular_wape(y_test, test["prediccion_baseline"])
    tqs_baseline = calcular_tqs(y_test, test["prediccion_baseline"])

    mae_rf = mean_absolute_error(y_test, test["prediccion_random_forest"])
    wape_rf = calcular_wape(y_test, test["prediccion_random_forest"])
    tqs_rf = calcular_tqs(y_test, test["prediccion_random_forest"])

    mejora_wape = wape_baseline - wape_rf
    mejora_tqs = tqs_baseline - tqs_rf

    # 15. Guardar predicciones y metricas.
    CARPETA_RESULTADOS.mkdir(exist_ok=True)

    columnas_salida = [
        "semana",
        VARIABLE_OBJETIVO,
        "lag_1",
        "prediccion_baseline",
        "prediccion_random_forest",
        "error_baseline",
        "error_random_forest",
        "quiebre_baseline",
        "quiebre_modelo",
    ]
    test[columnas_salida].to_csv(RUTA_PREDICCIONES, index=False)

    metricas = pd.DataFrame(
        {
            "modelo": ["Linea Base", "Random Forest Inicial"],
            "MAE": [mae_baseline, mae_rf],
            "WAPE": [wape_baseline, wape_rf],
            "TQS": [tqs_baseline, tqs_rf],
        }
    )
    metricas.to_csv(RUTA_METRICAS, index=False)

    # 16. Ordenar importancias de variables de mayor a menor.
    importancias = pd.DataFrame(
        {
            "variable": VARIABLES_MODELO,
            "importancia": modelo.feature_importances_,
        }
    ).sort_values("importancia", ascending=False)

    # 17. Mostrar resumen claro en consola.
    print("=============================================")
    print("RANDOM FOREST INICIAL - JESUS")
    print("=============================================")
    print(f"Train: {len(train)} semanas")
    print(f"Test: {len(test)} semanas")
    print(
        "Rango train: "
        f"{train['semana'].min().date()} a {train['semana'].max().date()}"
    )
    print(
        "Rango test: "
        f"{test['semana'].min().date()} a {test['semana'].max().date()}"
    )
    print()
    print("LINEA BASE EN TEST")
    print(f"MAE: {mae_baseline:.2f}")
    print(f"WAPE: {wape_baseline:.2f}%")
    print(f"TQS: {tqs_baseline:.2f}%")
    print()
    print("RANDOM FOREST")
    print(f"MAE: {mae_rf:.2f}")
    print(f"WAPE: {wape_rf:.2f}%")
    print(f"TQS: {tqs_rf:.2f}%")
    print()
    print("MEJORA RESPECTO A BASELINE")
    print(f"WAPE: {mejora_wape:.2f} puntos porcentuales")
    print(f"TQS: {mejora_tqs:.2f} puntos porcentuales")
    print()
    print("IMPORTANCIA DE VARIABLES")
    for _, fila in importancias.iterrows():
        print(f"{fila['variable']}: {fila['importancia']:.4f}")
    print("=============================================")
    print(f"CSV predicciones: {RUTA_PREDICCIONES}")
    print(f"CSV metricas: {RUTA_METRICAS}")


if __name__ == "__main__":
    main()
