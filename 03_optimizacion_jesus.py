from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RandomForestRegressor: Any = None
mean_absolute_error: Any = None
ParameterSampler: Any = None
TimeSeriesSplit: Any = None


RUTA_ENTRADA = Path("demanda_predictiva_takeshi.csv")
CARPETA_RESULTADOS = Path("resultados")
RUTA_PREDICCIONES = CARPETA_RESULTADOS / "random_forest_optimizado_jesus.csv"
RUTA_COMPARACION = CARPETA_RESULTADOS / "comparacion_modelos_jesus.csv"

VARIABLE_OBJETIVO = "demanda_real"
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

PESO_TQS_SCORE = 0.5
N_SPLITS_VALIDACION = 4
N_CONFIGURACIONES_BUSQUEDA = 80
FACTORES_SEGURIDAD = [1.00, 1.02, 1.05, 1.08, 1.10, 1.15, 1.20]
RANDOM_STATE = 42


def verificar_dependencias():
    # Verifica scikit-learn antes de entrenar o buscar hiperparametros.
    global RandomForestRegressor
    global mean_absolute_error
    global ParameterSampler
    global TimeSeriesSplit

    try:
        modulo_ensemble = import_module("sklearn.ensemble")
        modulo_metricas = import_module("sklearn.metrics")
        modulo_validacion = import_module("sklearn.model_selection")
    except ModuleNotFoundError as error:
        if error.name != "sklearn" and not str(error.name).startswith("sklearn."):
            raise
        print("Falta la dependencia: scikit-learn")
        print("No se entreno el modelo ni se generaron resultados.")
        return False

    RandomForestRegressor = modulo_ensemble.RandomForestRegressor
    mean_absolute_error = modulo_metricas.mean_absolute_error
    ParameterSampler = modulo_validacion.ParameterSampler
    TimeSeriesSplit = modulo_validacion.TimeSeriesSplit
    return True


def calcular_wape(real, prediccion):
    # WAPE global, con validacion para evitar division entre cero.
    real = pd.Series(real)
    prediccion = pd.Series(prediccion, index=real.index)
    suma_real = real.sum()
    if suma_real == 0:
        raise ValueError("No se puede calcular WAPE porque la suma de demanda_real es 0.")

    return (real - prediccion).abs().sum() / suma_real * 100


def calcular_metricas(real, prediccion):
    # Calcula metricas operativas y de error para una prediccion.
    real = pd.Series(real).reset_index(drop=True)
    prediccion = pd.Series(prediccion).reset_index(drop=True).clip(lower=0)
    quiebre = (real > prediccion).astype(int)
    sobreestimacion = (prediccion > real).astype(int)
    diferencia_subestimada = real[quiebre == 1] - prediccion[quiebre == 1]

    if diferencia_subestimada.empty:
        subestimacion_media = 0
    else:
        subestimacion_media = diferencia_subestimada.mean()

    return {
        "MAE": mean_absolute_error(real, prediccion),
        "WAPE": calcular_wape(real, prediccion),
        "TQS": quiebre.mean() * 100,
        "quiebres": int(quiebre.sum()),
        "sobreestimacion_pct": sobreestimacion.mean() * 100,
        "subestimacion_media": subestimacion_media,
        "score_negocio": calcular_wape(real, prediccion)
        + PESO_TQS_SCORE * quiebre.mean() * 100,
    }


def preparar_datos():
    # Leer el archivo preparado por Takeshi.
    df = pd.read_csv(RUTA_ENTRADA)

    # Validar columnas necesarias. No se usan los promedios moviles originales.
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

    # Convertir semana a datetime y ordenar cronologicamente.
    df["semana"] = pd.to_datetime(df["semana"], errors="coerce")
    df = df.sort_values("semana").reset_index(drop=True)

    if not df["semana"].is_monotonic_increasing:
        raise ValueError("Las semanas no quedaron ordenadas cronologicamente.")

    # Recalcular promedios moviles correctos usando solo semanas anteriores.
    df["promedio_movil_4_correcto"] = (
        df[VARIABLE_OBJETIVO].shift(1).rolling(window=4).mean()
    )
    df["promedio_movil_8_correcto"] = (
        df[VARIABLE_OBJETIVO].shift(1).rolling(window=8).mean()
    )

    # Eliminar solo filas con nulos en columnas necesarias para el modelo.
    columnas_modelo = ["semana", VARIABLE_OBJETIVO] + VARIABLES_MODELO
    df_modelo = df.dropna(subset=columnas_modelo).copy()

    if df_modelo.empty:
        raise ValueError("No hay filas validas para modelar despues de eliminar nulos.")

    return df_modelo


def dividir_train_test(df_modelo):
    # Mantener la misma division temporal 80/20 usada en 02_random_forest_jesus.py.
    indice_corte = int(len(df_modelo) * 0.8)
    if indice_corte == 0 or indice_corte == len(df_modelo):
        raise ValueError("No hay suficientes filas para separar entrenamiento y prueba.")

    train = df_modelo.iloc[:indice_corte].copy()
    test = df_modelo.iloc[indice_corte:].copy()

    if len(test) != 16:
        raise ValueError(f"El test final debe tener 16 semanas, pero tiene {len(test)}.")

    return train, test


def crear_modelo(parametros):
    # Modelo Random Forest con semilla fija para resultados reproducibles.
    return RandomForestRegressor(
        **parametros,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def promedio_metricas(metricas_folds):
    # Promedia las metricas obtenidas en los folds temporales.
    return {
        clave: float(np.mean([metricas[clave] for metricas in metricas_folds]))
        for clave in metricas_folds[0]
    }


def optimizar_hiperparametros(train):
    # Espacio de busqueda solicitado. Se muestrea una parte para evitar costo excesivo.
    espacio_parametros = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 4, 6, 10],
        "min_samples_leaf": [1, 2, 3, 4, 6],
        "max_features": [1.0, "sqrt", 0.7],
        "criterion": ["squared_error", "absolute_error"],
    }
    configuraciones = list(
        ParameterSampler(
            espacio_parametros,
            n_iter=N_CONFIGURACIONES_BUSQUEDA,
            random_state=RANDOM_STATE,
        )
    )

    x_train = train[VARIABLES_MODELO]
    y_train = train[VARIABLE_OBJETIVO]
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_VALIDACION)
    resultados = []

    for numero, parametros in enumerate(configuraciones, start=1):
        metricas_folds = []

        for indices_train, indices_validacion in tscv.split(x_train):
            x_fold_train = x_train.iloc[indices_train]
            y_fold_train = y_train.iloc[indices_train]
            x_fold_validacion = x_train.iloc[indices_validacion]
            y_fold_validacion = y_train.iloc[indices_validacion]

            modelo = crear_modelo(parametros)
            modelo.fit(x_fold_train, y_fold_train)
            prediccion = modelo.predict(x_fold_validacion)
            prediccion = np.clip(prediccion, 0, None)
            metricas_folds.append(calcular_metricas(y_fold_validacion, prediccion))

        metricas_promedio = promedio_metricas(metricas_folds)
        resultados.append(
            {
                "numero": numero,
                "parametros": parametros,
                **metricas_promedio,
            }
        )

    resultados = sorted(resultados, key=lambda item: item["score_negocio"])
    return resultados[0], resultados


def optimizar_factor_seguridad(train, mejores_parametros):
    # El factor se elige solo con validacion temporal dentro del entrenamiento.
    x_train = train[VARIABLES_MODELO]
    y_train = train[VARIABLE_OBJETIVO]
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_VALIDACION)
    resultados = []

    for factor in FACTORES_SEGURIDAD:
        metricas_folds = []

        for indices_train, indices_validacion in tscv.split(x_train):
            x_fold_train = x_train.iloc[indices_train]
            y_fold_train = y_train.iloc[indices_train]
            x_fold_validacion = x_train.iloc[indices_validacion]
            y_fold_validacion = y_train.iloc[indices_validacion]

            modelo = crear_modelo(mejores_parametros)
            modelo.fit(x_fold_train, y_fold_train)
            prediccion_base = np.clip(modelo.predict(x_fold_validacion), 0, None)
            prediccion_ajustada = np.clip(prediccion_base * factor, 0, None)
            metricas_folds.append(
                calcular_metricas(y_fold_validacion, prediccion_ajustada)
            )

        metricas_promedio = promedio_metricas(metricas_folds)
        resultados.append({"factor": factor, **metricas_promedio})

    resultados = sorted(resultados, key=lambda item: item["score_negocio"])
    return resultados[0], resultados


def entrenar_y_evaluar_final(train, test, mejores_parametros, mejor_factor):
    # Entrenar los modelos finales usando todo el conjunto de entrenamiento.
    x_train = train[VARIABLES_MODELO]
    y_train = train[VARIABLE_OBJETIVO]
    x_test = test[VARIABLES_MODELO]
    y_test = test[VARIABLE_OBJETIVO]

    rf_inicial = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_inicial.fit(x_train, y_train)

    rf_optimizado = crear_modelo(mejores_parametros)
    rf_optimizado.fit(x_train, y_train)

    resultados_test = test[["semana", VARIABLE_OBJETIVO]].copy()
    resultados_test["prediccion_baseline"] = test["lag_1"]
    resultados_test["prediccion_rf_inicial"] = np.clip(
        rf_inicial.predict(x_test),
        0,
        None,
    )
    resultados_test["prediccion_rf_optimizado"] = np.clip(
        rf_optimizado.predict(x_test),
        0,
        None,
    )
    resultados_test["prediccion_rf_ajustada"] = np.clip(
        resultados_test["prediccion_rf_optimizado"] * mejor_factor,
        0,
        None,
    )

    columnas_quiebre = {
        "baseline": "prediccion_baseline",
        "rf_inicial": "prediccion_rf_inicial",
        "rf_optimizado": "prediccion_rf_optimizado",
        "rf_ajustado": "prediccion_rf_ajustada",
    }
    for nombre, columna_prediccion in columnas_quiebre.items():
        resultados_test[f"quiebre_{nombre}"] = (
            resultados_test[VARIABLE_OBJETIVO] > resultados_test[columna_prediccion]
        ).astype(int)

    metricas_test = {
        "Linea Base": calcular_metricas(y_test, resultados_test["prediccion_baseline"]),
        "Random Forest Inicial": calcular_metricas(
            y_test,
            resultados_test["prediccion_rf_inicial"],
        ),
        "Random Forest Optimizado": calcular_metricas(
            y_test,
            resultados_test["prediccion_rf_optimizado"],
        ),
        "Random Forest Optimizado Ajustado": calcular_metricas(
            y_test,
            resultados_test["prediccion_rf_ajustada"],
        ),
    }

    return resultados_test, metricas_test, rf_optimizado


def guardar_archivos(resultados_test, metricas_test):
    # Crear carpeta de resultados y guardar los CSV solicitados.
    CARPETA_RESULTADOS.mkdir(exist_ok=True)

    columnas_predicciones = [
        "semana",
        VARIABLE_OBJETIVO,
        "prediccion_baseline",
        "prediccion_rf_inicial",
        "prediccion_rf_optimizado",
        "prediccion_rf_ajustada",
        "quiebre_baseline",
        "quiebre_rf_inicial",
        "quiebre_rf_optimizado",
        "quiebre_rf_ajustado",
    ]
    resultados_test[columnas_predicciones].to_csv(RUTA_PREDICCIONES, index=False)

    filas_metricas = []
    for modelo, metricas in metricas_test.items():
        filas_metricas.append(
            {
                "modelo": modelo,
                "MAE": metricas["MAE"],
                "WAPE": metricas["WAPE"],
                "TQS": metricas["TQS"],
            }
        )

    pd.DataFrame(filas_metricas).to_csv(RUTA_COMPARACION, index=False)


def imprimir_resultados(
    train,
    test,
    mejor_busqueda,
    mejor_factor_validacion,
    metricas_test,
    modelo_optimizado,
):
    # Mostrar resultados claros en consola para documentar la comparacion.
    metricas_baseline = metricas_test["Linea Base"]
    metricas_ajustado = metricas_test["Random Forest Optimizado Ajustado"]
    reduccion_tqs = metricas_baseline["TQS"] - metricas_ajustado["TQS"]
    cambio_wape = metricas_ajustado["WAPE"] - metricas_baseline["WAPE"]

    importancias = pd.DataFrame(
        {
            "variable": VARIABLES_MODELO,
            "importancia": modelo_optimizado.feature_importances_,
        }
    ).sort_values("importancia", ascending=False)

    print("=============================================")
    print("OPTIMIZACION RANDOM FOREST - JESUS")
    print("=============================================")
    print()
    print(f"Train: {len(train)} semanas")
    print(f"Test final: {len(test)} semanas")
    print(
        "Rango train: "
        f"{train['semana'].min().date()} a {train['semana'].max().date()}"
    )
    print(
        "Rango test: "
        f"{test['semana'].min().date()} a {test['semana'].max().date()}"
    )
    print()
    print("Mejores hiperparametros:")
    for clave, valor in mejor_busqueda["parametros"].items():
        print(f"{clave}: {valor}")
    print()
    print(f"Mejor factor de seguridad: {mejor_factor_validacion['factor']:.2f}")
    print()
    print("Resultado promedio de validacion:")
    print(f"WAPE: {mejor_factor_validacion['WAPE']:.2f}%")
    print(f"TQS: {mejor_factor_validacion['TQS']:.2f}%")
    print(f"Score negocio: {mejor_factor_validacion['score_negocio']:.2f}")
    print(
        "Subestimacion media: "
        f"{mejor_factor_validacion['subestimacion_media']:.2f}"
    )
    print(
        "Sobreestimacion: "
        f"{mejor_factor_validacion['sobreestimacion_pct']:.2f}%"
    )
    print()
    print("TEST FINAL")
    for nombre, metricas in metricas_test.items():
        print()
        print(nombre.upper())
        print(f"MAE: {metricas['MAE']:.2f}")
        print(f"WAPE: {metricas['WAPE']:.2f}%")
        print(f"TQS: {metricas['TQS']:.2f}%")
        print(f"Quiebres: {metricas['quiebres']}")
    print()
    print("Reduccion de TQS respecto a baseline:")
    print(f"{reduccion_tqs:.2f} puntos porcentuales")
    print()
    print("Cambio de WAPE respecto a baseline:")
    print(f"{cambio_wape:.2f} puntos porcentuales")
    print()
    print("5 variables mas importantes:")
    for _, fila in importancias.head(5).iterrows():
        print(f"{fila['variable']}: {fila['importancia']:.4f}")
    print()
    print("Archivos generados:")
    print(RUTA_PREDICCIONES)
    print(RUTA_COMPARACION)
    print("=============================================")


def main():
    if not verificar_dependencias():
        return

    df_modelo = preparar_datos()
    train, test = dividir_train_test(df_modelo)

    mejor_busqueda, _ = optimizar_hiperparametros(train)
    mejor_factor_validacion, _ = optimizar_factor_seguridad(
        train,
        mejor_busqueda["parametros"],
    )

    resultados_test, metricas_test, modelo_optimizado = entrenar_y_evaluar_final(
        train,
        test,
        mejor_busqueda["parametros"],
        mejor_factor_validacion["factor"],
    )
    guardar_archivos(resultados_test, metricas_test)
    imprimir_resultados(
        train,
        test,
        mejor_busqueda,
        mejor_factor_validacion,
        metricas_test,
        modelo_optimizado,
    )


if __name__ == "__main__":
    main()
