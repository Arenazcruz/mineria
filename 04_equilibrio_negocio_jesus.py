from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

RUTA_ENTRADA = Path("demanda_predictiva_takeshi.csv")
CARPETA_RESULTADOS = Path("resultados")
RUTA_FACTORES = CARPETA_RESULTADOS / "evaluacion_factores_jesus.csv"
RUTA_MODELO_FINAL = CARPETA_RESULTADOS / "modelo_final_jesus.csv"
RUTA_RESUMEN = CARPETA_RESULTADOS / "resumen_final_jesus.csv"

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

PARAMETROS_RF_OPTIMIZADO = {
    "n_estimators": 300,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "max_depth": 20,
    "criterion": "squared_error",
}
FACTORES_SEGURIDAD = [1.00, 1.02, 1.04, 1.05, 1.06, 1.08, 1.10, 1.12, 1.15, 1.20]
TOLERANCIA_WAPE = 10.0
N_SPLITS_VALIDACION = 4
RANDOM_STATE = 42


def calcular_wape(real, prediccion):
    # WAPE global, validando que no exista division entre cero.
    real = pd.Series(real).reset_index(drop=True)
    prediccion = pd.Series(prediccion).reset_index(drop=True)
    suma_real = real.sum()
    if suma_real == 0:
        raise ValueError("No se puede calcular WAPE porque la suma de demanda_real es 0.")

    return (real - prediccion).abs().sum() / suma_real * 100


def calcular_metricas(real, prediccion):
    # Calcula precision y metricas de negocio para una prediccion.
    real = pd.Series(real).reset_index(drop=True)
    prediccion = pd.Series(prediccion).reset_index(drop=True).clip(lower=0)
    quiebre = (real > prediccion).astype(int)
    sobreestimacion = (prediccion > real).astype(int)
    subestimaciones = real[quiebre == 1] - prediccion[quiebre == 1]

    if subestimaciones.empty:
        subestimacion_media = 0.0
    else:
        subestimacion_media = float(subestimaciones.mean())

    return {
        "MAE": float(mean_absolute_error(real, prediccion)),
        "WAPE": float(calcular_wape(real, prediccion)),
        "TQS": float(quiebre.mean() * 100),
        "quiebres": int(quiebre.sum()),
        "sobreestimacion": float(sobreestimacion.mean() * 100),
        "subestimacion_media": subestimacion_media,
    }


def preparar_datos():
    # Leer el archivo preparado por Takeshi.
    df = pd.read_csv(RUTA_ENTRADA)

    # Validar columnas necesarias. No se usan promedios moviles originales del CSV.
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

    # Convertir fecha y ordenar cronologicamente.
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

    columnas_modelo = ["semana", VARIABLE_OBJETIVO] + VARIABLES_MODELO
    df_modelo = df.dropna(subset=columnas_modelo).copy()

    if df_modelo.empty:
        raise ValueError("No hay filas validas para modelar despues de eliminar nulos.")

    return df_modelo


def dividir_train_test(df_modelo):
    # Mantiene la misma division temporal usada en 03_optimizacion_jesus.py.
    indice_corte = int(len(df_modelo) * 0.8)
    train = df_modelo.iloc[:indice_corte].copy()
    test = df_modelo.iloc[indice_corte:].copy()

    if len(test) != 16:
        raise ValueError(f"El test final debe tener 16 semanas, pero tiene {len(test)}.")

    return train, test


def crear_rf_optimizado():
    # Random Forest optimizado fijo, sin volver a buscar hiperparametros.
    return RandomForestRegressor(
        **PARAMETROS_RF_OPTIMIZADO,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def evaluar_factores_en_validacion(train):
    # Evalua factores solamente con TimeSeriesSplit dentro del entrenamiento.
    x_train = train[VARIABLES_MODELO]
    y_train = train[VARIABLE_OBJETIVO]
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_VALIDACION)

    reales_validacion = []
    predicciones_rf_base = []
    predicciones_baseline = []

    for indices_train, indices_validacion in tscv.split(x_train):
        x_fold_train = x_train.iloc[indices_train]
        y_fold_train = y_train.iloc[indices_train]
        x_fold_validacion = x_train.iloc[indices_validacion]
        y_fold_validacion = y_train.iloc[indices_validacion]

        modelo = crear_rf_optimizado()
        modelo.fit(x_fold_train, y_fold_train)

        reales_validacion.append(y_fold_validacion.reset_index(drop=True))
        predicciones_rf_base.append(
            pd.Series(modelo.predict(x_fold_validacion)).clip(lower=0)
        )
        predicciones_baseline.append(
            train.iloc[indices_validacion]["lag_1"].reset_index(drop=True)
        )

    real_validacion = pd.concat(reales_validacion, ignore_index=True)
    pred_rf_base = pd.concat(predicciones_rf_base, ignore_index=True)
    pred_baseline = pd.concat(predicciones_baseline, ignore_index=True)

    metricas_baseline_validacion = calcular_metricas(real_validacion, pred_baseline)
    limite_wape = metricas_baseline_validacion["WAPE"] + TOLERANCIA_WAPE

    filas = []
    for factor in FACTORES_SEGURIDAD:
        pred_factor = pred_rf_base * factor
        metricas = calcular_metricas(real_validacion, pred_factor)
        cumple_restriccion = metricas["WAPE"] <= limite_wape
        filas.append(
            {
                "factor": factor,
                "MAE": metricas["MAE"],
                "WAPE": metricas["WAPE"],
                "TQS": metricas["TQS"],
                "quiebres": metricas["quiebres"],
                "sobreestimacion": metricas["sobreestimacion"],
                "subestimacion_media": metricas["subestimacion_media"],
                "cumple_restriccion_wape": cumple_restriccion,
            }
        )

    tabla_factores = pd.DataFrame(filas)
    candidatos = tabla_factores[tabla_factores["cumple_restriccion_wape"]].copy()

    if candidatos.empty:
        factor_seleccionado = tabla_factores.sort_values(
            ["WAPE", "TQS", "factor"],
            ascending=[True, True, True],
        ).iloc[0]
        motivo = (
            "Ningun factor cumplio la restriccion de WAPE; "
            "se selecciono el menor WAPE."
        )
    else:
        factor_seleccionado = candidatos.sort_values(
            ["TQS", "WAPE", "factor"],
            ascending=[True, True, True],
        ).iloc[0]
        motivo = (
            "Se selecciono el menor TQS entre los factores que cumplen "
            "la restriccion de WAPE."
        )

    return tabla_factores, factor_seleccionado, metricas_baseline_validacion, motivo


def evaluar_modelos_en_test(train, test, factor_seleccionado):
    # Entrena con todo TRAIN y evalua una sola vez sobre TEST FINAL.
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

    rf_optimizado = crear_rf_optimizado()
    rf_optimizado.fit(x_train, y_train)

    resultados = test[["semana", VARIABLE_OBJETIVO]].copy()
    resultados["prediccion_baseline"] = test["lag_1"]
    resultados["prediccion_rf_inicial"] = np.clip(rf_inicial.predict(x_test), 0, None)
    resultados["prediccion_rf_optimizado"] = np.clip(
        rf_optimizado.predict(x_test),
        0,
        None,
    )
    resultados["prediccion_rf_final"] = np.clip(
        resultados["prediccion_rf_optimizado"] * factor_seleccionado,
        0,
        None,
    )

    resultados["error_baseline"] = (
        resultados[VARIABLE_OBJETIVO] - resultados["prediccion_baseline"]
    ).abs()
    resultados["error_rf_final"] = (
        resultados[VARIABLE_OBJETIVO] - resultados["prediccion_rf_final"]
    ).abs()
    resultados["quiebre_baseline"] = (
        resultados[VARIABLE_OBJETIVO] > resultados["prediccion_baseline"]
    ).astype(int)
    resultados["quiebre_rf_final"] = (
        resultados[VARIABLE_OBJETIVO] > resultados["prediccion_rf_final"]
    ).astype(int)

    metricas_test = {
        "Linea Base": calcular_metricas(y_test, resultados["prediccion_baseline"]),
        "RF Inicial": calcular_metricas(y_test, resultados["prediccion_rf_inicial"]),
        "RF Optimizado": calcular_metricas(
            y_test,
            resultados["prediccion_rf_optimizado"],
        ),
        "Modelo Final": calcular_metricas(y_test, resultados["prediccion_rf_final"]),
    }

    return resultados, metricas_test


def guardar_archivos(tabla_factores, resultados_test, metricas_test):
    # Guarda los tres archivos solicitados en la carpeta resultados.
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    tabla_factores.to_csv(RUTA_FACTORES, index=False)

    columnas_modelo_final = [
        "semana",
        VARIABLE_OBJETIVO,
        "prediccion_baseline",
        "prediccion_rf_inicial",
        "prediccion_rf_optimizado",
        "prediccion_rf_final",
        "error_baseline",
        "error_rf_final",
        "quiebre_baseline",
        "quiebre_rf_final",
    ]
    resultados_test[columnas_modelo_final].to_csv(RUTA_MODELO_FINAL, index=False)

    filas_resumen = []
    for modelo, metricas in metricas_test.items():
        filas_resumen.append(
            {
                "modelo": modelo,
                "MAE": metricas["MAE"],
                "WAPE": metricas["WAPE"],
                "TQS": metricas["TQS"],
                "quiebres": metricas["quiebres"],
            }
        )

    pd.DataFrame(filas_resumen).to_csv(RUTA_RESUMEN, index=False)


def imprimir_tabla_factores(tabla_factores):
    # Tabla ordenada para revisar el equilibrio entre WAPE y TQS.
    tabla = tabla_factores.copy().sort_values("factor")
    tabla["factor"] = tabla["factor"].map("{:.2f}".format)
    tabla["MAE"] = tabla["MAE"].map("{:.2f}".format)
    tabla["WAPE"] = tabla["WAPE"].map("{:.2f}%".format)
    tabla["TQS"] = tabla["TQS"].map("{:.2f}%".format)
    tabla["sobreestimacion"] = tabla["sobreestimacion"].map("{:.2f}%".format)
    tabla["subestimacion_media"] = tabla["subestimacion_media"].map("{:.2f}".format)
    tabla["cumple_restriccion_wape"] = tabla["cumple_restriccion_wape"].map(
        {True: "Si", False: "No"}
    )
    print(
        tabla.to_string(
            index=False,
            columns=[
                "factor",
                "MAE",
                "WAPE",
                "TQS",
                "quiebres",
                "sobreestimacion",
                "subestimacion_media",
                "cumple_restriccion_wape",
            ],
        )
    )


def imprimir_resultados(
    train,
    test,
    tabla_factores,
    factor_seleccionado,
    metricas_baseline_validacion,
    motivo_seleccion,
    metricas_test,
):
    # Consola final con validacion, test y analisis de granularidad del TQS.
    metricas_final = metricas_test["Modelo Final"]
    metricas_baseline = metricas_test["Linea Base"]
    cambio_wape = metricas_final["WAPE"] - metricas_baseline["WAPE"]
    reduccion_tqs = metricas_baseline["TQS"] - metricas_final["TQS"]
    quiebres_evitados = metricas_baseline["quiebres"] - metricas_final["quiebres"]
    puntos_por_quiebre = 100 / len(test)

    print("=============================================")
    print("EQUILIBRIO DE NEGOCIO - JESUS")
    print("=============================================")
    print()
    print(f"Train: {train['semana'].min().date()} a {train['semana'].max().date()}")
    print(f"Test: {test['semana'].min().date()} a {test['semana'].max().date()}")
    print()
    print("BASELINE EN VALIDACION")
    print(f"WAPE: {metricas_baseline_validacion['WAPE']:.2f}%")
    print(f"TQS: {metricas_baseline_validacion['TQS']:.2f}%")
    print(f"Tolerancia WAPE: {TOLERANCIA_WAPE:.2f} puntos porcentuales")
    print()
    print("TABLA DE VALIDACION")
    imprimir_tabla_factores(tabla_factores)
    print()
    print("FACTOR SELECCIONADO:")
    print(f"{factor_seleccionado['factor']:.2f}")
    print(motivo_seleccion)
    print()
    print("VALIDACION:")
    print(f"WAPE: {factor_seleccionado['WAPE']:.2f}%")
    print(f"TQS: {factor_seleccionado['TQS']:.2f}%")
    print(f"Sobreestimacion: {factor_seleccionado['sobreestimacion']:.2f}%")
    print(f"Subestimacion media: {factor_seleccionado['subestimacion_media']:.2f}")
    print()
    print("TEST FINAL:")

    for nombre, metricas in metricas_test.items():
        print()
        print(nombre.upper())
        print(f"MAE: {metricas['MAE']:.2f}")
        print(f"WAPE: {metricas['WAPE']:.2f}%")
        print(f"TQS: {metricas['TQS']:.2f}%")
        print(f"Quiebres: {metricas['quiebres']}")

    print()
    print(f"Cambio WAPE respecto baseline: {cambio_wape:.2f} puntos porcentuales")
    print(f"Reduccion TQS respecto baseline: {reduccion_tqs:.2f} puntos porcentuales")
    print(f"Quiebres evitados respecto baseline: {quiebres_evitados}")
    print()
    print("ANALISIS DEL 5%")
    print(
        f"Cada quiebre representa {puntos_por_quiebre:.2f} "
        "puntos porcentuales de TQS."
    )
    for quiebres in range(5):
        print(f"{quiebres} quiebres = {quiebres * puntos_por_quiebre:.2f}%")
    print("Un TQS exacto de 5% no es representable con 16 observaciones semanales.")
    print()
    print("Archivos generados:")
    print(RUTA_FACTORES)
    print(RUTA_MODELO_FINAL)
    print(RUTA_RESUMEN)
    print("=============================================")


def main():
    df_modelo = preparar_datos()
    train, test = dividir_train_test(df_modelo)
    resultado_validacion = evaluar_factores_en_validacion(train)
    tabla_factores = resultado_validacion[0]
    factor_seleccionado = resultado_validacion[1]
    metricas_baseline_validacion = resultado_validacion[2]
    motivo = resultado_validacion[3]

    resultados_test, metricas_test = evaluar_modelos_en_test(
        train,
        test,
        factor_seleccionado["factor"],
    )
    guardar_archivos(tabla_factores, resultados_test, metricas_test)
    imprimir_resultados(
        train,
        test,
        tabla_factores,
        factor_seleccionado,
        metricas_baseline_validacion,
        motivo,
        metricas_test,
    )


if __name__ == "__main__":
    main()
