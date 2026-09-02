from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


RUTA_ENTRADA = Path("demanda_predictiva_takeshi.csv")
CARPETA_RESULTADOS = Path("resultados")
RUTA_BACKTESTING = CARPETA_RESULTADOS / "backtesting_semanal_jesus.csv"
RUTA_RESUMEN = CARPETA_RESULTADOS / "resumen_backtesting_jesus.csv"

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

PARAMETROS_RF = {
    "n_estimators": 300,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "max_depth": 20,
    "criterion": "squared_error",
    "random_state": 42,
    "n_jobs": -1,
}
FACTOR_SEGURIDAD = 1.08
MIN_SEMANAS_ENTRENAMIENTO = 40


def calcular_wape(real, prediccion):
    # WAPE global: suma de errores absolutos dividida entre suma de demanda real.
    real = pd.Series(real).reset_index(drop=True)
    prediccion = pd.Series(prediccion).reset_index(drop=True)
    suma_real = real.sum()

    if suma_real == 0:
        raise ValueError("No se puede calcular WAPE porque la suma de demanda_real es 0.")

    return (real - prediccion).abs().sum() / suma_real * 100


def calcular_metricas(real, prediccion):
    # Calcula metricas globales para una serie completa de predicciones.
    real = pd.Series(real).reset_index(drop=True)
    prediccion = pd.Series(prediccion).reset_index(drop=True).clip(lower=0)
    quiebre = (real > prediccion).astype(int)
    sobreestimacion = (prediccion > real).astype(int)

    return {
        "MAE": float(mean_absolute_error(real, prediccion)),
        "WAPE": float(calcular_wape(real, prediccion)),
        "TQS": float(quiebre.mean() * 100),
        "quiebres": int(quiebre.sum()),
        "sobreestimacion": float(sobreestimacion.mean() * 100),
    }


def preparar_datos():
    # Leer el archivo preparado por Takeshi.
    df = pd.read_csv(RUTA_ENTRADA)

    # Validar columnas requeridas. No se usan los promedios moviles originales.
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

    # Eliminar solo filas con nulos en las columnas necesarias para el modelo.
    columnas_modelo = ["semana", VARIABLE_OBJETIVO] + VARIABLES_MODELO
    df_modelo = df.dropna(subset=columnas_modelo).copy()

    if len(df_modelo) <= MIN_SEMANAS_ENTRENAMIENTO:
        raise ValueError(
            "No hay suficientes filas para backtesting con "
            f"{MIN_SEMANAS_ENTRENAMIENTO} semanas iniciales."
        )

    return df_modelo


def crear_modelo():
    # Random Forest congelado: no se optimizan hiperparametros en esta etapa.
    return RandomForestRegressor(**PARAMETROS_RF)


def ejecutar_backtesting(df_modelo):
    # Backtesting expanding window: se entrena con pasado y se predice la semana siguiente.
    resultados = []

    for indice_prediccion in range(MIN_SEMANAS_ENTRENAMIENTO, len(df_modelo)):
        datos_entrenamiento = df_modelo.iloc[:indice_prediccion].copy()
        semana_a_predecir = df_modelo.iloc[indice_prediccion].copy()

        x_train = datos_entrenamiento[VARIABLES_MODELO]
        y_train = datos_entrenamiento[VARIABLE_OBJETIVO]
        x_prediccion = semana_a_predecir[VARIABLES_MODELO].to_frame().T

        modelo = crear_modelo()
        modelo.fit(x_train, y_train)

        prediccion_rf = float(modelo.predict(x_prediccion)[0])
        prediccion_rf = max(prediccion_rf, 0)
        prediccion_rf_ajustado = prediccion_rf * FACTOR_SEGURIDAD

        demanda_real = float(semana_a_predecir[VARIABLE_OBJETIVO])
        prediccion_baseline = float(semana_a_predecir["lag_1"])

        resultados.append(
            {
                "semana": semana_a_predecir["semana"],
                "demanda_real": demanda_real,
                "prediccion_baseline": prediccion_baseline,
                "prediccion_rf": prediccion_rf,
                "prediccion_rf_ajustado": prediccion_rf_ajustado,
                "error_baseline": abs(demanda_real - prediccion_baseline),
                "error_rf": abs(demanda_real - prediccion_rf),
                "error_rf_ajustado": abs(demanda_real - prediccion_rf_ajustado),
                "quiebre_baseline": int(demanda_real > prediccion_baseline),
                "quiebre_rf": int(demanda_real > prediccion_rf),
                "quiebre_rf_ajustado": int(demanda_real > prediccion_rf_ajustado),
            }
        )

    return pd.DataFrame(resultados)


def crear_resumen_metricas(backtesting):
    # Calcula metricas globales para cada metodo evaluado en backtesting.
    real = backtesting["demanda_real"]
    modelos = {
        "Linea Base": backtesting["prediccion_baseline"],
        "Random Forest": backtesting["prediccion_rf"],
        "Random Forest Ajustado 1.08": backtesting["prediccion_rf_ajustado"],
    }

    filas = []
    metricas_por_modelo = {}

    for nombre, prediccion in modelos.items():
        metricas = calcular_metricas(real, prediccion)
        metricas_por_modelo[nombre] = metricas
        filas.append(
            {
                "modelo": nombre,
                "MAE": metricas["MAE"],
                "WAPE": metricas["WAPE"],
                "TQS": metricas["TQS"],
                "quiebres": metricas["quiebres"],
                "sobreestimacion": metricas["sobreestimacion"],
            }
        )

    return pd.DataFrame(filas), metricas_por_modelo


def crear_resumen_por_anio(backtesting):
    # Analisis descriptivo por anio; no se usa para ajustar el modelo.
    resultados_anio = {}
    backtesting = backtesting.copy()
    backtesting["anio"] = pd.to_datetime(backtesting["semana"]).dt.year

    for anio, datos_anio in backtesting.groupby("anio"):
        if len(datos_anio) < 2:
            continue

        resumen_anio, _ = crear_resumen_metricas(datos_anio)
        resultados_anio[int(anio)] = resumen_anio

    return resultados_anio


def guardar_archivos(backtesting, resumen):
    # Guardar los CSV solicitados.
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    backtesting.to_csv(RUTA_BACKTESTING, index=False)
    resumen.to_csv(RUTA_RESUMEN, index=False)


def imprimir_metricas_modelo(nombre, metricas):
    print()
    print(nombre)
    print(f"MAE: {metricas['MAE']:.2f}")
    print(f"WAPE: {metricas['WAPE']:.2f}%")
    print(f"TQS: {metricas['TQS']:.2f}%")
    print(f"Quiebres: {metricas['quiebres']}")


def imprimir_tabla_comparacion(resumen):
    # Tabla compacta para comparar modelos en backtesting.
    tabla = resumen.copy()
    tabla["MAE"] = tabla["MAE"].map("{:.2f}".format)
    tabla["WAPE"] = tabla["WAPE"].map("{:.2f}%".format)
    tabla["TQS"] = tabla["TQS"].map("{:.2f}%".format)
    tabla["sobreestimacion"] = tabla["sobreestimacion"].map("{:.2f}%".format)
    print(
        tabla.to_string(
            index=False,
            columns=["modelo", "MAE", "WAPE", "TQS", "quiebres", "sobreestimacion"],
        )
    )


def imprimir_resultados(backtesting, resumen, metricas_por_modelo, resumen_por_anio):
    # Mostrar resultados globales, comparaciones y analisis por anio.
    metricas_baseline = metricas_por_modelo["Linea Base"]
    metricas_rf_ajustado = metricas_por_modelo["Random Forest Ajustado 1.08"]
    reduccion_tqs = metricas_baseline["TQS"] - metricas_rf_ajustado["TQS"]
    quiebres_evitados = (
        metricas_baseline["quiebres"] - metricas_rf_ajustado["quiebres"]
    )
    cambio_wape = metricas_rf_ajustado["WAPE"] - metricas_baseline["WAPE"]

    print("=============================================")
    print("BACKTESTING TEMPORAL - JESUS")
    print("=============================================")
    print()
    print(f"Semanas totales evaluadas: {len(backtesting)}")
    print()
    print("TABLA COMPARATIVA")
    imprimir_tabla_comparacion(resumen)

    imprimir_metricas_modelo("LINEA BASE", metricas_por_modelo["Linea Base"])
    imprimir_metricas_modelo("RANDOM FOREST", metricas_por_modelo["Random Forest"])
    imprimir_metricas_modelo(
        "RF AJUSTADO 1.08",
        metricas_por_modelo["Random Forest Ajustado 1.08"],
    )

    print()
    print(f"Reduccion de TQS: {reduccion_tqs:.2f} puntos porcentuales")
    print(f"Quiebres evitados: {quiebres_evitados}")
    print(f"Cambio de WAPE: {cambio_wape:.2f} puntos porcentuales")

    print()
    print("RESULTADOS POR ANIO")
    if not resumen_por_anio:
        print("No hay suficientes observaciones para calcular metricas por anio.")
    else:
        for anio, resumen_anio in resumen_por_anio.items():
            print()
            print(f"ANIO {anio}")
            imprimir_tabla_comparacion(resumen_anio)

    print()
    print("Archivos generados:")
    print(RUTA_BACKTESTING)
    print(RUTA_RESUMEN)
    print("=============================================")


def main():
    df_modelo = preparar_datos()
    backtesting = ejecutar_backtesting(df_modelo)
    resumen, metricas_por_modelo = crear_resumen_metricas(backtesting)
    resumen_por_anio = crear_resumen_por_anio(backtesting)

    guardar_archivos(backtesting, resumen)
    imprimir_resultados(backtesting, resumen, metricas_por_modelo, resumen_por_anio)


if __name__ == "__main__":
    main()
