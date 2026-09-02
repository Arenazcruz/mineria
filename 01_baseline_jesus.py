from pathlib import Path

import pandas as pd


RUTA_ENTRADA = Path("demanda_predictiva_takeshi.csv")
CARPETA_RESULTADOS = Path("resultados")
RUTA_SALIDA = CARPETA_RESULTADOS / "baseline_jesus.csv"


def main():
    # 1. Leer el archivo preparado por Takeshi.
    df = pd.read_csv(RUTA_ENTRADA)

    # 2. Validar que existan las columnas necesarias para esta linea base.
    columnas_necesarias = ["semana", "demanda_real", "lag_1"]
    columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas necesarias: {columnas_faltantes}")

    # 3. Convertir la columna semana a fecha y ordenar cronologicamente.
    df["semana"] = pd.to_datetime(df["semana"], errors="coerce")
    df = df.sort_values("semana").reset_index(drop=True)

    # 4. Validar que las semanas esten ordenadas cronologicamente.
    if not df["semana"].is_monotonic_increasing:
        raise ValueError("Las semanas no quedaron ordenadas cronologicamente.")

    # 5. Crear la prediccion baseline usando lag_1.
    #    No se usa la columna demanda_predicha de Takeshi.
    df["prediccion_baseline"] = df["lag_1"]

    # 6. Evaluar solo filas sin valores nulos en las columnas necesarias.
    columnas_evaluacion = ["semana", "demanda_real", "lag_1", "prediccion_baseline"]
    df_eval = df.dropna(subset=columnas_evaluacion).copy()

    if df_eval.empty:
        raise ValueError("No hay filas validas para evaluar despues de eliminar nulos.")

    # 7. Calcular error absoluto.
    df_eval["error_absoluto"] = (
        df_eval["demanda_real"] - df_eval["prediccion_baseline"]
    ).abs()

    # 8. Validar que no haya division entre cero en el calculo de WAPE.
    suma_demanda_real = df_eval["demanda_real"].sum()
    if suma_demanda_real == 0:
        raise ValueError("No se puede calcular WAPE porque la suma de demanda_real es 0.")

    # 9. Calcular WAPE global.
    wape = df_eval["error_absoluto"].sum() / suma_demanda_real * 100

    # 10. Calcular quiebre de stock: ocurre cuando la demanda real supera la prediccion.
    df_eval["quiebre_baseline"] = (
        df_eval["demanda_real"] > df_eval["prediccion_baseline"]
    ).astype(int)

    # 11. Calcular TQS como el porcentaje de semanas con quiebre.
    tqs = df_eval["quiebre_baseline"].mean() * 100
    semanas_evaluadas = len(df_eval)

    # 12. Crear carpeta de resultados si no existe.
    CARPETA_RESULTADOS.mkdir(exist_ok=True)

    # 13. Guardar resultados minimos solicitados.
    columnas_salida = [
        "semana",
        "demanda_real",
        "lag_1",
        "prediccion_baseline",
        "error_absoluto",
        "quiebre_baseline",
    ]
    df_eval[columnas_salida].to_csv(RUTA_SALIDA, index=False)

    # 14. Mostrar resumen claro en consola.
    print("======================================")
    print("LINEA BASE - JESUS")
    print("======================================")
    print(f"Semanas evaluadas: {semanas_evaluadas}")
    print(f"WAPE: {wape:.2f}%")
    print(f"TQS: {tqs:.2f}%")
    print("======================================")
    print(f"CSV generado: {RUTA_SALIDA}")


if __name__ == "__main__":
    main()
