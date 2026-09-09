"""Genera el informe exploratorio.

Instalar dependencias: python -m pip install -r requirements-eda.txt
Ejecutar: python analizar_eda.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "resultados" / "eda"


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in rows],
    ])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory, missing = [], []
    for path in sorted(ROOT.glob("*.csv")):
        frame = pd.read_csv(path)
        inventory.append((path.name, len(frame), len(frame.columns)))
        for column, count in frame.isna().sum().items():
            if count:
                missing.append((path.name, column, int(count)))
    pd.DataFrame(inventory, columns=["archivo", "registros", "variables"]).to_csv(
        OUT / "inventario.csv", index=False)
    pd.DataFrame(missing, columns=["archivo", "variable", "faltantes"]).to_csv(
        OUT / "faltantes.csv", index=False)

    df = pd.read_csv(ROOT / "demanda_predictiva_takeshi.csv", parse_dates=["semana"])
    df = df.sort_values("semana").reset_index(drop=True)
    orders = pd.read_csv(ROOT / "olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
    items = pd.read_csv(ROOT / "olist_order_items_dataset.csv")
    joined = items.merge(orders[["order_id", "order_purchase_timestamp", "order_status"]],
                         on="order_id", how="inner", validate="many_to_one")
    weekly = joined.groupby(joined.order_purchase_timestamp.dt.to_period("W").dt.start_time).size()
    reconstructed = pd.DataFrame({"demanda_real": weekly})
    for k in range(1, 5):
        reconstructed[f"lag_{k}"] = weekly.shift(k)
    for k in (4, 8):
        reconstructed[f"promedio_movil_{k}"] = weekly.rolling(k).mean()
    generated_missing = reconstructed.isna().sum()
    valid = reconstructed.dropna()
    assert len(joined) == len(items)
    assert df.semana.tolist() == valid.index.tolist()
    for column in reconstructed:
        np.testing.assert_allclose(df[column], valid[column])
    assert not df.isna().any().any()
    assert df.semana.diff().dropna().eq(pd.Timedelta(days=7)).all()

    y = df.demanda_real
    q1, q3 = y.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df.loc[~y.between(low, high), ["semana", "demanda_real"]]
    outliers.to_csv(OUT / "atipicos_demanda.csv", index=False)
    correlations = df[["demanda_real", "lag_1", "lag_2", "lag_3", "lag_4"]].corr()
    correlations.to_csv(OUT / "correlaciones.csv")
    r = y.corr(df.lag_1)
    r_diff = y.diff().corr(df.lag_1.diff())
    metrics = pd.read_csv(ROOT / "resultados" / "resumen_backtesting_jesus.csv")
    backtest = pd.read_csv(ROOT / "resultados" / "backtesting_semanal_jesus.csv")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    fig.suptitle("Olist · Exploración de la demanda semanal\n88 semanas | enero de 2017 – septiembre de 2018", fontsize=17)
    ax = axes[0, 0]
    ax.hist(y, bins=np.arange(0, 4001, 400), color="#227c9d", edgecolor="white")
    ax.axvline(y.median(), color="#c05a19", linestyle="--", label=f"Mediana: {y.median():,.0f}")
    ax.set(title="1. Distribución de la demanda", xlabel="Artículos por semana", ylabel="Número de semanas")
    ax.legend()
    ax = axes[0, 1]
    ax.boxplot(y, orientation="horizontal", patch_artist=True,
               boxprops={"facecolor": "#b8dce9"}, medianprops={"color": "#c05a19"})
    ax.axvline(high, color="#b23b3b", linestyle="--", label=f"Límite IQR: {high:,.1f}")
    ax.set(title="2. Un valor atípico superior: 3.497 artículos", xlabel="Artículos por semana", yticks=[])
    ax.legend()
    ax = axes[1, 0]
    ax.scatter(df.lag_1, y, alpha=.75, s=32, color="#227c9d")
    ax.plot([0, 3600], [0, 3600], color="#999999", linestyle="--", label="Demanda igual a la semana anterior")
    ax.set(title=f"3. Demanda actual y anterior · Pearson r = {r:.3f}",
           xlabel="Demanda de la semana anterior (lag_1)", ylabel="Demanda de la semana actual")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(df.semana, y, color="#227c9d", linewidth=1.6)
    ax.scatter(outliers.semana, outliers.demanda_real, color="#b23b3b", zorder=3)
    ax.annotate("20/11/2017: 3.497", (outliers.iloc[0].semana, outliers.iloc[0].demanda_real),
                xytext=(-100, -25), textcoords="offset points", fontsize=9)
    ax.set(title="4. Evolución temporal y caída al final del registro", xlabel="Semana", ylabel="Artículos por semana")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(OUT / "graficos_eda.png", dpi=180)
    fig.savefig(OUT / "graficos_eda.pdf")
    plt.close(fig)

    types = [
        ("Fecha", "semana", "1; se convierte desde texto al leer el CSV"),
        ("Enteros", "demanda_real, mes, semana_del_anio, trimestre, quiebre_stock", "5; quiebre_stock es binaria"),
        ("Decimales", "lag_1, lag_2, lag_3, lag_4, promedio_movil_4, promedio_movil_8, demanda_predicha", "7; los rezagos representan conteos"),
    ]
    inventory_table = table(["Archivo", "Registros", "Variables"], inventory)
    missing_table = table(["Archivo", "Variable", "Faltantes"], missing)
    type_table = table(["Tipo principal", "Variables", "Cantidad y observación"], types)
    corr_table = table(["Relación con demanda_real", "Pearson r"],
                       [(c, f"{correlations.loc['demanda_real', c]:.3f}") for c in correlations.columns[1:]])
    metric_table = table(["Modelo", "WAPE", "Semanas subestimadas", "Tasa de subestimación"],
                         [(row.modelo, f"{row.WAPE:.2f}%", int(row.quiebres), f"{row.TQS:.2f}%")
                          for row in metrics.itertuples()])
    report = f"""# Análisis exploratorio del proyecto de predicción semanal de demanda

Las cifras se calcularon directamente con los CSV locales. Se responde sobre el conjunto preparado para el modelo y se distingue de las tablas originales.

## 1. ¿Qué datos tenemos?

La fuente del proyecto son las tablas de comercio electrónico **Olist** presentes en la carpeta. `tarea_takeshi.py` une `olist_orders_dataset.csv` (99.441 pedidos, 8 variables) con `olist_order_items_dataset.csv` (112.650 artículos, 7 variables) mediante `order_id`. Cuenta las filas de artículos por semana y construye `demanda_predictiva_takeshi.csv`.

El conjunto preparado contiene **{len(df)} registros semanales y {len(df.columns)} variables**, desde el **{df.semana.min():%d/%m/%Y} hasta el {df.semana.max():%d/%m/%Y}**. Suma **{int(y.sum()):,} artículos registrados**. La variable objetivo es `demanda_real`.

{type_table}

**Alcance:** se agregan artículos de {items.seller_id.nunique():,} vendedores de toda la base; no se filtra una sola tienda. Son artículos registrados en pedidos, una aproximación a la demanda, y no una medición de toda la demanda potencial ni de ventas perdidas. El proceso tampoco filtra por estado: incluye {int((joined.order_status == 'canceled').sum())} filas de artículos de pedidos cancelados.

## 2. ¿Hay datos faltantes?

**El archivo preparado tiene cero valores faltantes en sus 13 variables.** Las tablas originales sí contienen faltantes:

{missing_table}

Las demás tablas no tienen celdas nulas detectadas por `pandas.isna()`. Los faltantes de pedidos están en fechas de aprobación o entrega; el proceso usa `order_purchase_timestamp`, que está completo. Las tablas de reseñas y productos no intervienen en este cálculo de demanda. Por eso no se imputaron esas columnas: no son necesarias para esta transformación. No se comprobó aquí la existencia de códigos especiales que pudieran representar faltantes.

Al crear los rezagos aparecen {int(generated_missing.lag_1)}, {int(generated_missing.lag_2)}, {int(generated_missing.lag_3)} y {int(generated_missing.lag_4)} nulos en `lag_1` a `lag_4`. Los promedios móviles de 4 y 8 observaciones generan {int(generated_missing.promedio_movil_4)} y {int(generated_missing.promedio_movil_8)} nulos. `demanda_predicha`, copia del promedio de 4 observaciones, también tiene 3 nulos antes de limpiar. Se aplica `dropna()`: se eliminan **7 filas iniciales**, pasando de **{len(weekly)} a {len(df)} semanas**. Son filas que todavía no tienen suficiente historia, por lo que no se inventan valores para completarlas.

Los scripts posteriores recalculan promedios con `shift(1).rolling(...)` para usar solo el pasado; esto descarta otras 8 filas iniciales del archivo preparado y deja 80 semanas para modelar. Además, en la agregación original hay 11 semanas del calendario sin filas antes de 2017: el script las omite, por lo que los primeros rezagos pueden referirse a la observación anterior y no a la semana calendario anterior. Las 88 semanas del CSV final sí son consecutivas. Es necesario revisar el período inicial antes de interpretar esos primeros rezagos.

## 3. ¿Existen valores atípicos?

Se aplicó la **regla de 1,5 veces el rango intercuartílico (IQR)** a `demanda_real`, sobre las 88 semanas, y se representó con un diagrama de caja:

- Primer cuartil: **{q1:.2f}**; tercer cuartil: **{q3:.2f}**.
- IQR = Q3 − Q1 = **{iqr:.2f}**.
- Límites: **{low:.2f}** y **{high:.2f} artículos**.
- Se detecta **{len(outliers)} semana atípica**: **20/11/2017, con 3.497 artículos**.

El pico equivale a **{3497 / y.median():.2f} veces la mediana**. Es un valor observado que requiere revisión; el criterio estadístico no demuestra que sea un error ni explica su causa. Se conserva, porque un pico real puede ser relevante para abastecimiento. También deben revisarse las últimas semanas (132 y 1 artículos): no son atípicas según este IQR, pero podrían reflejar cobertura incompleta del final del conjunto. No debe concluirse automáticamente que el negocio dejó de vender.

## 4. ¿Cómo se distribuyen los datos?

El histograma muestra una demanda heterogénea: **mínimo {y.min()}, máximo {y.max()}, media {y.mean():.2f} y mediana {y.median():.0f} artículos por semana**. El 50% central se sitúa entre **{q1:.2f} y {q3:.2f} artículos**. La asimetría muestral es **{y.skew():.2f}**, ligeramente positiva, con una cola hacia valores altos y un pico extremo.

El histograma reúne períodos diferentes: la serie temporal muestra un crecimiento general durante buena parte del período y una caída al final. Por tanto, esta distribución no debe interpretarse como una demanda estable en el tiempo ni permite afirmar por sí sola una distribución normal.

## 5. ¿Qué relaciones existen entre variables?

El gráfico de dispersión relaciona la demanda de la semana anterior (`lag_1`) con la demanda actual:

{corr_table}

Se observa una **relación positiva fuerte con `lag_1` (r = {r:.3f})**: las semanas de mayor demanda suelen seguir a semanas de demanda alta. Esto respalda el uso de la historia reciente como referencia inicial.

**Limitaciones:** la correlación no implica causalidad ni garantiza buen pronóstico. Son observaciones temporales dependientes y parte de la relación puede deberse a la tendencia compartida. Al correlacionar los cambios semanales de ambas series, el coeficiente baja a **{r_diff:.3f}**; la asociación en niveles no se reproduce en los cambios. Se necesitan evaluaciones que respeten el orden temporal.

Además, los `promedio_movil_4`, `promedio_movil_8` y `demanda_predicha` originales incluyen la demanda de la semana actual. Usarlos para pronosticar esa misma semana produciría fuga de información. Los scripts `02` a `05` corrigen los promedios usando `shift(1)`; por eso aquí se destaca la relación con `lag_1`.

## 6. Hallazgo interesante

**La reducción de subestimaciones en los resultados guardados aparece al agregar un margen de seguridad del 8%, con un aumento del error de pronóstico.** En las {len(backtest)} semanas del backtesting existente:

{metric_table}

El Random Forest sin ajuste tiene las mismas 18 semanas subestimadas que la línea base. Con el factor 1,08, pasan a 10: son **8 semanas menos**, una reducción relativa del **44,44%**, y la tasa baja **20 puntos porcentuales**. El WAPE, que mide el error absoluto total respecto al volumen real, sube aproximadamente **7 puntos porcentuales** frente a la línea base.

Para el negocio, esto permite discutir cuánto inventario adicional conviene mantener frente al riesgo de quedarse corto. El modelo ajustado reduce subestimaciones, pero no es el más preciso. Para decidir si conviene económicamente faltan costos de almacenamiento, faltantes y márgenes. Estos resultados proceden de los CSV del proyecto, no de un reentrenamiento en este análisis ni de una validación externa nueva.

**Interpretación de la métrica:** el proyecto llama `quiebre_stock` a `demanda_real > predicción`. Es una señal simulada de riesgo de faltante si se abasteciera exactamente esa cantidad; no prueba quiebres reales porque no hay inventario disponible ni ventas perdidas observadas.

## Gráficos

![Histograma, diagrama de caja, dispersión y serie temporal](graficos_eda.png)

[Descargar gráficos en PDF](graficos_eda.pdf).

## Inventario completo de archivos de datos

{inventory_table}

Las tablas tienen unidades de observación diferentes; no se deben sumar sus registros como si fueran una sola muestra independiente.

## Reproducibilidad

Ejecutar `python analizar_eda.py` desde un entorno con pandas, numpy y matplotlib. El análisis verifica que la demanda, los rezagos y los promedios del CSV preparado coincidan con su reconstrucción desde pedidos y artículos. No modifica los datos originales ni los modelos. Los resultados de negocio se leen de `resultados/resumen_backtesting_jesus.csv` y `resultados/backtesting_semanal_jesus.csv`.
"""
    (OUT / "informe_eda.md").write_text(report, encoding="utf-8")
    print(f"Informe generado: {OUT / 'informe_eda.md'}")
    print(f"Verificado: {len(df)} semanas, {len(df.columns)} variables, {len(outliers)} atípico, r={r:.6f}")


if __name__ == "__main__":
    main()
