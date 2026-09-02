import pandas as pd

# ==========================================
# 1. LECTURA DE LOS DATOS
# ==========================================

orders = pd.read_csv("olist_orders_dataset.csv")
items = pd.read_csv("olist_order_items_dataset.csv")

print("Datos cargados correctamente")
print("Pedidos:", orders.shape)
print("Productos vendidos:", items.shape)


# ==========================================
# 2. PREPARAR LAS FECHAS
# ==========================================

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"]
)

# Unimos pedidos con los productos vendidos
df = items.merge(
    orders[["order_id", "order_purchase_timestamp"]],
    on="order_id",
    how="inner"
)


# ==========================================
# 3. CREAR DEMANDA SEMANAL
# ==========================================

df["semana"] = (
    df["order_purchase_timestamp"]
    .dt.to_period("W")
    .apply(lambda x: x.start_time)
)

# Contamos productos vendidos por semana
demanda_semanal = (
    df.groupby("semana")
    .size()
    .reset_index(name="demanda_real")
)

# Ordenamos por fecha
demanda_semanal = demanda_semanal.sort_values("semana")


# ==========================================
# 4. VARIABLES DE REZAGO (LAGS)
# ==========================================

demanda_semanal["lag_1"] = demanda_semanal["demanda_real"].shift(1)
demanda_semanal["lag_2"] = demanda_semanal["demanda_real"].shift(2)
demanda_semanal["lag_3"] = demanda_semanal["demanda_real"].shift(3)
demanda_semanal["lag_4"] = demanda_semanal["demanda_real"].shift(4)


# ==========================================
# 5. PROMEDIOS MOVILES
# ==========================================

demanda_semanal["promedio_movil_4"] = (
    demanda_semanal["demanda_real"]
    .rolling(window=4)
    .mean()
)

demanda_semanal["promedio_movil_8"] = (
    demanda_semanal["demanda_real"]
    .rolling(window=8)
    .mean()
)


# ==========================================
# 6. VARIABLES DE TEMPORALIDAD
# ==========================================

demanda_semanal["mes"] = demanda_semanal["semana"].dt.month

demanda_semanal["semana_del_anio"] = (
    demanda_semanal["semana"]
    .dt.isocalendar()
    .week
)

demanda_semanal["trimestre"] = demanda_semanal["semana"].dt.quarter


# ==========================================
# 7. DEMANDA PREDICHA
# ==========================================

# Usamos el promedio movil de 4 semanas
# como una prediccion inicial de la demanda
demanda_semanal["demanda_predicha"] = (
    demanda_semanal["promedio_movil_4"]
)


# ==========================================
# 8. CALCULAR QUIEBRE DE STOCK
# ==========================================

# Quiebre = 1 si demanda real > demanda predicha
# Quiebre = 0 en caso contrario

demanda_semanal["quiebre_stock"] = (
    demanda_semanal["demanda_real"]
    > demanda_semanal["demanda_predicha"]
).astype(int)


# ==========================================
# 9. ELIMINAR FILAS SIN DATOS
# ==========================================

demanda_final = demanda_semanal.dropna()


# ==========================================
# 10. GUARDAR RESULTADO
# ==========================================

demanda_final.to_csv(
    "demanda_predictiva_takeshi.csv",
    index=False
)

print("\nPROCESO TERMINADO CORRECTAMENTE")
print("\nPrimeras filas del resultado:")
print(demanda_final.head())

print("\nColumnas creadas:")
print(demanda_final.columns.tolist())

print("\nPorcentaje de quiebre de stock:")
print(
    demanda_final["quiebre_stock"].mean() * 100
)