# Predicción Semanal de Demanda - Parte de Jesús

## 1. Objetivo

El objetivo tecnico de esta parte del proyecto es predecir la demanda semanal de una tienda utilizando regresion supervisada. El objetivo de negocio es reducir los quiebres de stock causados por subestimar la demanda, es decir, disminuir los casos en los que la demanda real supera la cantidad prevista.

La prediccion de demanda consiste en estimar cuantas unidades se necesitaran en una semana futura. El quiebre de stock ocurre cuando la demanda real es mayor que la prediccion usada para abastecerse. La precision predictiva mide que tan cerca esta la prediccion del valor real. La politica de abastecimiento decide como usar esa prediccion, por ejemplo agregando un margen de seguridad para reducir subestimaciones.

## 2. Flujo general

```text
demanda_predictiva_takeshi.csv
        |
        v
01 Baseline
        |
        v
02 Random Forest inicial
        |
        v
03 Optimizacion
        |
        v
04 Equilibrio de negocio
        |
        v
05 Backtesting temporal
        |
        v
Resultados finales
```

El flujo empieza con el archivo `demanda_predictiva_takeshi.csv`, que contiene la demanda semanal ya preparada. Primero se construye una linea base simple con `lag_1`. Luego se entrena un primer `RandomForestRegressor`. Despues se optimizan hiperparametros y se analiza un factor de seguridad. En la etapa de equilibrio se selecciona un factor que intenta reducir quiebres sin aumentar excesivamente el WAPE. Finalmente, el backtesting temporal evalua el comportamiento historico del modelo de forma mas robusta.

## 3. Archivo de entrada

El trabajo de esta parte empieza desde:

```text
demanda_predictiva_takeshi.csv
```

Las columnas utilizadas por los scripts son:

- `semana`: fecha de la semana evaluada.
- `demanda_real`: demanda observada en esa semana. Es la variable objetivo.
- `lag_1`: demanda de la semana anterior.
- `lag_2`: demanda de hace dos semanas.
- `lag_3`: demanda de hace tres semanas.
- `lag_4`: demanda de hace cuatro semanas.
- `mes`: mes asociado a la semana.
- `semana_del_anio`: numero de semana dentro del anio.
- `trimestre`: trimestre del anio.
- `promedio_movil_4_correcto`: promedio de las 4 semanas anteriores.
- `promedio_movil_8_correcto`: promedio de las 8 semanas anteriores.

Un `lag` es una variable que representa informacion pasada. Por ejemplo, `lag_1` es la demanda de la semana inmediatamente anterior y `lag_2` es la demanda de hace dos semanas.

Los scripts `02`, `03`, `04` y `05` recalculan los promedios moviles correctos usando:

```python
df["demanda_real"].shift(1).rolling(window=4).mean()
df["demanda_real"].shift(1).rolling(window=8).mean()
```

El uso de `shift(1)` es importante porque obliga a que el promedio movil use solo semanas anteriores. Asi se evita fuga de informacion, ya que no se incorpora la demanda de la semana que se esta intentando predecir.

## 4. 01_baseline_jesus.py

Este script construye una linea base. Una linea base es un modelo simple que sirve como punto de comparacion para saber si un modelo mas complejo realmente aporta valor.

Pasos principales:

- Lee `demanda_predictiva_takeshi.csv`.
- Convierte `semana` a formato de fecha.
- Ordena los datos cronologicamente.
- Define `prediccion_baseline = lag_1`.
- Calcula `error_absoluto = abs(demanda_real - prediccion_baseline)`.
- Calcula WAPE global.
- Define `quiebre_baseline = 1` si `demanda_real > prediccion_baseline`.
- Calcula TQS como el promedio de `quiebre_baseline` multiplicado por 100.
- Guarda `resultados/baseline_jesus.csv`.

Resultados actuales:

```text
Semanas evaluadas: 88
WAPE: 16.40%
TQS: 56.82%
```

Estas metricas corresponden a una evaluacion general del baseline sobre las 88 semanas disponibles en ese archivo. No deben compararse directamente como resultado principal contra metricas obtenidas en otros subconjuntos temporales, porque no todas las etapas usan exactamente las mismas semanas.

## 5. 02_random_forest_jesus.py

Este script entrena un primer modelo `RandomForestRegressor`. Se usa Random Forest porque permite modelar relaciones no lineales entre variables historicas de demanda y demanda futura, y porque puede combinar distintas variables predictivas sin exigir una forma funcional simple.

La variable objetivo es:

```text
demanda_real
```

Las variables predictivas son:

```text
lag_1
lag_2
lag_3
lag_4
promedio_movil_4_correcto
promedio_movil_8_correcto
mes
semana_del_anio
trimestre
```

La division es temporal, no aleatoria. No se usa `shuffle`, porque en series temporales el modelo debe entrenarse con el pasado y evaluarse en semanas posteriores.

Division utilizada:

```text
Train: 64 semanas, 2017-02-27 a 2018-05-14
Test: 16 semanas, 2018-05-21 a 2018-09-03
```

Metricas en test:

```text
LINEA BASE EN TEST
MAE: 361.94
WAPE: 24.99%
TQS: 37.50%

RANDOM FOREST INICIAL
MAE: 464.15
WAPE: 32.05%
TQS: 25.00%
```

Interpretacion: el Random Forest inicial redujo el TQS frente a la linea base en el conjunto de test, pero empeoro MAE y WAPE. Esto significa que redujo algunos quiebres de stock, aunque sus predicciones fueron menos precisas en promedio.

Las variables mas importantes reportadas en esta etapa fueron:

- `lag_1`
- `promedio_movil_4_correcto`
- `promedio_movil_8_correcto`
- `lag_3`
- `semana_del_anio`

## 6. 03_optimizacion_jesus.py

Este script optimiza hiperparametros del Random Forest usando validacion temporal. El test final permanece separado y no se utiliza para elegir configuraciones.

La validacion se realiza con `TimeSeriesSplit`, lo que respeta el orden temporal. En cada division, el modelo se entrena con semanas pasadas y se valida con semanas posteriores. Esto evita usar informacion futura durante la seleccion del modelo.

La busqueda evalua combinaciones de hiperparametros como `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` y `criterion`. Ademas, calcula metricas de negocio como WAPE, TQS, sobreestimacion y subestimacion media.

Mejores hiperparametros encontrados:

```text
n_estimators: 300
min_samples_split: 2
min_samples_leaf: 1
max_features: 1.0
max_depth: 20
criterion: squared_error
```

Tambien se prueba un factor de seguridad:

```text
prediccion_ajustada = prediccion_random_forest * factor_seguridad
```

Inicialmente, en esta etapa se selecciono un factor `1.20`, porque reducia fuertemente los quiebres en validacion segun el score de negocio usado en ese archivo.

Resultados en test:

```text
RF optimizado
MAE: 470.30
WAPE: 32.48%
TQS: 25.00%
Quiebres: 4

RF optimizado ajustado 1.20
MAE: 731.56
WAPE: 50.52%
TQS: 12.50%
Quiebres: 2
```

Interpretacion: el factor `1.20` redujo los quiebres, pero produjo demasiado error y mayor sobreestimacion. Por eso no se tomo inmediatamente como solucion final.

## 7. 04_equilibrio_negocio_jesus.py

Este script busca un equilibrio entre reducir TQS y mantener WAPE bajo control. Ya no intenta minimizar quiebres a cualquier costo.

Factores evaluados:

```text
1.00, 1.02, 1.04, 1.05, 1.06, 1.08, 1.10, 1.12, 1.15, 1.20
```

La regla de negocio usa:

```text
TOLERANCIA_WAPE = 10.0
```

La seleccion funciona asi:

1. Primero se identifican los factores cuyo WAPE no supere el WAPE baseline de validacion por mas de 10 puntos porcentuales.
2. Entre los factores que cumplen, se elige el menor TQS.
3. Si hay empate, se elige el menor WAPE.
4. Si el empate continua, se elige el menor factor de seguridad.

Factor seleccionado:

```text
FACTOR_SEGURIDAD = 1.08
```

Resultados de validacion:

```text
WAPE: 23.72%
TQS: 45.83%
Sobreestimacion: 54.17%
```

Resultados en test del modelo final:

```text
MAE: 557.95
WAPE: 38.53%
TQS: 12.50%
Quiebres: 2
```

Comparacion contra baseline en test:

| Modelo | WAPE | TQS | Quiebres |
|---|---:|---:|---:|
| Baseline | 24.99% | 37.50% | 6 |
| Modelo final | 38.53% | 12.50% | 2 |

El modelo final reduce el TQS en 25 puntos porcentuales y evita 4 quiebres respecto al baseline. A cambio, el WAPE aumenta 13.54 puntos porcentuales. Esto muestra un trade-off entre precision predictiva y disponibilidad de inventario.

## 8. Meta del 5%

El test final contiene 16 semanas. Por lo tanto, cada quiebre equivale a:

```text
100 / 16 = 6.25 puntos porcentuales de TQS
```

Equivalencia:

```text
0 quiebres = 0.00%
1 quiebre = 6.25%
2 quiebres = 12.50%
3 quiebres = 18.75%
4 quiebres = 25.00%
```

Un TQS exactamente igual a 5% no puede representarse con una muestra de 16 semanas. Por esta razon no se afirma que se haya alcanzado una meta exacta de 5%.

## 9. 05_backtesting_jesus.py

Este script realiza la validacion final mediante backtesting temporal tipo walk-forward con expanding window.

Walk-forward validation significa que el modelo avanza en el tiempo simulando como se habria usado historicamente. Expanding window significa que la ventana de entrenamiento crece: se empieza con un minimo de semanas, se predice la siguiente, luego esa semana pasa a formar parte del historial para predecir la proxima.

Ejemplo:

```text
Semanas 1-40 -> predecir semana 41
Semanas 1-41 -> predecir semana 42
Semanas 1-42 -> predecir semana 43
```

Este enfoque es mas robusto que evaluar una sola division train/test, porque prueba el modelo en muchas fechas historicas distintas y respeta el orden temporal. En cada iteracion se entrena solo con pasado y se predice una semana futura.

Parametros congelados:

```text
n_estimators: 300
min_samples_split: 2
min_samples_leaf: 1
max_features: 1.0
max_depth: 20
criterion: squared_error
random_state: 42
n_jobs: -1
FACTOR_SEGURIDAD: 1.08
MIN_SEMANAS_ENTRENAMIENTO: 40
```

Resultados finales de backtesting:

```text
Semanas evaluadas: 40

LINEA BASE
MAE: 265.62
WAPE: 15.88%
TQS: 45.00%
Quiebres: 18

RANDOM FOREST
MAE: 334.57
WAPE: 20.00%
TQS: 45.00%
Quiebres: 18

RANDOM FOREST AJUSTADO 1.08
MAE: 382.66
WAPE: 22.87%
TQS: 25.00%
Quiebres: 10
```

Interpretacion: en backtesting, el Random Forest sin ajuste no redujo los quiebres respecto al baseline, ya que ambos tuvieron TQS de 45% y 18 quiebres. La reduccion aparece al combinar el modelo predictivo con el factor de seguridad `1.08`.

El TQS baja de 45% a 25%. Los quiebres pasan de 18 a 10, por lo que se evitan 8 quiebres. El WAPE aumenta de 15.88% a 22.87%, es decir, sube 7 puntos porcentuales.

## 10. Resultado principal

Los resultados principales deben tomarse del backtesting temporal, porque es la validacion mas robusta de esta parte del proyecto.

| Modelo | MAE | WAPE | TQS | Quiebres |
|---|---:|---:|---:|---:|
| Linea Base | 265.62 | 15.88% | 45.00% | 18 |
| Random Forest | 334.57 | 20.00% | 45.00% | 18 |
| Random Forest Ajustado 1.08 | 382.66 | 22.87% | 25.00% | 10 |

## 11. Interpretacion de negocio

El modelo ajustado sacrifica cierta precision para disminuir el riesgo de quedarse sin inventario. En terminos de negocio, esto puede ser razonable cuando el costo de perder ventas por quiebre de stock es mas alto que el costo de mantener algo mas de inventario.

El Random Forest genera la prediccion de demanda. El factor `1.08` funciona como una politica o margen de seguridad: agrega aproximadamente 8% sobre la prediccion para reducir subestimaciones. No se debe decir que el Random Forest por si solo evita quiebres; la reduccion principal en backtesting aparece con:

```text
modelo predictivo + factor de seguridad 1.08
```

## 12. Metricas

MAE:

```math
MAE = \frac{1}{n} \sum |y - \hat{y}|
```

Mide el error absoluto promedio. Mientras menor sea, mas cerca estan las predicciones de la demanda real.

WAPE:

```math
WAPE = \frac{\sum |y - \hat{y}|}{\sum y} \times 100
```

Mide el error absoluto ponderado por el volumen total de demanda. En los scripts se calcula de forma global, no como promedio semanal.

TQS:

```math
TQS = \frac{\text{numero de semanas donde demanda\_real > prediccion}}{\text{numero total de semanas}} \times 100
```

Mide el porcentaje de semanas en las que se subestimo la demanda y, por tanto, existiria riesgo de quiebre de stock.

## 13. Archivos generados

Los scripts generan y/o usan estos archivos dentro de `resultados/`:

- `resultados/baseline_jesus.csv`
- `resultados/random_forest_inicial_jesus.csv`
- `resultados/metricas_modelos_jesus.csv`
- `resultados/random_forest_optimizado_jesus.csv`
- `resultados/comparacion_modelos_jesus.csv`
- `resultados/evaluacion_factores_jesus.csv`
- `resultados/modelo_final_jesus.csv`
- `resultados/resumen_final_jesus.csv`
- `resultados/backtesting_semanal_jesus.csv`
- `resultados/resumen_backtesting_jesus.csv`

Estos archivos existen actualmente en la carpeta `resultados/`.

## 14. Como ejecutar

El orden de ejecucion recomendado es:

```bash
python 01_baseline_jesus.py
python 02_random_forest_jesus.py
python 03_optimizacion_jesus.py
python 04_equilibrio_negocio_jesus.py
python 05_backtesting_jesus.py
```

Dependencias importadas por los scripts:

- `pandas`
- `numpy`
- `scikit-learn`
- `pathlib`, incluido en la biblioteca estandar de Python.
- `importlib` y `typing`, usados en `03_optimizacion_jesus.py` y tambien parte de la biblioteca estandar.

## 15. Conclusion

En esta parte del proyecto se desarrollo un flujo de prediccion semanal de demanda con regresion supervisada. Primero se construyo una linea base con `lag_1`, luego se entreno un Random Forest inicial, se optimizaron hiperparametros con validacion temporal y se agrego un factor de seguridad seleccionado sin usar el test final.

La validacion final se realizo mediante backtesting temporal tipo expanding window. En 40 semanas evaluadas, el modelo ajustado redujo los quiebres de 18 a 10. El TQS bajo de 45% a 25%, mientras que el WAPE aumento de 15.88% a 22.87%. Por tanto, existe un compromiso claro entre precision predictiva y reduccion de quiebres de stock.

No se afirma que se haya alcanzado un TQS de 5%, porque con las muestras evaluadas esa meta exacta no es representable y los resultados obtenidos no llegan a ese nivel.

## Nota de consistencia

La documentacion fue elaborada leyendo los scripts `01` a `05` y verificando los CSV existentes en `resultados/`. No se encontro una inconsistencia importante entre los scripts y los resultados documentados. La aclaracion principal es que las metricas de `01_baseline_jesus.py` se calculan sobre 88 semanas, mientras que las comparaciones de modelos usan subconjuntos temporales o backtesting; por eso el resultado principal debe basarse en el backtesting temporal.
