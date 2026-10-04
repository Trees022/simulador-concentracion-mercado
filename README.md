# Simulador de concentración de mercado

Proyecto educativo de Organización Industrial desarrollado con Python.

## Estado actual

Están implementados el módulo matemático, el motor Monte Carlo, la interfaz
Streamlit, los gráficos Plotly y el evaluador automático con retroalimentación
cuantitativa.

## Preparación y pruebas

```powershell
python -m pip install -r requirements.txt
python -m pytest
```

Para abrir la aplicación:

```powershell
python -m streamlit run app.py
```

La muestra Monte Carlo se guarda en `session_state` y solo se genera al pulsar
el botón correspondiente. Cambiar el caso o el indicador mostrado conserva la
muestra. Si cambian `N`, `k`, las iteraciones o la semilla, la comparación se
bloquea hasta ejecutar una simulación compatible.

## Convenciones matemáticas

- Las cuotas se expresan internamente como proporciones entre 0 y 1.
- Se exige que cada mercado sume 1 con tolerancia absoluta `1e-10` y
  tolerancia relativa cero.
- Las entradas incorrectas nunca se normalizan automáticamente.
- `N` incluye todas las cuotas, incluidas las cuotas cero.
- Las cuotas no se redondean antes de calcular los indicadores.

## Método Monte Carlo

Los mercados se generan con `numpy.random.default_rng` y una distribución
Dirichlet cuyos `N` parámetros son iguales a 1. `Dirichlet(1,...,1)` es
uniforme sobre el conjunto de vectores de cuotas no negativas que suman 1.
Esto no implica que cada cuota individual sea uniforme ni que el modelo
represente todos los mercados reales.

Para la misma matriz de mercados se calculan de forma vectorizada:

- CRk;
- IHH decimal e IHH en puntos;
- índice de dominancia de García Alba;
- entropía original y normalizada.

La esperanza teórica del IHH decimal bajo este supuesto es:

\[
E[IHH]=\frac{2}{N+1}.
\]

Una semilla permite reproducir la simulación. Los casos aleatorios
particulares usan un generador separado. Además, las cuotas del caso no son un
argumento del motor Monte Carlo: pueden cambiarse sin regenerar la muestra.

## Percentil empírico

\[
100\frac{\#\{x_j \leq x_{caso}\}}{\text{número de simulaciones}}
\]

Los empates están incluidos en el numerador. En el caso de la entropía, un
percentil alto significa mayor entropía, no mayor concentración.

## Límites y rendimiento

- `N`: entero entre 2 y 100.
- `k`: entero entre 1 y `N`.
- Iteraciones predeterminadas: 1.000.
- Máximo inicial: 100.000 iteraciones.

La medición local final del caso máximo (`N=100`, `k=4`, 100.000 iteraciones),
incluyendo generación, indicadores y compactación para `session_state`, produjo:

- tiempo: 0,433 segundos;
- arreglos conservados por `session_state`: 3,052 MiB;
- memoria trazada tras liberar la matriz completa: 3,899 MiB;
- pico transitorio de memoria trazada: 166,027 MiB.

La medición se realizó con Python 3.13.0 y NumPy 2.2.4. Los valores pueden
cambiar según el equipo y el entorno de despliegue.

La matriz de cuotas se libera después de obtener las cuatro series necesarias
para la interfaz. El límite de 100.000 todavía debe verificarse en el servidor
de Streamlit Community Cloud.

> Aumentar las iteraciones incrementa el tiempo de respuesta y el consumo de
> memoria y procesamiento.

## Evaluación didáctica del IHH

La pregunta principal clasifica el caso con precisión completa según esta
convención operativa del simulador:

- baja: `IHH < 1500`;
- moderada: `1500 ≤ IHH < 2500`;
- alta: `IHH ≥ 2500`.

La asignación de los valores exactamente iguales a 1.500 y 2.500 es una
convención didáctica para cubrir las fronteras. La retroalimentación siempre
usa el IHH y su percentil Monte Carlo, aunque el gráfico seleccionado muestre
CRk, ID o IE.

Referencia: [Guía para el Análisis de Operaciones de Concentración
Horizontales, FNE, mayo de 2022](https://www.fne.gob.cl/wp-content/uploads/2022/05/20220531.-Guia-para-el-Analisis-de-Operaciones-de-Concentracion-Horizontales-version-final-en-castellano.pdf).

La concentración por sí sola no demuestra una conducta anticompetitiva ni
reemplaza el análisis de una operación de concentración. El percentil depende
de `N` y del modelo Dirichlet y no constituye un umbral normativo. Las preguntas
complementarias sobre CRk, ID y entropía no asignan umbrales normativos a esos
indicadores.

La interfaz redondea únicamente la presentación: CRk e IHH se muestran con 2
decimales; ID, IE y la entropía normalizada, con 4. Los cálculos, percentiles y
clasificaciones conservan la precisión completa. La retroalimentación distingue
la clasificación obtenida mediante los umbrales didácticos del percentil que
compara el caso con la muestra Monte Carlo real de la simulación vigente.

Para que el despliegue pueda abrirse sin iniciar sesión, se puede usar un
repositorio público —la aplicación será pública por defecto— o cambiar la
privacidad a **This app is public and searchable** desde la configuración de
Streamlit Community Cloud.
