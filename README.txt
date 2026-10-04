SIMULADOR DE CONCENTRACIÓN DE MERCADO
=====================================

Aplicación educativa de Organización Industrial desarrollada con Python,
Streamlit, NumPy, pandas y Plotly. Permite construir un caso, generar mercados
hipotéticos mediante Monte Carlo, comparar indicadores y responder una
evaluación con reglas matemáticas.

VERSIÓN PROBADA
---------------
Python 3.13.0 en Windows, con estas dependencias directas:

- NumPy 2.2.4
- pandas 3.0.6
- Plotly 7.1.0
- Streamlit 1.65.0
- pytest 8.4.2

Las versiones están fijadas en requirements.txt. Streamlit Community Cloud
permite escoger la versión de Python en la configuración avanzada. Para
reproducir la validación local, seleccionar Python 3.13.

CONTENIDO
---------
- app.py: interfaz Streamlit.
- concentracion/indices.py: validación e indicadores.
- concentracion/simulacion.py: motor Monte Carlo y percentiles.
- concentracion/graficos.py: gráficos Plotly.
- concentracion/evaluacion.py: evaluación automática.
- tests/: pruebas automáticas.
- .streamlit/config.toml: tema y configuración de la aplicación.

INSTALACIÓN LIMPIA
------------------
Desde la raíz del proyecto, en PowerShell:

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

En macOS o Linux, activar el entorno con:

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

EJECUCIÓN
---------
Desde la raíz del proyecto:

python -m streamlit run app.py

Si se usa el entorno virtual de PowerShell sin activarlo:

.\.venv\Scripts\python.exe -m streamlit run app.py

PRUEBAS
-------
python -m pytest -q

RECORRIDO
---------
1. Configurar N, indicador, k cuando se usa CRk, iteraciones y semilla.
2. Pulsar "Ejecutar simulación".
3. Editar exactamente N cuotas porcentuales, generar un caso aleatorio o
   cargar el ejemplo 40-30-20-10.
4. Revisar cuotas, indicadores, histograma y percentil empírico.
5. Responder la evaluación principal de IHH y las preguntas complementarias.

La muestra se almacena en session_state y solo se regenera al pulsar el botón.
Cambiar el caso o el indicador visualizado no ejecuta nuevamente Monte Carlo.
Cambiar N, k, iteraciones o semilla bloquea las comparaciones hasta ejecutar
una muestra compatible.

FÓRMULAS
--------
Las cuotas internas s_i son proporciones entre 0 y 1.

CRk = suma de las k cuotas más grandes.

IHH decimal = suma(s_i^2).
IHH en puntos = 10.000 * suma(s_i^2).

ID de García Alba = suma(s_i^4) / [suma(s_i^2)]^2.

Entropía = -suma[s_i * ln(s_i)].
Las cuotas cero aportan cero por continuidad.

Entropía normalizada = Entropía / ln(N).
N incluye todas las cuotas, también las cuotas cero.

VALIDACIÓN NUMÉRICA
-------------------
- N debe ser entero entre 2 y 100.
- k debe ser entero entre 1 y N.
- Cada cuota porcentual debe estar entre 0 y 100.
- Deben existir exactamente N cuotas.
- La suma debe ser cercana a 1 con tolerancia absoluta 1e-10 y tolerancia
  relativa cero.
- Las entradas incorrectas no se normalizan silenciosamente.
- No se redondean cuotas antes de calcular; solo se redondea la presentación.

MÉTODO MONTE CARLO Y SUPUESTOS
------------------------------
Cada mercado se genera con numpy.random.default_rng y Dirichlet(1,...,1).
Esta distribución es uniforme sobre el conjunto de vectores no negativos que
suman 1. No significa que cada cuota individual sea uniforme ni que el modelo
represente todos los mercados reales.

Se calculan CRk, IHH decimal y en puntos, ID, entropía original y normalizada
para los mismos mercados. La esperanza teórica del IHH decimal es 2/(N+1).
La semilla permite reproducir la muestra. El caso aleatorio utiliza un
generador separado.

Percentil = 100 * (valores simulados <= valor del caso) / simulaciones.
Los empates se incluyen. Para entropía, un percentil alto significa mayor
entropía, no mayor concentración. El percentil depende de N y de Dirichlet y
no constituye un umbral normativo.

LÍMITES Y MEMORIA
-----------------
- Iteraciones predeterminadas: 1.000.
- Máximo inicial: 100.000.
- Aumentar las iteraciones incrementa la latencia, memoria y procesamiento.

Medición local final para N=100, k=4 y 100.000 iteraciones, incluyendo todos
los indicadores, en Python 3.13.0 y NumPy 2.2.4:

- Tiempo: 0,433 segundos.
- Pico transitorio de memoria trazada: 166,027 MiB.
- Arreglos conservados por session_state después de compactar: 3,052 MiB.
- Memoria trazada actual después de liberar el resultado completo: 3,899 MiB.

La matriz de 10 millones de cuotas se usa para calcular y validar, pero no se
conserva en session_state. Solo se guarda la última muestra compacta. Estas son
mediciones locales; el tiempo, el pico y el límite de 100.000 todavía deben
comprobarse en el servidor de Streamlit Community Cloud.

EVALUACIÓN DIDÁCTICA DEL IHH
----------------------------
Convención operativa del simulador:

- Baja: IHH < 1500.
- Moderada: 1500 <= IHH < 2500.
- Alta: IHH >= 2500.

Los valores exactamente iguales a 1500 y 2500 se asignan por una convención
didáctica para cubrir las fronteras. La clasificación usa precisión completa.
La concentración por sí sola no demuestra una conducta anticompetitiva ni
reemplaza el análisis de una operación de concentración.

FUENTES
-------
Fiscalía Nacional Económica de Chile. Guía para el Análisis de Operaciones de
Concentración Horizontales, mayo de 2022:
https://www.fne.gob.cl/wp-content/uploads/2022/05/20220531.-Guia-para-el-Analisis-de-Operaciones-de-Concentracion-Horizontales-version-final-en-castellano.pdf

Documentación oficial de despliegue de Streamlit Community Cloud:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

DESPLIEGUE EN STREAMLIT COMMUNITY CLOUD
---------------------------------------
1. Subir estos archivos a un repositorio de GitHub.
2. Ingresar a https://share.streamlit.io y conectar la cuenta de GitHub.
3. Elegir "Create app" y confirmar que ya existe una aplicación.
4. Seleccionar el repositorio y la rama que contenga esta entrega.
5. Indicar app.py como archivo principal.
6. En "Advanced settings", seleccionar Python 3.13.
7. No agregar secretos: esta aplicación no requiere credenciales.
8. Para acceso sin inicio de sesión, usar un repositorio público (la app será
   pública por defecto) o, si el repositorio es privado, abrir la configuración
   de la app y elegir "This app is public and searchable" en "Who can view
   this app".
9. Pulsar "Deploy" y revisar los registros de instalación y ejecución.
10. Probar especialmente N=100 y 100.000 iteraciones antes de considerar fijo
   ese límite en el servidor.

Información oficial para compartir una aplicación pública:
https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app
