"""Interfaz Streamlit del simulador de concentración de mercado."""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import streamlit as st

from concentracion.evaluacion import (
    ACLARACION_COMPETENCIA,
    ACLARACION_CONVENCION,
    CLASIFICACIONES,
    FUENTE_FNE_TITULO,
    FUENTE_FNE_URL,
    PREGUNTA_PRINCIPAL,
    evaluar_respuesta_ihh,
    huella_cuotas,
)
from concentracion.graficos import (
    INDICADORES,
    convertir_valor_presentacion,
    crear_grafico_cuotas,
    crear_histograma_comparativo,
    formatear_numero_indicador,
    formatear_valor_indicador,
    obtener_valores_simulados,
)
from concentracion.indices import (
    TOLERANCIA_SUMA,
    calcular_crk,
    calcular_entropia,
    calcular_entropia_normalizada,
    calcular_id_garcia_alba,
    calcular_ihh_decimal,
    calcular_ihh_puntos,
    validar_cuotas,
)
from concentracion.simulacion import (
    ADVERTENCIA_RENDIMIENTO,
    ITERACIONES_PREDETERMINADAS,
    MAX_ITERACIONES,
    NOTA_PERCENTIL_ENTROPIA,
    calcular_percentil_empirico,
    compactar_resultado,
    generar_caso_aleatorio,
    simular_mercados,
)


st.set_page_config(
    page_title="Simulador de concentración de mercado",
    page_icon="📊",
    layout="wide",
)


ETIQUETAS_INDICADORES = {
    "CRk": "CRk · Razón de concentración",
    "IHH": "IHH · Herfindahl-Hirschman",
    "ID": "ID · Dominancia de García Alba",
    "IE": "IE · Entropía",
}


def _inicializar_estado() -> None:
    """Crea solo las claves necesarias durante la primera ejecución."""
    valores_iniciales = {
        "config_n": 4,
        "config_k": 2,
        "config_iteraciones": ITERACIONES_PREDETERMINADAS,
        "config_semilla": 2026,
        "indicador_seleccionado": "IHH",
        "resultado_simulacion": None,
        "configuracion_simulada": None,
        "tiempo_simulacion": None,
        "version_simulacion": 0,
        "cuotas_caso_pct": [25.0, 25.0, 25.0, 25.0],
        "caso_n": 4,
        "version_editor": 0,
        "mensaje_caso": None,
        "contexto_evaluacion": None,
        "respuesta_ihh": None,
        "retroalimentacion_ihh": None,
        "respuesta_crk_complementaria": "",
        "respuesta_id_complementaria": None,
        "respuesta_ie_complementaria": None,
        "retroalimentacion_complementaria": None,
        "firma_complementaria": None,
    }
    for clave, valor in valores_iniciales.items():
        if clave not in st.session_state:
            st.session_state[clave] = valor


def _reiniciar_caso_para_n(n: int) -> None:
    """Asegura exactamente N filas cuando cambia el tamaño del mercado."""
    if st.session_state.caso_n == n:
        return
    st.session_state.cuotas_caso_pct = [100.0 / n] * n
    st.session_state.caso_n = n
    st.session_state.version_editor += 1
    st.session_state.mensaje_caso = (
        f"El caso se reinició con {n} cuotas iguales para coincidir con N={n}."
    )


def _generar_caso_desde_boton(n: int) -> None:
    """Callback: genera cuotas con un RNG independiente de Monte Carlo."""
    cuotas = generar_caso_aleatorio(n)
    st.session_state.cuotas_caso_pct = (cuotas * 100.0).tolist()
    st.session_state.caso_n = n
    st.session_state.version_editor += 1
    st.session_state.mensaje_caso = (
        "Se generó un caso aleatorio independiente de la muestra Monte Carlo."
    )


def _cargar_ejemplo_desde_boton() -> None:
    """Callback: carga 40-30-20-10 y ajusta la configuración a N=4."""
    st.session_state.config_n = 4
    if st.session_state.config_k > 4:
        st.session_state.config_k = 4
    st.session_state.cuotas_caso_pct = [40.0, 30.0, 20.0, 10.0]
    st.session_state.caso_n = 4
    st.session_state.version_editor += 1
    st.session_state.mensaje_caso = (
        "Se cargó el ejemplo 40 %, 30 %, 20 % y 10 %. N se ajustó a 4. "
        "Si la simulación guardada usa otra configuración, debes ejecutarla nuevamente."
    )


def _validar_porcentajes_caso(
    valores: object, n: int
) -> tuple[np.ndarray | None, float | None, list[str]]:
    """Valida entradas porcentuales sin corregirlas ni normalizarlas."""
    errores: list[str] = []
    try:
        porcentajes = np.asarray(valores, dtype=np.float64)
    except (TypeError, ValueError):
        return None, None, ["Todas las cuotas deben ser valores numéricos."]

    if porcentajes.ndim != 1 or porcentajes.size != n:
        return None, None, [f"La tabla debe contener exactamente {n} cuotas."]

    no_finitas = np.flatnonzero(~np.isfinite(porcentajes))
    if no_finitas.size:
        filas = ", ".join(str(int(i) + 1) for i in no_finitas)
        errores.append(f"Hay valores vacíos o no finitos en las filas: {filas}.")

    fuera_de_rango = np.flatnonzero(
        np.isfinite(porcentajes) & ((porcentajes < 0.0) | (porcentajes > 100.0))
    )
    if fuera_de_rango.size:
        filas = ", ".join(str(int(i) + 1) for i in fuera_de_rango)
        errores.append(
            f"Cada cuota debe estar entre 0 % y 100 %. Revisa las filas: {filas}."
        )

    if errores:
        return None, None, errores

    total = math.fsum(porcentajes.tolist())
    cuotas = porcentajes / 100.0
    try:
        cuotas_validadas = validar_cuotas(cuotas, n)
    except ValueError:
        errores.append(
            "Las cuotas deben sumar 100 % dentro de una tolerancia absoluta "
            f"equivalente a {TOLERANCIA_SUMA:g} en proporciones. "
            f"La suma ingresada es {total:.12g} %. No se normalizarán los datos."
        )
        return None, total, errores
    return cuotas_validadas, total, errores


def _calcular_indicadores_caso(cuotas: np.ndarray, n: int, k: int) -> dict[str, float]:
    """Calcula valores internos sin redondearlos."""
    return {
        "CRk": calcular_crk(cuotas, n, k),
        "IHH": calcular_ihh_puntos(cuotas, n),
        "ID": calcular_id_garcia_alba(cuotas, n),
        "IE": calcular_entropia(cuotas, n),
    }


def _configuracion_actual(n: int, k: int, iteraciones: int, semilla: int) -> dict[str, int]:
    return {
        "n": n,
        "k": k,
        "iteraciones": iteraciones,
        "semilla": semilla,
    }


def _reiniciar_evaluacion_si_cambio(contexto_actual: tuple[str | None, int | None]) -> None:
    """Elimina respuestas antiguas cuando cambia el caso o la simulación."""
    if st.session_state.contexto_evaluacion == contexto_actual:
        return
    st.session_state.respuesta_ihh = None
    st.session_state.retroalimentacion_ihh = None
    st.session_state.respuesta_crk_complementaria = ""
    st.session_state.respuesta_id_complementaria = None
    st.session_state.respuesta_ie_complementaria = None
    st.session_state.retroalimentacion_complementaria = None
    st.session_state.firma_complementaria = None
    st.session_state.contexto_evaluacion = contexto_actual


_inicializar_estado()

st.markdown(
    """
    <style>
    .block-container {max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {border: 1px solid #d1d5db; padding: 0.8rem; border-radius: 0.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Simulador de concentración de mercado")
st.caption(
    "Explora estructuras hipotéticas de mercado, define un caso y compáralo "
    "con una referencia Monte Carlo."
)

with st.expander("Fórmulas utilizadas"):
    st.markdown(
        r"""
        Las cuotas se calculan internamente como proporciones \(s_i\) sin
        redondear.

        - \(CR_k=\sum_{i=1}^{k}s_{(i)}\), sumando las cuotas mayores.
        - \(IHH=\sum_i s_i^2\); en puntos se multiplica por 10.000.
        - \(ID=\frac{\sum_i s_i^4}{(\sum_i s_i^2)^2}\).
        - \(IE=-\sum_i s_i\ln(s_i)\), en nats. Las cuotas cero aportan cero.
        - Entropía normalizada: \(IE/\ln(N)\), usando todas las cuotas en N.
        """
    )

with st.expander("Supuesto de la simulación Dirichlet"):
    st.markdown(
        "Cada mercado se genera con **Dirichlet(1,…,1)**. Es uniforme sobre "
        "el conjunto de vectores no negativos que suman 1, pero esto no "
        "significa que cada cuota individual sea uniforme ni que el modelo "
        "represente todos los mercados reales. Es una referencia educativa."
    )

st.header("1. Configurar y simular el mercado")

columna_n, columna_indicador = st.columns(2)
with columna_n:
    n = int(
        st.number_input(
            "Cantidad de empresas (N)",
            min_value=2,
            max_value=100,
            step=1,
            key="config_n",
        )
    )
with columna_indicador:
    indicador = st.selectbox(
        "Indicador para comparar",
        options=list(INDICADORES),
        format_func=lambda clave: ETIQUETAS_INDICADORES[clave],
        key="indicador_seleccionado",
    )

if st.session_state.config_k > n:
    st.session_state.config_k = n

columna_k, columna_iteraciones, columna_semilla = st.columns(3)
with columna_k:
    if indicador == "CRk":
        k = int(
            st.number_input(
                "Empresas incluidas en CRk (k)",
                min_value=1,
                max_value=n,
                step=1,
                key="config_k",
            )
        )
    else:
        k = int(st.session_state.config_k)
        st.text_input(
            "Empresas incluidas en CRk (k)",
            value=str(k),
            disabled=True,
            help="k se aplica al seleccionar CRk.",
        )
with columna_iteraciones:
    iteraciones = int(
        st.number_input(
            "Iteraciones",
            min_value=1,
            max_value=MAX_ITERACIONES,
            step=100,
            key="config_iteraciones",
        )
    )
with columna_semilla:
    semilla = int(
        st.number_input(
            "Semilla",
            min_value=0,
            max_value=2_147_483_647,
            step=1,
            key="config_semilla",
        )
    )

st.warning(
    f"{ADVERTENCIA_RENDIMIENTO} El máximo de {MAX_ITERACIONES:,} se justificó "
    "con mediciones locales y puede requerir ajustes en el servidor de despliegue."
    .replace(",", ".")
)

configuracion_actual = _configuracion_actual(n, k, iteraciones, semilla)

if st.button("Ejecutar simulación", type="primary", key="ejecutar_simulacion"):
    inicio = time.perf_counter()
    with st.spinner("Generando mercados y calculando indicadores…"):
        try:
            resultado_nuevo = simular_mercados(
                n=n,
                k=k,
                iteraciones=iteraciones,
                semilla=semilla,
            )
        except ValueError as error:
            st.error(f"No se pudo ejecutar la simulación: {error}")
        else:
            # Solo se conserva la muestra más reciente para limitar la memoria.
            st.session_state.resultado_simulacion = compactar_resultado(
                resultado_nuevo
            )
            del resultado_nuevo
            st.session_state.configuracion_simulada = configuracion_actual.copy()
            st.session_state.tiempo_simulacion = time.perf_counter() - inicio
            st.session_state.version_simulacion += 1

resultado = st.session_state.resultado_simulacion
configuracion_simulada = st.session_state.configuracion_simulada
simulacion_compatible = (
    resultado is not None and configuracion_simulada == configuracion_actual
)

if resultado is None:
    st.info("Configura el mercado y pulsa «Ejecutar simulación» para comenzar.")
elif simulacion_compatible:
    st.success(
        f"Simulación disponible: {resultado.iteraciones:,} mercados en "
        f"{st.session_state.tiempo_simulacion:.3f} segundos."
        .replace(",", ".")
    )
else:
    st.error(
        "La configuración cambió después de la última simulación. Los resultados "
        "guardados no son compatibles; ejecuta nuevamente antes de comparar."
    )

st.header("2. Definir el caso particular")
_reiniciar_caso_para_n(n)

columna_aleatorio, columna_ejemplo = st.columns(2)
with columna_aleatorio:
    st.button(
        "Generar caso aleatorio",
        key="generar_caso_aleatorio",
        on_click=_generar_caso_desde_boton,
        args=(n,),
        width="stretch",
    )
with columna_ejemplo:
    st.button(
        "Cargar ejemplo 40 %, 30 %, 20 % y 10 %",
        key="cargar_ejemplo",
        on_click=_cargar_ejemplo_desde_boton,
        width="stretch",
    )

if st.session_state.mensaje_caso:
    st.info(st.session_state.mensaje_caso)
    st.session_state.mensaje_caso = None

datos_editor = pd.DataFrame(
    {
        "Empresa": [f"Empresa {i}" for i in range(1, n + 1)],
        "Cuota (%)": st.session_state.cuotas_caso_pct,
    }
)
tabla_editada = st.data_editor(
    datos_editor,
    key=f"editor_cuotas_{st.session_state.version_editor}",
    hide_index=True,
    num_rows="fixed",
    width="stretch",
    disabled=["Empresa"],
    column_config={
        "Empresa": st.column_config.TextColumn("Empresa"),
        "Cuota (%)": st.column_config.NumberColumn(
            "Cuota (%)",
            help="Cada cuota debe estar entre 0 y 100. El total debe ser 100.",
            min_value=0.0,
            max_value=100.0,
            step=0.01,
            format="%.6f",
            required=True,
        ),
    },
)

valores_editados = tabla_editada["Cuota (%)"].tolist()
st.session_state.cuotas_caso_pct = valores_editados
cuotas_caso, suma_porcentajes, errores_caso = _validar_porcentajes_caso(
    valores_editados, n
)

if suma_porcentajes is None:
    st.metric("Suma ingresada", "No calculable")
else:
    st.metric("Suma ingresada", f"{suma_porcentajes:.10f} %")

for error in errores_caso:
    st.error(error)

indicadores_caso: dict[str, float] | None = None
if cuotas_caso is not None:
    st.success("Caso válido: las cuotas se usarán sin normalización ni redondeo interno.")
    indicadores_caso = _calcular_indicadores_caso(cuotas_caso, n, k)

    st.plotly_chart(
        crear_grafico_cuotas(cuotas_caso),
        width="stretch",
        config={"displaylogo": False, "responsive": True},
    )

    filas_indicadores = [
        {
            "Indicador": f"CR{k}",
            "Valor": formatear_numero_indicador(
                convertir_valor_presentacion(indicadores_caso["CRk"], "CRk"),
                "CRk",
            ),
            "Unidad": "%",
        },
        {
            "Indicador": "IHH",
            "Valor": formatear_numero_indicador(indicadores_caso["IHH"], "IHH"),
            "Unidad": "puntos",
        },
        {
            "Indicador": "ID",
            "Valor": formatear_numero_indicador(indicadores_caso["ID"], "ID"),
            "Unidad": "adimensional",
        },
        {
            "Indicador": "IE",
            "Valor": formatear_numero_indicador(indicadores_caso["IE"], "IE"),
            "Unidad": "nats",
        },
    ]
    st.subheader("Indicadores del caso")
    st.dataframe(
        pd.DataFrame(filas_indicadores),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"IHH decimal: {calcular_ihh_decimal(cuotas_caso, n):.2f} · "
        f"Entropía normalizada: {calcular_entropia_normalizada(cuotas_caso, n):.4f}"
    )

st.header("3. Comparar los resultados")

if resultado is None:
    st.info("Primero ejecuta una simulación en el paso 1.")
elif not simulacion_compatible:
    st.warning(
        "La comparación está bloqueada porque N, k, semilla o iteraciones no "
        "coinciden con la simulación guardada. Ejecuta nuevamente el paso 1."
    )
elif cuotas_caso is None or indicadores_caso is None:
    st.warning("Corrige las cuotas del caso antes de realizar la comparación.")
else:
    valores_simulados = obtener_valores_simulados(resultado, indicador)
    valor_caso_presentacion = convertir_valor_presentacion(
        indicadores_caso[indicador], indicador
    )
    percentil = calcular_percentil_empirico(
        valores_simulados, valor_caso_presentacion
    )

    columna_percentil, columna_valor = st.columns(2)
    with columna_percentil:
        st.metric("Percentil empírico", f"{percentil:.2f} %")
    with columna_valor:
        st.metric(
            "Valor del caso",
            formatear_valor_indicador(valor_caso_presentacion, indicador),
        )

    st.plotly_chart(
        crear_histograma_comparativo(
            valores_simulados,
            valor_caso_presentacion,
            indicador,
            k=k,
        ),
        width="stretch",
        config={"displaylogo": False, "responsive": True},
    )
    st.caption(
        "Percentil = 100 × simulaciones con valor menor o igual al caso / "
        "cantidad total. Los empates se incluyen."
    )
    if indicador == "IE":
        st.info(NOTA_PERCENTIL_ENTROPIA)

st.header("4. Evaluar la comprensión")

st.markdown(f"**Referencia:** [{FUENTE_FNE_TITULO}]({FUENTE_FNE_URL}).")
st.caption(
    "La guía usa el IHH como punto de partida del análisis. En este ejercicio "
    "se adopta una clasificación didáctica explícita para cubrir las fronteras."
)
st.info(
    "**Convención del simulador:** baja si IHH < 1.500; moderada si "
    "1.500 ≤ IHH < 2.500; alta si IHH ≥ 2.500. "
    f"{ACLARACION_CONVENCION} {ACLARACION_COMPETENCIA}"
)

huella_actual = huella_cuotas(cuotas_caso, n) if cuotas_caso is not None else None
version_compatible = (
    int(st.session_state.version_simulacion) if simulacion_compatible else None
)
contexto_evaluacion_actual = (huella_actual, version_compatible)
_reiniciar_evaluacion_si_cambio(contexto_evaluacion_actual)

evaluacion_habilitada = (
    cuotas_caso is not None
    and indicadores_caso is not None
    and resultado is not None
    and simulacion_compatible
)

if not evaluacion_habilitada:
    st.warning(
        "La evaluación está bloqueada. Se necesita un caso válido y una "
        "simulación compatible con N, k, iteraciones y semilla."
    )

st.subheader("Pregunta principal · IHH")
st.write(PREGUNTA_PRINCIPAL)
respuesta_ihh = st.radio(
    "Selecciona una alternativa",
    options=list(CLASIFICACIONES),
    index=None,
    format_func=str.capitalize,
    key="respuesta_ihh",
    horizontal=True,
    disabled=not evaluacion_habilitada,
)

retroalimentacion_guardada = st.session_state.retroalimentacion_ihh
if (
    retroalimentacion_guardada is not None
    and respuesta_ihh != retroalimentacion_guardada.respuesta_usuario
):
    st.session_state.retroalimentacion_ihh = None

if st.button(
    "Comprobar respuesta",
    key="comprobar_respuesta_ihh",
    type="primary",
    disabled=not evaluacion_habilitada,
):
    if respuesta_ihh is None:
        st.warning("Selecciona baja, moderada o alta antes de comprobar.")
    else:
        st.session_state.retroalimentacion_ihh = evaluar_respuesta_ihh(
            respuesta_ihh,
            cuotas_caso,
            n,
            resultado.ihh_puntos,
        )

retroalimentacion = st.session_state.retroalimentacion_ihh
if retroalimentacion is not None:
    if retroalimentacion.es_correcta:
        st.success("Respuesta correcta.")
    else:
        st.error(
            f"Respuesta incorrecta. Elegiste «{retroalimentacion.respuesta_usuario}»."
        )

    st.markdown(
        f"""
        1. **Clasificación correcta:** {retroalimentacion.clasificacion_correcta}.
        2. **IHH del caso:** {retroalimentacion.ihh_puntos:.2f} puntos.
        3. **Intervalo:** {retroalimentacion.intervalo}.
        4. **Justificación:** {retroalimentacion.justificacion}
        5. **Percentil real del IHH:** {retroalimentacion.percentil:.2f} %.
        6. **Interpretación:** {retroalimentacion.explicacion_percentil}
        7. **Umbrales y percentil:** {retroalimentacion.distincion_umbral_percentil}
        8. **Alcance:** {retroalimentacion.aclaracion_percentil}
        """
    )
    st.caption(
        "Esta evaluación siempre utiliza IHH y su distribución simulada, "
        "aunque el gráfico principal muestre CRk, ID o IE."
    )

if evaluacion_habilitada:
    with st.expander("Preguntas complementarias"):
        st.caption(
            "Estas preguntas interpretan CRk, ID e IE. No se aplican umbrales "
            "normativos a estos indicadores."
        )
        respuesta_crk = st.text_input(
            f"1. ¿Qué porcentaje concentran las {k} empresas mayores según CR{k}?",
            key="respuesta_crk_complementaria",
            placeholder="Escribe un porcentaje, por ejemplo 70",
        )

        opciones_id = [
            "Cómo se reparte entre las empresas el peso de sus cuotas al cuadrado.",
            "La cantidad total de empresas, sin considerar sus cuotas.",
            "Una categoría legal automática de conducta anticompetitiva.",
        ]
        respuesta_id = st.radio(
            "2. ¿Qué mide el ID de García Alba?",
            options=opciones_id,
            index=None,
            key="respuesta_id_complementaria",
        )

        opciones_ie = [
            "Porque aumenta cuando las cuotas se distribuyen de forma más uniforme.",
            "Porque solo considera la cuota de la empresa más grande.",
            "Porque aplica el mismo umbral normativo que el IHH.",
        ]
        respuesta_ie = st.radio(
            "3. ¿Por qué una entropía mayor representa cuotas más repartidas?",
            options=opciones_ie,
            index=None,
            key="respuesta_ie_complementaria",
        )

        firma_complementaria_actual = (respuesta_crk, respuesta_id, respuesta_ie)
        if st.session_state.firma_complementaria != firma_complementaria_actual:
            st.session_state.retroalimentacion_complementaria = None

        if st.button(
            "Comprobar preguntas complementarias",
            key="comprobar_complementarias",
        ):
            mensajes: list[tuple[bool, str]] = []
            crk_correcto = indicadores_caso["CRk"] * 100.0
            try:
                crk_ingresado = float(respuesta_crk.strip().replace(",", "."))
            except (AttributeError, ValueError):
                mensajes.append(
                    (False, f"CR{k}: ingresa un número. El valor es {crk_correcto:.2f} %.")
                )
            else:
                coincide = math.isclose(
                    crk_ingresado,
                    crk_correcto,
                    rel_tol=0.0,
                    abs_tol=0.01,
                )
                mensajes.append(
                    (
                        coincide,
                        f"CR{k}: las {k} empresas mayores concentran "
                        f"{crk_correcto:.2f} %.",
                    )
                )

            mensajes.append(
                (
                    respuesta_id == opciones_id[0],
                    "ID: mide cómo se distribuye el peso relativo de las cuotas "
                    "al cuadrado; aumenta cuando ese peso está más dominado por "
                    "pocas empresas.",
                )
            )
            mensajes.append(
                (
                    respuesta_ie == opciones_ie[0],
                    "IE: aumenta cuando la incertidumbre sobre qué empresa posee "
                    "una unidad de cuota es mayor, lo que ocurre con cuotas más repartidas.",
                )
            )
            st.session_state.retroalimentacion_complementaria = mensajes
            st.session_state.firma_complementaria = firma_complementaria_actual

        if st.session_state.retroalimentacion_complementaria is not None:
            for es_correcta, mensaje in st.session_state.retroalimentacion_complementaria:
                if es_correcta:
                    st.success(mensaje)
                else:
                    st.error(mensaje)

st.divider()
st.caption(
    "Herramienta educativa: los resultados dependen del supuesto de generación "
    "y no sustituyen un análisis económico del mercado real."
)
