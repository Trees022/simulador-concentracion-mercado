"""Gráficos Plotly para cuotas e indicadores simulados.

Este módulo no depende de Streamlit. Las figuras se construyen por separado
para poder verificarlas con pruebas automáticas y reutilizarlas en la interfaz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray

from .indices import validar_cuotas
from .simulacion import ResultadoSimulacion, ResultadoSimulacionCompacto


COLOR_SIMULACIONES = "#2563EB"
COLOR_CASO = "#DC2626"
COLOR_CUOTAS = "#0F766E"


@dataclass(frozen=True)
class DefinicionIndicador:
    """Metadatos de presentación para un indicador."""

    clave: str
    nombre: str
    unidad: str
    campo_simulacion: str
    factor_presentacion: float
    decimales: int


INDICADORES: dict[str, DefinicionIndicador] = {
    "CRk": DefinicionIndicador(
        clave="CRk",
        nombre="Razón de concentración",
        unidad="%",
        campo_simulacion="crk",
        factor_presentacion=100.0,
        decimales=2,
    ),
    "IHH": DefinicionIndicador(
        clave="IHH",
        nombre="Índice Herfindahl-Hirschman",
        unidad="puntos",
        campo_simulacion="ihh_puntos",
        factor_presentacion=1.0,
        decimales=2,
    ),
    "ID": DefinicionIndicador(
        clave="ID",
        nombre="Índice de dominancia de García Alba",
        unidad="adimensional",
        campo_simulacion="id_garcia_alba",
        factor_presentacion=1.0,
        decimales=4,
    ),
    "IE": DefinicionIndicador(
        clave="IE",
        nombre="Índice de entropía",
        unidad="nats",
        campo_simulacion="entropia",
        factor_presentacion=1.0,
        decimales=4,
    ),
}


def obtener_definicion_indicador(clave: str) -> DefinicionIndicador:
    """Obtiene metadatos o informa claramente una clave inválida."""
    try:
        return INDICADORES[clave]
    except KeyError as error:
        opciones = ", ".join(INDICADORES)
        raise ValueError(f"Indicador desconocido. Use una de estas opciones: {opciones}.") from error


def nombre_eje_indicador(clave: str, k: int | None = None) -> str:
    """Construye el nombre del eje horizontal con su unidad."""
    definicion = obtener_definicion_indicador(clave)
    if clave == "CRk":
        sufijo = str(k) if k is not None else "k"
        return f"CR{sufijo} ({definicion.unidad})"
    return f"{clave} ({definicion.unidad})"


def obtener_valores_simulados(
    resultado: ResultadoSimulacion | ResultadoSimulacionCompacto, clave: str
) -> NDArray[np.float64]:
    """Extrae una serie simulada y la convierte solo para presentación."""
    definicion = obtener_definicion_indicador(clave)
    valores = getattr(resultado, definicion.campo_simulacion)
    return np.asarray(valores * definicion.factor_presentacion, dtype=np.float64)


def convertir_valor_presentacion(valor: float, clave: str) -> float:
    """Convierte un resultado interno a la unidad mostrada en la interfaz."""
    definicion = obtener_definicion_indicador(clave)
    return float(valor) * definicion.factor_presentacion


def formatear_valor_indicador(valor: float, clave: str) -> str:
    """Redondea únicamente el texto de presentación de un indicador."""
    definicion = obtener_definicion_indicador(clave)
    return f"{formatear_numero_indicador(valor, clave)} {definicion.unidad}"


def formatear_numero_indicador(valor: float, clave: str) -> str:
    """Formatea solo el número, sin alterar el valor usado en los cálculos."""
    definicion = obtener_definicion_indicador(clave)
    return f"{float(valor):.{definicion.decimales}f}"


def crear_grafico_cuotas(cuotas: ArrayLike) -> go.Figure:
    """Crea un gráfico de barras con cuotas internas expresadas en proporción."""
    vector_sin_validar = np.asarray(cuotas, dtype=np.float64)
    if vector_sin_validar.ndim != 1:
        raise ValueError("Las cuotas del gráfico deben formar un vector.")
    vector = validar_cuotas(vector_sin_validar, int(vector_sin_validar.size))
    porcentajes = vector * 100.0
    empresas = [f"Empresa {posicion}" for posicion in range(1, vector.size + 1)]

    figura = go.Figure(
        data=[
            go.Bar(
                x=empresas,
                y=porcentajes,
                name="Cuota de mercado",
                marker_color=COLOR_CUOTAS,
                hovertemplate="%{x}<br>Cuota: %{y:.4f} %<extra></extra>",
            )
        ]
    )
    figura.update_layout(
        title="Cuotas del caso particular",
        xaxis_title="Empresa",
        yaxis_title="Cuota de mercado (%)",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=25, t=65, b=50),
        showlegend=False,
    )
    figura.update_yaxes(rangemode="tozero", range=[0.0, max(100.0, float(np.max(porcentajes)) * 1.08)])
    return figura


def _validar_datos_histograma(
    valores_simulados: ArrayLike, valor_caso: float
) -> tuple[NDArray[np.float64], float]:
    try:
        valores = np.asarray(valores_simulados, dtype=np.float64)
        caso = float(valor_caso)
    except (TypeError, ValueError) as error:
        raise ValueError("El histograma requiere valores numéricos.") from error
    if valores.ndim != 1 or valores.size == 0:
        raise ValueError("El histograma requiere un vector no vacío.")
    if not np.all(np.isfinite(valores)) or not np.isfinite(caso):
        raise ValueError("El histograma requiere valores finitos.")
    return valores, caso


def crear_histograma_comparativo(
    valores_simulados: ArrayLike,
    valor_caso: float,
    indicador: str,
    k: int | None = None,
) -> go.Figure:
    """Compara la distribución simulada con una línea para el caso particular.

    El rango horizontal siempre incluye tanto las simulaciones como el caso.
    Cuando la distribución es constante se construye un intervalo artificial
    pequeño, evitando rangos degenerados en Plotly.
    """
    definicion = obtener_definicion_indicador(indicador)
    valores, caso = _validar_datos_histograma(valores_simulados, valor_caso)
    minimo_simulado = float(np.min(valores))
    maximo_simulado = float(np.max(valores))

    escala = max(abs(minimo_simulado), abs(maximo_simulado), abs(caso), 1.0)
    rango_casi_constante = (maximo_simulado - minimo_simulado) <= escala * 1e-12

    if rango_casi_constante:
        centro_simulado = float(np.mean(valores))
        semiancho = escala * 0.05
        bordes = np.linspace(
            centro_simulado - semiancho,
            centro_simulado + semiancho,
            12,
        )
    else:
        cantidad_barras = min(50, max(10, int(round(np.sqrt(valores.size)))))
        bordes = np.histogram_bin_edges(valores, bins=cantidad_barras)

    frecuencias, bordes = np.histogram(valores, bins=bordes)
    centros = (bordes[:-1] + bordes[1:]) / 2.0
    anchos = np.diff(bordes)
    frecuencia_maxima = max(int(np.max(frecuencias)), 1)
    etiqueta_caso = formatear_valor_indicador(caso, indicador)

    figura = go.Figure()
    figura.add_trace(
        go.Bar(
            x=centros,
            y=frecuencias,
            width=anchos,
            name="Simulaciones Monte Carlo",
            marker_color=COLOR_SIMULACIONES,
            opacity=0.82,
            hovertemplate=(
                f"Valor: %{{x:.{definicion.decimales}f}}<br>"
                "Frecuencia: %{y} mercados<extra></extra>"
            ),
        )
    )
    figura.add_trace(
        go.Scatter(
            x=[caso, caso],
            y=[0.0, frecuencia_maxima * 1.08],
            mode="lines",
            name="Caso particular",
            line=dict(color=COLOR_CASO, width=4, dash="dash"),
            hovertemplate=f"Caso particular: {etiqueta_caso}<extra></extra>",
        )
    )

    extremo_inferior = min(minimo_simulado, caso)
    extremo_superior = max(maximo_simulado, caso)
    amplitud = extremo_superior - extremo_inferior
    if amplitud == 0.0:
        amplitud = max(abs(extremo_inferior), 1.0) * 0.1
    margen = amplitud * 0.08

    figura.add_annotation(
        x=caso,
        y=frecuencia_maxima * 1.10,
        text=f"Caso: {etiqueta_caso}",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLOR_CASO,
        font=dict(color=COLOR_CASO),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLOR_CASO,
    )
    figura.update_layout(
        title=f"Distribución simulada: {definicion.nombre}",
        xaxis_title=nombre_eje_indicador(indicador, k),
        yaxis_title="Frecuencia (mercados simulados)",
        template="plotly_white",
        bargap=0.04,
        height=500,
        margin=dict(l=45, r=25, t=80, b=55),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
    )
    figura.update_xaxes(
        range=[extremo_inferior - margen, extremo_superior + margen]
    )
    figura.update_yaxes(range=[0.0, frecuencia_maxima * 1.24])
    return figura
