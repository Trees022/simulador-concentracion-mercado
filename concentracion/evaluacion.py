"""Reglas matemáticas para el evaluador educativo de concentración."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .indices import calcular_ihh_puntos, validar_cuotas
from .simulacion import calcular_percentil_empirico


FUENTE_FNE_URL = (
    "https://www.fne.gob.cl/wp-content/uploads/2022/05/"
    "20220531.-Guia-para-el-Analisis-de-Operaciones-de-Concentracion-"
    "Horizontales-version-final-en-castellano.pdf"
)
FUENTE_FNE_TITULO = (
    "Guía para el Análisis de Operaciones de Concentración Horizontales, "
    "FNE, mayo de 2022"
)

PREGUNTA_PRINCIPAL = (
    "Según el criterio de IHH utilizado en esta aplicación, ¿cómo "
    "clasificarías la concentración del caso particular?"
)
CLASIFICACIONES = ("baja", "moderada", "alta")

UMBRAL_MODERADA = 1_500.0
UMBRAL_ALTA = 2_500.0

ACLARACION_CONVENCION = (
    "La asignación de los valores exactamente iguales a 1.500 y 2.500 es una "
    "convención operativa y didáctica de este simulador."
)
ACLARACION_COMPETENCIA = (
    "La concentración por sí sola no demuestra una conducta anticompetitiva "
    "ni reemplaza el análisis de una operación de concentración."
)
ACLARACION_PERCENTIL = (
    "El percentil depende de N y del modelo Dirichlet(1,…,1) utilizado; no es "
    "un umbral normativo."
)


@dataclass(frozen=True)
class RetroalimentacionIHH:
    """Resultado completo e inmutable de una respuesta evaluada."""

    es_correcta: bool
    respuesta_usuario: str
    clasificacion_correcta: str
    ihh_puntos: float
    intervalo: str
    justificacion: str
    percentil: float
    explicacion_percentil: str
    aclaracion_percentil: str = ACLARACION_PERCENTIL


def _validar_ihh(ihh_puntos: float) -> float:
    try:
        valor = float(ihh_puntos)
    except (TypeError, ValueError) as error:
        raise ValueError("El IHH debe ser un número finito.") from error
    if not math.isfinite(valor):
        raise ValueError("El IHH debe ser un número finito.")
    if valor < 0.0:
        raise ValueError("El IHH no puede ser negativo.")
    return valor


def clasificar_ihh(ihh_puntos: float) -> str:
    """Clasifica el IHH sin redondearlo antes de comparar los umbrales."""
    valor = _validar_ihh(ihh_puntos)
    if valor < UMBRAL_MODERADA:
        return "baja"
    if valor < UMBRAL_ALTA:
        return "moderada"
    return "alta"


def intervalo_ihh(clasificacion: str) -> str:
    """Devuelve el intervalo didáctico correspondiente a una clasificación."""
    intervalos = {
        "baja": "IHH < 1.500 puntos",
        "moderada": "1.500 ≤ IHH < 2.500 puntos",
        "alta": "IHH ≥ 2.500 puntos",
    }
    try:
        return intervalos[clasificacion]
    except KeyError as error:
        raise ValueError("La clasificación debe ser baja, moderada o alta.") from error


def _formatear_numero(valor: float, decimales: int = 6) -> str:
    """Formatea para explicar, sin modificar el valor usado para decidir."""
    return f"{valor:.{decimales}f}".rstrip("0").rstrip(".")


def _crear_justificacion(cuotas: np.ndarray, ihh_puntos: float) -> str:
    porcentajes = cuotas * 100.0
    if porcentajes.size <= 12:
        terminos = " + ".join(
            f"{_formatear_numero(float(cuota))}²" for cuota in porcentajes
        )
        return (
            f"Con cuotas porcentuales {_formatear_lista(porcentajes)}, el IHH "
            f"es {terminos} = {_formatear_numero(ihh_puntos)} puntos."
        )

    cinco_mayores = np.sort(porcentajes)[-5:][::-1]
    return (
        f"El caso contiene {porcentajes.size} cuotas. Sus cinco mayores son "
        f"{_formatear_lista(cinco_mayores)}; al sumar los cuadrados de las "
        f"{porcentajes.size} cuotas porcentuales se obtiene un IHH de "
        f"{_formatear_numero(ihh_puntos)} puntos."
    )


def _formatear_lista(valores: np.ndarray) -> str:
    return "[" + ", ".join(_formatear_numero(float(v)) for v in valores) + "] %"


def evaluar_respuesta_ihh(
    respuesta: str,
    cuotas: ArrayLike,
    n: int,
    ihh_simulados: ArrayLike,
) -> RetroalimentacionIHH:
    """Evalúa una respuesta y calcula siempre el percentil sobre IHH simulado."""
    if not isinstance(respuesta, str):
        raise ValueError("La respuesta debe ser baja, moderada o alta.")
    respuesta_normalizada = respuesta.strip().lower()
    if respuesta_normalizada not in CLASIFICACIONES:
        raise ValueError("La respuesta debe ser baja, moderada o alta.")

    cuotas_validadas = validar_cuotas(cuotas, n)
    ihh = calcular_ihh_puntos(cuotas_validadas, n)
    correcta = clasificar_ihh(ihh)
    percentil = calcular_percentil_empirico(ihh_simulados, ihh)
    explicacion = (
        f"El {percentil:.2f} % de los mercados simulados tiene un IHH menor o "
        "igual al de este caso."
    )

    return RetroalimentacionIHH(
        es_correcta=respuesta_normalizada == correcta,
        respuesta_usuario=respuesta_normalizada,
        clasificacion_correcta=correcta,
        ihh_puntos=ihh,
        intervalo=intervalo_ihh(correcta),
        justificacion=_crear_justificacion(cuotas_validadas, ihh),
        percentil=percentil,
        explicacion_percentil=explicacion,
    )


def huella_cuotas(cuotas: ArrayLike, n: int) -> str:
    """Crea una huella estable para invalidar respuestas si cambia el caso."""
    vector = validar_cuotas(cuotas, n)
    contenido = n.to_bytes(2, byteorder="little", signed=False) + vector.tobytes()
    return hashlib.sha256(contenido).hexdigest()
