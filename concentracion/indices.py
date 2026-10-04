"""Validación de cuotas e indicadores de concentración de mercado.

Todas las funciones trabajan con cuotas expresadas como proporciones entre
0 y 1. Las entradas se validan, pero nunca se normalizan silenciosamente.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


TOLERANCIA_SUMA = 1e-10
N_MINIMO = 2
N_MAXIMO = 100


def _validar_entero(nombre: str, valor: object, minimo: int, maximo: int) -> int:
    """Valida un entero dentro de un intervalo cerrado."""
    if isinstance(valor, (bool, np.bool_)) or not isinstance(valor, (int, np.integer)):
        raise ValueError(f"{nombre} debe ser un número entero.")

    valor_entero = int(valor)
    if not minimo <= valor_entero <= maximo:
        raise ValueError(f"{nombre} debe estar entre {minimo} y {maximo}.")
    return valor_entero


def validar_cuotas(
    cuotas: Sequence[float] | NDArray[np.floating], n: int
) -> NDArray[np.float64]:
    """Valida y devuelve una copia ``float64`` de las cuotas.

    Se exige que ``n`` esté entre 2 y 100, que el vector sea unidimensional,
    contenga exactamente ``n`` cuotas finitas y no negativas, y que su suma
    sea cercana a 1 con tolerancia absoluta de ``1e-10``. Las cuotas cero
    cuentan como empresas al comprobar la longitud.
    """
    n_validado = _validar_entero("N", n, N_MINIMO, N_MAXIMO)

    try:
        vector = np.asarray(cuotas, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Las cuotas deben ser valores numéricos.") from error

    if vector.ndim != 1:
        raise ValueError("Las cuotas deben entregarse en un vector unidimensional.")
    if vector.size != n_validado:
        raise ValueError(
            f"La cantidad de cuotas debe ser exactamente N={n_validado}; "
            f"se recibieron {vector.size}."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("Todas las cuotas deben ser valores finitos.")
    if np.any(vector < 0.0):
        raise ValueError("Las cuotas no pueden ser negativas.")

    # math.fsum reduce el error de acumulación sin modificar ninguna cuota.
    suma = math.fsum(vector.tolist())
    if not math.isclose(suma, 1.0, rel_tol=0.0, abs_tol=TOLERANCIA_SUMA):
        raise ValueError(
            "Las cuotas deben sumar 1 dentro de una tolerancia absoluta de "
            f"{TOLERANCIA_SUMA:g}; la suma recibida es {suma:.17g}."
        )

    return vector.copy()


def calcular_crk(
    cuotas: Sequence[float] | NDArray[np.floating], n: int, k: int
) -> float:
    """Calcula la participación acumulada de las ``k`` empresas mayores."""
    n_validado = _validar_entero("N", n, N_MINIMO, N_MAXIMO)
    k_validado = _validar_entero("k", k, 1, n_validado)
    vector = validar_cuotas(cuotas, n_validado)
    mayores = np.sort(vector)[-k_validado:]
    return float(math.fsum(mayores.tolist()))


def calcular_ihh_decimal(
    cuotas: Sequence[float] | NDArray[np.floating], n: int
) -> float:
    """Calcula el IHH decimal: suma de las cuotas elevadas al cuadrado."""
    vector = validar_cuotas(cuotas, n)
    return float(np.dot(vector, vector))


def calcular_ihh_puntos(
    cuotas: Sequence[float] | NDArray[np.floating], n: int
) -> float:
    """Calcula el IHH en su escala habitual de 0 a 10.000 puntos."""
    return calcular_ihh_decimal(cuotas, n) * 10_000.0


def calcular_id_garcia_alba(
    cuotas: Sequence[float] | NDArray[np.floating], n: int
) -> float:
    """Calcula el índice de dominancia de García Alba.

    ID = sum(s_i**4) / (sum(s_i**2))**2
    """
    vector = validar_cuotas(cuotas, n)
    cuadrados = vector * vector
    ihh_decimal = float(np.sum(cuadrados))
    return float(np.dot(cuadrados, cuadrados) / (ihh_decimal * ihh_decimal))


def calcular_entropia(
    cuotas: Sequence[float] | NDArray[np.floating], n: int
) -> float:
    """Calcula la entropía de Shannon usando logaritmos naturales.

    Por continuidad, cada término correspondiente a una cuota cero aporta
    cero y no se evalúa ``log(0)``.
    """
    vector = validar_cuotas(cuotas, n)
    positivas = vector[vector > 0.0]
    entropia = -float(np.dot(positivas, np.log(positivas)))
    return 0.0 if entropia == 0.0 else entropia


def calcular_entropia_normalizada(
    cuotas: Sequence[float] | NDArray[np.floating], n: int
) -> float:
    """Calcula la entropía dividida por ln(N), incluyendo cuotas cero en N."""
    n_validado = _validar_entero("N", n, N_MINIMO, N_MAXIMO)
    entropia = calcular_entropia(cuotas, n_validado)
    return entropia / math.log(n_validado)
