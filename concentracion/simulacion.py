"""Motor vectorizado para simulaciones Monte Carlo de cuotas de mercado.

La simulación y el caso particular están deliberadamente separados. Un
``ResultadoSimulacion`` depende solo de su configuración y semilla; calcular
otro caso o su percentil no consume ni modifica el generador usado por Monte
Carlo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .indices import (
    N_MAXIMO,
    N_MINIMO,
    TOLERANCIA_SUMA,
    _validar_entero,
    validar_cuotas,
)


ITERACIONES_PREDETERMINADAS = 1_000
MAX_ITERACIONES = 100_000

ADVERTENCIA_RENDIMIENTO = (
    "Aumentar las iteraciones incrementa el tiempo de respuesta y el consumo "
    "de memoria y procesamiento."
)

NOTA_PERCENTIL_ENTROPIA = (
    "En entropía, un percentil alto significa mayor entropía, no mayor "
    "concentración."
)


@dataclass(frozen=True)
class ResultadoSimulacion:
    """Cuotas e indicadores calculados para los mismos mercados simulados."""

    cuotas: NDArray[np.float64]
    crk: NDArray[np.float64]
    ihh_decimal: NDArray[np.float64]
    ihh_puntos: NDArray[np.float64]
    id_garcia_alba: NDArray[np.float64]
    entropia: NDArray[np.float64]
    entropia_normalizada: NDArray[np.float64]

    @property
    def iteraciones(self) -> int:
        """Cantidad de mercados contenidos en el resultado."""
        return int(self.cuotas.shape[0])

    @property
    def n(self) -> int:
        """Cantidad de empresas de cada mercado."""
        return int(self.cuotas.shape[1])


@dataclass(frozen=True)
class ResultadoSimulacionCompacto:
    """Series mínimas que la interfaz necesita conservar en sesión.

    No contiene la matriz de cuotas ni indicadores auxiliares que la interfaz
    no presenta. Sus arreglos son referencias a los ya calculados, por lo que
    compactar no crea copias grandes.
    """

    n: int
    iteraciones: int
    crk: NDArray[np.float64]
    ihh_puntos: NDArray[np.float64]
    id_garcia_alba: NDArray[np.float64]
    entropia: NDArray[np.float64]


def compactar_resultado(
    resultado: ResultadoSimulacion,
) -> ResultadoSimulacionCompacto:
    """Descarta de la sesión datos que no se necesitan para la interfaz."""
    if not isinstance(resultado, ResultadoSimulacion):
        raise TypeError("Se esperaba un ResultadoSimulacion completo.")
    return ResultadoSimulacionCompacto(
        n=resultado.n,
        iteraciones=resultado.iteraciones,
        crk=resultado.crk,
        ihh_puntos=resultado.ihh_puntos,
        id_garcia_alba=resultado.id_garcia_alba,
        entropia=resultado.entropia,
    )


def _validar_semilla(semilla: int | None) -> int | None:
    """Valida una semilla entera no negativa o ``None``."""
    if semilla is None:
        return None
    if isinstance(semilla, (bool, np.bool_)) or not isinstance(
        semilla, (int, np.integer)
    ):
        raise ValueError("La semilla debe ser un número entero o None.")
    if int(semilla) < 0:
        raise ValueError("La semilla no puede ser negativa.")
    return int(semilla)


def _validar_matriz_cuotas(cuotas: ArrayLike, n: int) -> NDArray[np.float64]:
    """Valida simultáneamente todas las filas de una matriz de cuotas."""
    n_validado = _validar_entero("N", n, N_MINIMO, N_MAXIMO)
    try:
        matriz = np.asarray(cuotas, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Las cuotas simuladas deben ser valores numéricos.") from error

    if matriz.ndim != 2:
        raise ValueError("Las cuotas simuladas deben formar una matriz bidimensional.")
    if matriz.shape[0] == 0:
        raise ValueError("La matriz debe contener al menos un mercado.")
    if matriz.shape[1] != n_validado:
        raise ValueError(
            f"Cada mercado debe contener exactamente N={n_validado} cuotas; "
            f"se recibieron {matriz.shape[1]}."
        )
    if not np.all(np.isfinite(matriz)):
        raise ValueError("Todas las cuotas simuladas deben ser valores finitos.")
    if np.any(matriz < 0.0):
        raise ValueError("Las cuotas simuladas no pueden ser negativas.")

    sumas = np.sum(matriz, axis=1)
    filas_validas = np.isclose(
        sumas, 1.0, rtol=0.0, atol=TOLERANCIA_SUMA
    )
    if not np.all(filas_validas):
        desviacion_maxima = float(np.max(np.abs(sumas - 1.0)))
        raise ValueError(
            "Cada mercado debe sumar 1 dentro de una tolerancia absoluta de "
            f"{TOLERANCIA_SUMA:g}; la desviación máxima es "
            f"{desviacion_maxima:.17g}."
        )
    return matriz


def calcular_indicadores_vectorizados(
    cuotas: ArrayLike, n: int, k: int
) -> ResultadoSimulacion:
    """Calcula todos los indicadores por filas, sin redondear las cuotas."""
    n_validado = _validar_entero("N", n, N_MINIMO, N_MAXIMO)
    k_validado = _validar_entero("k", k, 1, n_validado)
    matriz = _validar_matriz_cuotas(cuotas, n_validado)

    # np.partition evita ordenar completamente cada mercado para obtener CRk.
    particion = np.partition(matriz, n_validado - k_validado, axis=1)
    crk = np.sum(particion[:, -k_validado:], axis=1)
    del particion

    cuadrados = matriz * matriz
    ihh_decimal = np.sum(cuadrados, axis=1)
    ihh_puntos = ihh_decimal * 10_000.0
    cuartas = np.einsum("ij,ij->i", cuadrados, cuadrados)
    id_garcia_alba = cuartas / (ihh_decimal * ihh_decimal)
    del cuadrados, cuartas

    # Los ceros aportan cero por continuidad. El argumento where evita log(0).
    log_cuotas = np.zeros_like(matriz)
    np.log(matriz, out=log_cuotas, where=matriz > 0.0)
    entropia = -np.einsum("ij,ij->i", matriz, log_cuotas)
    entropia[entropia == 0.0] = 0.0
    entropia_normalizada = entropia / math.log(n_validado)

    return ResultadoSimulacion(
        cuotas=matriz,
        crk=crk,
        ihh_decimal=ihh_decimal,
        ihh_puntos=ihh_puntos,
        id_garcia_alba=id_garcia_alba,
        entropia=entropia,
        entropia_normalizada=entropia_normalizada,
    )


def simular_mercados(
    n: int,
    k: int,
    iteraciones: int = ITERACIONES_PREDETERMINADAS,
    semilla: int | None = None,
) -> ResultadoSimulacion:
    """Genera mercados Dirichlet(1, ..., 1) y calcula sus indicadores.

    Cada llamada crea su propio ``numpy.random.Generator``. Por ello, el
    resultado es reproducible con la misma configuración y semilla, sin
    depender de otros casos aleatorios generados por la aplicación.
    """
    n_validado = _validar_entero("N", n, N_MINIMO, N_MAXIMO)
    k_validado = _validar_entero("k", k, 1, n_validado)
    iteraciones_validadas = _validar_entero(
        "La cantidad de iteraciones", iteraciones, 1, MAX_ITERACIONES
    )
    semilla_validada = _validar_semilla(semilla)

    generador = np.random.default_rng(semilla_validada)
    parametros = np.ones(n_validado, dtype=np.float64)
    cuotas = generador.dirichlet(parametros, size=iteraciones_validadas)
    return calcular_indicadores_vectorizados(cuotas, n_validado, k_validado)


def generar_caso_aleatorio(
    n: int, semilla: int | None = None
) -> NDArray[np.float64]:
    """Genera un caso Dirichlet independiente del generador de Monte Carlo."""
    n_validado = _validar_entero("N", n, N_MINIMO, N_MAXIMO)
    semilla_validada = _validar_semilla(semilla)
    generador_caso = np.random.default_rng(semilla_validada)
    cuotas = generador_caso.dirichlet(np.ones(n_validado, dtype=np.float64))
    return validar_cuotas(cuotas, n_validado)


def calcular_percentil_empirico(
    valores_simulados: ArrayLike, valor_caso: float
) -> float:
    """Calcula el porcentaje de simulaciones menores o iguales al caso.

    Los empates se incluyen en el numerador. Por ejemplo, si el valor del caso
    coincide con dos observaciones simuladas, ambas cuentan como menores o
    iguales. Para la entropía, un percentil alto indica mayor entropía y no
    mayor concentración.
    """
    try:
        valores = np.asarray(valores_simulados, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Los valores simulados deben ser numéricos.") from error

    if valores.ndim != 1:
        raise ValueError("Los valores simulados deben formar un vector.")
    if valores.size == 0:
        raise ValueError("Debe existir al menos un valor simulado.")
    if not np.all(np.isfinite(valores)):
        raise ValueError("Todos los valores simulados deben ser finitos.")
    if isinstance(valor_caso, (bool, np.bool_)):
        raise ValueError("El valor del caso debe ser un número finito.")
    try:
        caso = float(valor_caso)
    except (TypeError, ValueError) as error:
        raise ValueError("El valor del caso debe ser un número finito.") from error
    if not math.isfinite(caso):
        raise ValueError("El valor del caso debe ser un número finito.")

    menores_o_iguales = np.count_nonzero(valores <= caso)
    return 100.0 * float(menores_o_iguales) / float(valores.size)
