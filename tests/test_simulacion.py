"""Pruebas del motor Monte Carlo y sus cálculos vectorizados."""

import math

import numpy as np
import pytest

from concentracion.indices import (
    calcular_crk,
    calcular_entropia,
    calcular_entropia_normalizada,
    calcular_id_garcia_alba,
    calcular_ihh_decimal,
    calcular_ihh_puntos,
)
from concentracion.simulacion import (
    ADVERTENCIA_RENDIMIENTO,
    ITERACIONES_PREDETERMINADAS,
    MAX_ITERACIONES,
    calcular_indicadores_vectorizados,
    calcular_percentil_empirico,
    compactar_resultado,
    generar_caso_aleatorio,
    simular_mercados,
)


def test_misma_configuracion_y_semilla_repite_resultados() -> None:
    primero = simular_mercados(n=6, k=3, iteraciones=40, semilla=12345)
    segundo = simular_mercados(n=6, k=3, iteraciones=40, semilla=12345)

    for nombre in primero.__dataclass_fields__:
        np.testing.assert_array_equal(
            getattr(primero, nombre), getattr(segundo, nombre)
        )


def test_todas_las_filas_cumplen_restricciones_de_cuotas() -> None:
    resultado = simular_mercados(n=12, k=4, iteraciones=500, semilla=81)
    assert np.all(np.isfinite(resultado.cuotas))
    assert np.all(resultado.cuotas >= 0.0)
    assert np.all(
        np.isclose(
            np.sum(resultado.cuotas, axis=1),
            1.0,
            rtol=0.0,
            atol=1e-10,
        )
    )


@pytest.mark.parametrize("n", [2, 100])
def test_funcionan_extremos_de_n(n: int) -> None:
    resultado = simular_mercados(n=n, k=n, iteraciones=25, semilla=9)
    assert resultado.cuotas.shape == (25, n)
    np.testing.assert_allclose(resultado.crk, 1.0, rtol=0.0, atol=1e-10)


def test_usa_mil_iteraciones_por_defecto() -> None:
    resultado = simular_mercados(n=4, k=2, semilla=7)
    assert ITERACIONES_PREDETERMINADAS == 1_000
    assert resultado.iteraciones == 1_000
    assert resultado.cuotas.shape == (1_000, 4)


def test_crk_es_uno_cuando_k_es_n() -> None:
    resultado = simular_mercados(n=7, k=7, iteraciones=80, semilla=12)
    np.testing.assert_allclose(resultado.crk, 1.0, rtol=0.0, atol=1e-10)


def test_calculos_vectorizados_coinciden_con_funciones_individuales() -> None:
    cuotas = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.4, 0.3, 0.2, 0.1],
            [1.0, 0.0, 0.0, 0.0],
            [0.6, 0.2, 0.15, 0.05],
        ]
    )
    resultado = calcular_indicadores_vectorizados(cuotas, n=4, k=2)

    for posicion, vector in enumerate(cuotas):
        assert resultado.crk[posicion] == pytest.approx(calcular_crk(vector, 4, 2))
        assert resultado.ihh_decimal[posicion] == pytest.approx(
            calcular_ihh_decimal(vector, 4)
        )
        assert resultado.ihh_puntos[posicion] == pytest.approx(
            calcular_ihh_puntos(vector, 4)
        )
        assert resultado.id_garcia_alba[posicion] == pytest.approx(
            calcular_id_garcia_alba(vector, 4)
        )
        assert resultado.entropia[posicion] == pytest.approx(
            calcular_entropia(vector, 4)
        )
        assert resultado.entropia_normalizada[posicion] == pytest.approx(
            calcular_entropia_normalizada(vector, 4)
        )


def test_percentil_sin_empate_adicional() -> None:
    assert calcular_percentil_empirico([1, 2, 3, 4], 2) == pytest.approx(50.0)


def test_percentil_incluye_todos_los_empates() -> None:
    assert calcular_percentil_empirico([1, 2, 2, 4], 2) == pytest.approx(75.0)


def test_generar_caso_no_altera_reproducibilidad_de_monte_carlo() -> None:
    antes = simular_mercados(n=5, k=2, iteraciones=30, semilla=2026)
    caso = generar_caso_aleatorio(n=5, semilla=88)
    despues = simular_mercados(n=5, k=2, iteraciones=30, semilla=2026)

    assert caso.shape == (5,)
    assert math.isclose(math.fsum(caso.tolist()), 1.0, rel_tol=0.0, abs_tol=1e-10)
    np.testing.assert_array_equal(antes.cuotas, despues.cuotas)


def test_casos_aleatorios_son_reproducibles_con_semilla_propia() -> None:
    primero = generar_caso_aleatorio(n=8, semilla=314)
    segundo = generar_caso_aleatorio(n=8, semilla=314)
    np.testing.assert_array_equal(primero, segundo)


def test_media_ihh_dirichlet_se_aproxima_a_valor_teorico() -> None:
    n = 5
    iteraciones = 20_000
    resultado = simular_mercados(n=n, k=2, iteraciones=iteraciones, semilla=1701)

    media_teorica = 2.0 / (n + 1)
    # Para Dirichlet(1,...,1):
    # E[IHH²] = 4(N+5) / ((N+1)(N+2)(N+3)).
    segundo_momento = 4.0 * (n + 5) / ((n + 1) * (n + 2) * (n + 3))
    varianza = segundo_momento - media_teorica**2
    error_estandar_monte_carlo = math.sqrt(varianza / iteraciones)

    # Seis errores estándar toleran la variación Monte Carlo sin pedir igualdad.
    assert abs(np.mean(resultado.ihh_decimal) - media_teorica) <= (
        6.0 * error_estandar_monte_carlo
    )


@pytest.mark.parametrize(
    ("argumentos", "mensaje"),
    [
        ({"n": 1, "k": 1}, "N debe estar entre 2 y 100"),
        ({"n": 101, "k": 1}, "N debe estar entre 2 y 100"),
        ({"n": 4.0, "k": 2}, "N debe ser un número entero"),
        ({"n": 4, "k": 0}, "k debe estar entre 1 y 4"),
        ({"n": 4, "k": 5}, "k debe estar entre 1 y 4"),
        ({"n": 4, "k": 2.0}, "k debe ser un número entero"),
        ({"n": 4, "k": 2, "iteraciones": 0}, "iteraciones debe estar entre"),
        (
            {"n": 4, "k": 2, "iteraciones": MAX_ITERACIONES + 1},
            "iteraciones debe estar entre",
        ),
        ({"n": 4, "k": 2, "iteraciones": 10.0}, "iteraciones debe ser"),
        ({"n": 4, "k": 2, "semilla": -1}, "semilla no puede ser negativa"),
        ({"n": 4, "k": 2, "semilla": 1.5}, "semilla debe ser"),
    ],
)
def test_simulacion_rechaza_entradas_invalidas(
    argumentos: dict[str, object], mensaje: str
) -> None:
    with pytest.raises(ValueError, match=mensaje):
        simular_mercados(**argumentos)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("valores", "caso", "mensaje"),
    [
        ([], 1.0, "al menos un valor"),
        ([[1.0, 2.0]], 1.0, "formar un vector"),
        ([1.0, np.nan], 1.0, "deben ser finitos"),
        ([1.0, 2.0], np.inf, "caso debe ser un número finito"),
        ([1.0, 2.0], True, "caso debe ser un número finito"),
        (["a", "b"], 1.0, "valores simulados deben ser numéricos"),
    ],
)
def test_percentil_rechaza_entradas_invalidas(
    valores: object, caso: object, mensaje: str
) -> None:
    with pytest.raises(ValueError, match=mensaje):
        calcular_percentil_empirico(valores, caso)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("cuotas", "n", "k", "mensaje"),
    [
        ([[0.5, 0.5, 0.0]], 2, 1, "exactamente N=2"),
        ([[0.5, 0.5], [0.8, -0.1]], 2, 1, "no pueden ser negativas"),
        ([[0.5, 0.5], [np.nan, 0.0]], 2, 1, "valores finitos"),
        ([[0.5, 0.6]], 2, 1, "debe sumar 1"),
        ([], 2, 1, "matriz bidimensional"),
    ],
)
def test_calculo_vectorizado_rechaza_matrices_invalidas(
    cuotas: object, n: int, k: int, mensaje: str
) -> None:
    with pytest.raises(ValueError, match=mensaje):
        calcular_indicadores_vectorizados(cuotas, n, k)


def test_texto_de_advertencia_esta_disponible() -> None:
    assert "tiempo de respuesta" in ADVERTENCIA_RENDIMIENTO
    assert "memoria" in ADVERTENCIA_RENDIMIENTO
    assert "procesamiento" in ADVERTENCIA_RENDIMIENTO


def test_compactar_resultado_no_copia_ni_conserva_matriz_de_cuotas() -> None:
    completo = simular_mercados(n=6, k=3, iteraciones=20, semilla=77)
    compacto = compactar_resultado(completo)

    assert compacto.n == 6
    assert compacto.iteraciones == 20
    assert not hasattr(compacto, "cuotas")
    assert np.shares_memory(compacto.crk, completo.crk)
    assert np.shares_memory(compacto.ihh_puntos, completo.ihh_puntos)
    assert np.shares_memory(compacto.id_garcia_alba, completo.id_garcia_alba)
    assert np.shares_memory(compacto.entropia, completo.entropia)
