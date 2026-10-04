"""Pruebas del módulo matemático de concentración."""

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
    validar_cuotas,
)


CUOTAS_IGUALES = [0.25, 0.25, 0.25, 0.25]
CUOTAS_DESIGUALES = [0.4, 0.3, 0.2, 0.1]


def test_cuatro_empresas_iguales() -> None:
    assert calcular_crk(CUOTAS_IGUALES, n=4, k=2) == pytest.approx(0.5)
    assert calcular_ihh_decimal(CUOTAS_IGUALES, n=4) == pytest.approx(0.25)
    assert calcular_ihh_puntos(CUOTAS_IGUALES, n=4) == pytest.approx(2500.0)
    assert calcular_id_garcia_alba(CUOTAS_IGUALES, n=4) == pytest.approx(0.25)
    assert calcular_entropia(CUOTAS_IGUALES, n=4) == pytest.approx(math.log(4))
    assert calcular_entropia_normalizada(CUOTAS_IGUALES, n=4) == pytest.approx(1.0)


def test_cuatro_empresas_desiguales() -> None:
    assert calcular_crk(CUOTAS_DESIGUALES, n=4, k=2) == pytest.approx(0.7)
    assert calcular_ihh_decimal(CUOTAS_DESIGUALES, n=4) == pytest.approx(0.3)
    assert calcular_ihh_puntos(CUOTAS_DESIGUALES, n=4) == pytest.approx(3000.0)
    assert calcular_id_garcia_alba(CUOTAS_DESIGUALES, n=4) == pytest.approx(
        0.3933333333
    )
    assert calcular_entropia(CUOTAS_DESIGUALES, n=4) == pytest.approx(
        1.2798542258
    )


def test_monopolio_con_cuotas_cero() -> None:
    cuotas = [1.0, 0.0, 0.0, 0.0]
    assert calcular_ihh_puntos(cuotas, n=4) == pytest.approx(10_000.0)
    assert calcular_id_garcia_alba(cuotas, n=4) == pytest.approx(1.0)
    assert calcular_entropia(cuotas, n=4) == pytest.approx(0.0)
    assert calcular_entropia_normalizada(cuotas, n=4) == pytest.approx(0.0)


@pytest.mark.parametrize("cuotas", [CUOTAS_IGUALES, CUOTAS_DESIGUALES])
def test_crk_es_uno_cuando_k_es_n(cuotas: list[float]) -> None:
    assert calcular_crk(cuotas, n=4, k=4) == pytest.approx(1.0)


def test_indicadores_no_dependen_del_orden() -> None:
    originales = np.array(CUOTAS_DESIGUALES)
    reordenadas = originales[[2, 0, 3, 1]]

    funciones = [
        calcular_ihh_decimal,
        calcular_ihh_puntos,
        calcular_id_garcia_alba,
        calcular_entropia,
        calcular_entropia_normalizada,
    ]
    for funcion in funciones:
        assert funcion(originales, 4) == pytest.approx(funcion(reordenadas, 4))
    assert calcular_crk(originales, 4, 2) == pytest.approx(
        calcular_crk(reordenadas, 4, 2)
    )


@pytest.mark.parametrize(
    ("cuotas", "n", "mensaje"),
    [
        ([0.5, 0.5], 1, "N debe estar entre 2 y 100"),
        ([0.5, 0.5], 101, "N debe estar entre 2 y 100"),
        ([0.5, 0.5], 2.0, "N debe ser un número entero"),
        ([0.5, 0.5], True, "N debe ser un número entero"),
        ([0.5, 0.3, 0.2], 2, "cantidad de cuotas debe ser exactamente N=2"),
        ([[0.5, 0.5]], 2, "vector unidimensional"),
        ([0.8, -0.2, 0.4], 3, "cuotas no pueden ser negativas"),
        ([0.5, math.inf], 2, "cuotas deben ser valores finitos"),
        ([0.5, math.nan], 2, "cuotas deben ser valores finitos"),
        ([0.6, 0.5], 2, "cuotas deben sumar 1"),
        (["no", "numérico"], 2, "cuotas deben ser valores numéricos"),
    ],
)
def test_entradas_invalidas_producen_mensajes_claros(
    cuotas: object, n: object, mensaje: str
) -> None:
    with pytest.raises(ValueError, match=mensaje):
        validar_cuotas(cuotas, n)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("k", "mensaje"),
    [
        (0, "k debe estar entre 1 y 4"),
        (5, "k debe estar entre 1 y 4"),
        (2.0, "k debe ser un número entero"),
        (True, "k debe ser un número entero"),
    ],
)
def test_k_invalido_produce_mensaje_claro(k: object, mensaje: str) -> None:
    with pytest.raises(ValueError, match=mensaje):
        calcular_crk(CUOTAS_IGUALES, n=4, k=k)  # type: ignore[arg-type]


def test_suma_dentro_de_tolerancia_se_acepta_sin_normalizar() -> None:
    cuotas = [0.5, 0.5 + 5e-11]
    validadas = validar_cuotas(cuotas, n=2)
    assert validadas[1] == cuotas[1]
    assert math.fsum(validadas.tolist()) != 1.0


def test_suma_fuera_de_tolerancia_se_rechaza() -> None:
    with pytest.raises(ValueError, match="tolerancia absoluta"):
        validar_cuotas([0.5, 0.5 + 2e-10], n=2)


def test_extremo_n_igual_a_dos() -> None:
    cuotas = [0.6, 0.4]
    assert validar_cuotas(cuotas, n=2).size == 2
    assert calcular_crk(cuotas, n=2, k=2) == pytest.approx(1.0)
    assert calcular_entropia_normalizada(cuotas, n=2) <= 1.0


def test_extremo_n_igual_a_cien() -> None:
    cuotas = np.full(100, 0.01)
    assert validar_cuotas(cuotas, n=100).size == 100
    assert calcular_crk(cuotas, n=100, k=100) == pytest.approx(1.0)
    assert calcular_ihh_puntos(cuotas, n=100) == pytest.approx(100.0)
    assert calcular_entropia_normalizada(cuotas, n=100) == pytest.approx(1.0)


def test_entropia_normalizada_usa_n_incluyendo_ceros() -> None:
    cuotas = [0.5, 0.5, 0.0, 0.0]
    esperado = math.log(2) / math.log(4)
    assert calcular_entropia_normalizada(cuotas, n=4) == pytest.approx(esperado)
