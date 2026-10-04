"""Pruebas de clasificación y retroalimentación educativa."""

import numpy as np
import pytest

from concentracion.evaluacion import (
    ACLARACION_PERCENTIL,
    clasificar_ihh,
    evaluar_respuesta_ihh,
    intervalo_ihh,
)


@pytest.mark.parametrize(
    ("ihh", "esperada"),
    [
        (1_499.999999, "baja"),
        (1_500.0, "moderada"),
        (1_500.000001, "moderada"),
        (2_499.999999, "moderada"),
        (2_500.0, "alta"),
        (2_500.000001, "alta"),
    ],
)
def test_clasificacion_cubre_fronteras_sin_redondear(
    ihh: float, esperada: str
) -> None:
    assert clasificar_ihh(ihh) == esperada


def test_respuesta_correcta() -> None:
    retroalimentacion = evaluar_respuesta_ihh(
        "alta", [0.4, 0.3, 0.2, 0.1], 4, [1_000, 2_000, 3_000, 4_000]
    )
    assert retroalimentacion.es_correcta is True
    assert retroalimentacion.clasificacion_correcta == "alta"


def test_respuesta_incorrecta_informa_la_correcta() -> None:
    retroalimentacion = evaluar_respuesta_ihh(
        "baja", [0.4, 0.3, 0.2, 0.1], 4, [1_000, 2_000, 3_000, 4_000]
    )
    assert retroalimentacion.es_correcta is False
    assert retroalimentacion.clasificacion_correcta == "alta"


def test_ejemplo_40_30_20_10_es_alta() -> None:
    resultado = evaluar_respuesta_ihh(
        "alta", [0.4, 0.3, 0.2, 0.1], 4, [3_000]
    )
    assert resultado.ihh_puntos == pytest.approx(3_000.0)
    assert resultado.clasificacion_correcta == "alta"


def test_diez_cuotas_iguales_es_baja() -> None:
    cuotas = np.full(10, 0.1)
    resultado = evaluar_respuesta_ihh("baja", cuotas, 10, [1_000])
    assert resultado.ihh_puntos == pytest.approx(1_000.0)
    assert resultado.clasificacion_correcta == "baja"


def test_cinco_cuotas_iguales_es_moderada() -> None:
    cuotas = np.full(5, 0.2)
    resultado = evaluar_respuesta_ihh("moderada", cuotas, 5, [2_000])
    assert resultado.ihh_puntos == pytest.approx(2_000.0)
    assert resultado.clasificacion_correcta == "moderada"


def test_retroalimentacion_contiene_valor_intervalo_y_percentil() -> None:
    resultado = evaluar_respuesta_ihh(
        "alta", [0.4, 0.3, 0.2, 0.1], 4, [1_000, 2_000, 3_000, 4_000]
    )
    assert resultado.percentil == pytest.approx(75.0)
    assert "3000" in resultado.justificacion
    assert resultado.intervalo == "IHH ≥ 2.500 puntos"
    assert (
        resultado.explicacion_percentil
        == "El 75.00 % de los mercados simulados tiene un IHH menor o igual al de este caso."
    )
    assert resultado.aclaracion_percentil == ACLARACION_PERCENTIL
    assert "clasificación «alta»" in resultado.distincion_umbral_percentil
    assert "3000.00 puntos" in resultado.distincion_umbral_percentil
    assert "IHH ≥ 2.500 puntos" in resultado.distincion_umbral_percentil
    assert "percentil 75.00 %" in resultado.distincion_umbral_percentil
    assert "no modifica la clasificación" in resultado.distincion_umbral_percentil


def test_distincion_usa_el_percentil_real_sin_ejemplo_fijo() -> None:
    resultado = evaluar_respuesta_ihh(
        "baja", np.full(10, 0.1), 10, [900.0, 1_100.0, 1_200.0]
    )
    assert resultado.percentil == pytest.approx(100.0 / 3.0)
    assert "clasificación «baja»" in resultado.distincion_umbral_percentil
    assert "percentil 33.33 %" in resultado.distincion_umbral_percentil


@pytest.mark.parametrize(
    ("clasificacion", "intervalo"),
    [
        ("baja", "IHH < 1.500 puntos"),
        ("moderada", "1.500 ≤ IHH < 2.500 puntos"),
        ("alta", "IHH ≥ 2.500 puntos"),
    ],
)
def test_intervalos_documentados(clasificacion: str, intervalo: str) -> None:
    assert intervalo_ihh(clasificacion) == intervalo


@pytest.mark.parametrize("ihh", [float("nan"), float("inf"), -1.0])
def test_ihh_invalido_se_rechaza(ihh: float) -> None:
    with pytest.raises(ValueError):
        clasificar_ihh(ihh)


def test_respuesta_invalida_se_rechaza() -> None:
    with pytest.raises(ValueError, match="baja, moderada o alta"):
        evaluar_respuesta_ihh(
            "otra", [0.4, 0.3, 0.2, 0.1], 4, [1_000, 2_000, 3_000]
        )
