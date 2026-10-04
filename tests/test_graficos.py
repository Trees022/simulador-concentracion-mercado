"""Pruebas estructurales de las figuras Plotly."""

import numpy as np
import pytest

from concentracion.graficos import (
    crear_grafico_cuotas,
    crear_histograma_comparativo,
    nombre_eje_indicador,
    obtener_valores_simulados,
)
from concentracion.simulacion import simular_mercados


def test_grafico_cuotas_convierte_proporciones_a_porcentajes() -> None:
    figura = crear_grafico_cuotas([0.4, 0.3, 0.2, 0.1])
    np.testing.assert_allclose(figura.data[0].y, [40.0, 30.0, 20.0, 10.0])
    assert figura.layout.yaxis.title.text == "Cuota de mercado (%)"


@pytest.mark.parametrize(
    ("indicador", "texto_eje"),
    [
        ("CRk", "CR2 (%)"),
        ("IHH", "IHH (puntos)"),
        ("ID", "ID (adimensional)"),
        ("IE", "IE (nats)"),
    ],
)
def test_los_cuatro_histogramas_tienen_ejes_y_leyenda(
    indicador: str, texto_eje: str
) -> None:
    figura = crear_histograma_comparativo(
        [1.0, 2.0, 3.0, 4.0], 2.5, indicador, k=2
    )
    assert figura.layout.xaxis.title.text == texto_eje
    assert figura.layout.yaxis.title.text == "Frecuencia (mercados simulados)"
    assert figura.data[0].name == "Simulaciones Monte Carlo"
    assert figura.data[1].name == "Caso particular"


def test_histograma_amplia_eje_para_caso_fuera_de_rango() -> None:
    figura = crear_histograma_comparativo([1.0, 2.0, 3.0], 10.0, "IHH")
    assert figura.layout.xaxis.range[0] < 1.0
    assert figura.layout.xaxis.range[1] > 10.0


def test_histograma_admite_distribucion_constante_crk() -> None:
    resultado = simular_mercados(n=4, k=4, iteraciones=100, semilla=91)
    valores = obtener_valores_simulados(resultado, "CRk")
    figura = crear_histograma_comparativo(valores, 100.0, "CRk", k=4)
    assert len(figura.data) == 2
    assert sum(figura.data[0].y) == 100
    assert figura.layout.xaxis.range[0] < 100.0 < figura.layout.xaxis.range[1]


def test_extrae_cada_indicador_en_su_unidad_de_presentacion() -> None:
    resultado = simular_mercados(n=4, k=2, iteraciones=10, semilla=5)
    np.testing.assert_allclose(obtener_valores_simulados(resultado, "CRk"), resultado.crk * 100)
    np.testing.assert_allclose(obtener_valores_simulados(resultado, "IHH"), resultado.ihh_puntos)
    np.testing.assert_allclose(obtener_valores_simulados(resultado, "ID"), resultado.id_garcia_alba)
    np.testing.assert_allclose(obtener_valores_simulados(resultado, "IE"), resultado.entropia)


def test_indicador_desconocido_se_rechaza() -> None:
    with pytest.raises(ValueError, match="Indicador desconocido"):
        nombre_eje_indicador("otro")
