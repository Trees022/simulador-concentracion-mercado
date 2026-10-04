"""Pruebas estructurales de las figuras Plotly."""

import numpy as np
import pytest

from concentracion.graficos import (
    crear_grafico_cuotas,
    crear_histograma_comparativo,
    formatear_numero_indicador,
    formatear_valor_indicador,
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


@pytest.mark.parametrize(
    ("indicador", "valor", "numero", "con_unidad"),
    [
        ("CRk", 70.126, "70.13", "70.13 %"),
        ("IHH", 2_345.678, "2345.68", "2345.68 puntos"),
        ("ID", 0.3933333333, "0.3933", "0.3933 adimensional"),
        ("IE", 1.2798542258, "1.2799", "1.2799 nats"),
    ],
)
def test_formato_visible_respeta_decimales_por_indicador(
    indicador: str, valor: float, numero: str, con_unidad: str
) -> None:
    assert formatear_numero_indicador(valor, indicador) == numero
    assert formatear_valor_indicador(valor, indicador) == con_unidad


@pytest.mark.parametrize(
    ("indicador", "formato"),
    [("CRk", ".2f"), ("IHH", ".2f"), ("ID", ".4f"), ("IE", ".4f")],
)
def test_hover_del_histograma_usa_decimales_del_indicador(
    indicador: str, formato: str
) -> None:
    figura = crear_histograma_comparativo([1.0, 2.0, 3.0], 2.0, indicador, k=2)
    assert formato in figura.data[0].hovertemplate
