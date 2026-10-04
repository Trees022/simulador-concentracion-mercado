"""Pruebas de humo del flujo principal de Streamlit."""

import numpy as np
from pathlib import Path

from streamlit.testing.v1 import AppTest

from concentracion.indices import calcular_ihh_puntos
from concentracion.simulacion import calcular_percentil_empirico


def _abrir_app() -> AppTest:
    ruta_app = Path(__file__).resolve().parents[1] / "app.py"
    return AppTest.from_file(ruta_app, default_timeout=30).run()


def test_ejecutar_y_cambiar_indicador_no_regenera_la_muestra() -> None:
    app = _abrir_app()
    assert len(app.exception) == 0

    app.button(key="ejecutar_simulacion").click().run()
    muestra_original = app.session_state["resultado_simulacion"].ihh_puntos.copy()
    assert muestra_original.shape == (1_000,)
    assert len(app.get("plotly_chart")) == 2

    app.selectbox(key="indicador_seleccionado").set_value("ID").run()
    np.testing.assert_array_equal(
        muestra_original,
        app.session_state["resultado_simulacion"].ihh_puntos,
    )
    assert len(app.exception) == 0
    assert len(app.get("plotly_chart")) == 2


def test_indicadores_visibles_respetan_los_decimales_solicitados() -> None:
    app = _abrir_app()
    tabla = app.dataframe[1].value

    assert tabla["Valor"].tolist() == ["50.00", "2500.00", "0.2500", "1.3863"]
    assert any(
        caption.value
        == "IHH decimal: 0.25 · Entropía normalizada: 1.0000"
        for caption in app.caption
    )


def test_cambiar_n_bloquea_comparacion_y_adapta_el_caso() -> None:
    app = _abrir_app()
    app.button(key="ejecutar_simulacion").click().run()
    app.number_input(key="config_n").set_value(5).run()

    assert len(app.session_state["cuotas_caso_pct"]) == 5
    assert len(app.get("plotly_chart")) == 1
    assert any("comparación está bloqueada" in aviso.value for aviso in app.warning)
    assert len(app.exception) == 0


def test_suma_manual_invalida_no_se_normaliza() -> None:
    app = _abrir_app()
    app.session_state["cuotas_caso_pct"] = [60.0, 50.0, 0.0, 0.0]
    app.session_state["version_editor"] = 1
    app.run()

    assert any("suma ingresada es 110" in error.value for error in app.error)
    assert any("No se normalizarán" in error.value for error in app.error)
    assert len(app.get("plotly_chart")) == 0
    assert len(app.exception) == 0


def test_evaluador_usa_percentil_ihh_aunque_el_grafico_muestre_id() -> None:
    app = _abrir_app()
    app.button(key="ejecutar_simulacion").click().run()
    app.selectbox(key="indicador_seleccionado").set_value("ID").run()
    muestra_original = app.session_state["resultado_simulacion"].ihh_puntos.copy()

    app.radio(key="respuesta_ihh").set_value("alta").run()
    app.button(key="comprobar_respuesta_ihh").click().run()

    retroalimentacion = app.session_state["retroalimentacion_ihh"]
    cuotas = np.asarray(app.session_state["cuotas_caso_pct"]) / 100.0
    ihh_caso = calcular_ihh_puntos(cuotas, 4)
    esperado = calcular_percentil_empirico(
        app.session_state["resultado_simulacion"].ihh_puntos,
        ihh_caso,
    )
    assert retroalimentacion.percentil == esperado
    np.testing.assert_array_equal(
        muestra_original,
        app.session_state["resultado_simulacion"].ihh_puntos,
    )
    assert app.session_state["indicador_seleccionado"] == "ID"
    assert len(app.exception) == 0


def test_modificar_caso_reinicia_respuesta_y_retroalimentacion() -> None:
    app = _abrir_app()
    app.button(key="ejecutar_simulacion").click().run()
    app.radio(key="respuesta_ihh").set_value("alta").run()
    app.button(key="comprobar_respuesta_ihh").click().run()
    assert app.session_state["retroalimentacion_ihh"] is not None

    app.session_state["cuotas_caso_pct"] = [50.0, 20.0, 20.0, 10.0]
    app.session_state["version_editor"] += 1
    app.run()

    assert app.session_state["respuesta_ihh"] is None
    assert app.session_state["retroalimentacion_ihh"] is None
    assert len(app.exception) == 0


def test_evaluacion_bloqueada_sin_simulacion_compatible() -> None:
    app = _abrir_app()
    assert app.button(key="comprobar_respuesta_ihh").disabled is True

    app.button(key="ejecutar_simulacion").click().run()
    assert app.button(key="comprobar_respuesta_ihh").disabled is False
    app.number_input(key="config_n").set_value(5).run()
    assert app.button(key="comprobar_respuesta_ihh").disabled is True
    assert any("evaluación está bloqueada" in aviso.value for aviso in app.warning)
    assert len(app.exception) == 0


def test_nueva_simulacion_invalida_retroalimentacion_anterior() -> None:
    app = _abrir_app()
    app.button(key="ejecutar_simulacion").click().run()
    app.radio(key="respuesta_ihh").set_value("moderada").run()
    app.button(key="comprobar_respuesta_ihh").click().run()
    version_anterior = app.session_state["version_simulacion"]
    assert app.session_state["retroalimentacion_ihh"] is not None

    app.button(key="ejecutar_simulacion").click().run()
    assert app.session_state["version_simulacion"] == version_anterior + 1
    assert app.session_state["respuesta_ihh"] is None
    assert app.session_state["retroalimentacion_ihh"] is None
    assert len(app.exception) == 0
