import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from extract.key_value_extractor import (
    KeyValueExtractor,
    MAX_NORMALIZED_DISTANCE,
    NORMALIZED_BBOX_MAX,
    estimate_line_tolerance,
    sort_key,
)


def make_extractor(**overrides):
    """Instancia KeyValueExtractor sin cargar el modelo ni el LayoutExtractor,
    solo configurando los parametros de la heuristica."""
    extractor = object.__new__(KeyValueExtractor)
    extractor._configure_extraction_parameters(
        overrides.get("field_key_prefix"),
        overrides.get("field_value_prefix"),
        overrides.get("header_prefix"),
        overrides.get("item_prefix"),
        overrides.get("row_tolerance"),
        overrides.get("edge_tolerance"),
        overrides.get("column_tolerance"),
        overrides.get("ambiguity_k"),
    )
    return extractor


# ---------------------------------------------------------------------------
# PAGE_MAX_DISTANCE: ahora se calcula, ya no viene de .env
# ---------------------------------------------------------------------------

def test_normalized_distance_is_diagonal_of_0_1000_space():
    assert NORMALIZED_BBOX_MAX == 1000
    assert MAX_NORMALIZED_DISTANCE == pytest.approx(math.sqrt(2) * 1000)


def test_max_normalized_distance_covers_any_real_bbox_distance():
    # Ninguna distancia entre dos puntos dentro de [0,1000]x[0,1000]
    # puede superar la diagonal calculada.
    corners = [(0, 0), (1000, 0), (0, 1000), (1000, 1000)]
    for (ax, ay) in corners:
        for (bx, by) in corners:
            distance = math.hypot(ax - bx, ay - by)
            assert distance <= MAX_NORMALIZED_DISTANCE


def test_extractor_uses_computed_constant_not_env():
    extractor = make_extractor()
    assert extractor.PAGE_MAX_DISTANCE == MAX_NORMALIZED_DISTANCE
    assert KeyValueExtractor.PAGE_MAX_DISTANCE == MAX_NORMALIZED_DISTANCE


def test_configure_extraction_parameters_no_longer_accepts_page_max_distance():
    import inspect

    params = inspect.signature(KeyValueExtractor._configure_extraction_parameters).parameters
    assert "page_max_distance" not in params
    params_init = inspect.signature(KeyValueExtractor.__init__).parameters
    assert "page_max_distance" not in params_init


# ---------------------------------------------------------------------------
# _score_value_candidate: el tier siempre debe dominar sobre la distancia
# ---------------------------------------------------------------------------

def test_higher_tier_wins_at_equal_distance():
    # A igual distancia, el tier mas alto (misma fila = 1.2) siempre gana
    # sobre el tier mas bajo (alineado debajo = 1). Esto SI esta garantizado
    # matematicamente: tier_a * K - d > tier_b * K - d  <=>  tier_a > tier_b.
    extractor = make_extractor()
    key_bbox = (0, 0, 10, 10)
    same_row_value = (20, 0, 30, 10)
    below_value = (0, 20, 10, 30)

    score_same_row = extractor._score_value_candidate(key_bbox, same_row_value)
    score_below = extractor._score_value_candidate(key_bbox, below_value)
    assert score_same_row is not None and score_below is not None
    assert score_same_row > score_below


def test_tier_bonus_does_not_strictly_dominate_distance():
    # OJO: a diferencia de lo que se podria asumir, el tier NO domina de
    # forma absoluta sobre la distancia: la ventaja que da el tier es solo
    # 0.2 * PAGE_MAX_DISTANCE (~282.8), mientras que la distancia real puede
    # variar hasta la diagonal completa (~1414.2). Por eso un candidato del
    # tier "correcto" pero muy lejano puede perder frente a uno del tier
    # inferior pero muy cercano. Este comportamiento ya existia con el
    # PAGE_MAX_DISTANCE fijo (2000) y se mantiene con el valor calculado.
    extractor = make_extractor()
    key_bbox = (0, 0, 10, 10)
    far_same_row_value = (1000, 0, 1010, 10)
    close_below_value = (0, 11, 10, 20)

    score_same_row = extractor._score_value_candidate(key_bbox, far_same_row_value)
    score_below = extractor._score_value_candidate(key_bbox, close_below_value)

    assert score_same_row is not None and score_below is not None
    assert score_below > score_same_row


def test_score_value_candidate_returns_none_when_unrelated():
    extractor = make_extractor()
    key_bbox = (0, 0, 10, 10)
    unrelated_value = (500, 500, 510, 510)
    assert extractor._score_value_candidate(key_bbox, unrelated_value) is None


def test_score_value_candidate_prefers_closer_within_same_tier():
    extractor = make_extractor()
    key_bbox = (0, 0, 10, 10)
    close_value = (15, 0, 25, 10)
    far_value = (900, 0, 910, 10)

    close_score = extractor._score_value_candidate(key_bbox, close_value)
    far_score = extractor._score_value_candidate(key_bbox, far_value)
    assert close_score > far_score


# ---------------------------------------------------------------------------
# _is_reliable
# ---------------------------------------------------------------------------

def test_is_reliable_single_candidate_always_true():
    extractor = make_extractor()
    assert extractor._is_reliable([{"score": 5}]) is True


def test_is_reliable_false_when_scores_are_close():
    extractor = make_extractor(ambiguity_k=0.05)
    candidates = [{"score": 100}, {"score": 98}]
    assert extractor._is_reliable(candidates) is False


def test_is_reliable_true_when_best_clearly_wins():
    extractor = make_extractor(ambiguity_k=0.05)
    candidates = [{"score": 100}, {"score": 10}]
    assert extractor._is_reliable(candidates) is True


def test_is_reliable_false_when_best_score_not_positive():
    extractor = make_extractor()
    candidates = [{"score": -1}, {"score": -2}]
    assert extractor._is_reliable(candidates) is False


# ---------------------------------------------------------------------------
# _build_form: flujo completo de emparejamiento clave -> valor
# ---------------------------------------------------------------------------

def test_build_form_matches_key_to_value_same_row():
    extractor = make_extractor()
    entities = [
        {"text": "Nombre", "bbox": (0, 0, 50, 10), "label": "FIELD_KEY_NAME", "page": 1},
        {"text": "Juan Perez", "bbox": (60, 0, 120, 10), "label": "FIELD_VALUE_NAME", "page": 1},
    ]
    form = extractor._build_form(entities)
    assert len(form) == 1
    assert form[0]["field"] == "Nombre"
    assert form[0]["value"] == "Juan Perez"
    assert form[0]["is_reliable"] is True


def test_build_form_ignores_key_without_matching_value():
    extractor = make_extractor()
    entities = [
        {"text": "Nombre", "bbox": (0, 0, 50, 10), "label": "FIELD_KEY_NAME", "page": 1},
    ]
    form = extractor._build_form(entities)
    assert form == []


def test_build_form_two_keys_do_not_steal_each_others_value():
    extractor = make_extractor()
    entities = [
        {"text": "Nombre", "bbox": (0, 0, 50, 10), "label": "FIELD_KEY_NAME", "page": 1},
        {"text": "Juan Perez", "bbox": (60, 0, 120, 10), "label": "FIELD_VALUE_NAME", "page": 1},
        {"text": "Fecha", "bbox": (0, 20, 50, 30), "label": "FIELD_KEY_DATE", "page": 1},
        {"text": "01/01/2026", "bbox": (60, 20, 120, 30), "label": "FIELD_VALUE_DATE", "page": 1},
    ]
    form = extractor._build_form(entities)
    values_by_field = {f["field"]: f["value"] for f in form}
    assert values_by_field["Nombre"] == "Juan Perez"
    from datetime import datetime
    assert values_by_field["Fecha"] == datetime(2026, 1, 1)


# ---------------------------------------------------------------------------
# _build_tables / _resolve_row / _score_column_candidate
# ---------------------------------------------------------------------------

def test_build_tables_groups_items_into_rows_by_header():
    extractor = make_extractor()
    entities = [
        {"text": "Producto", "bbox": (0, 0, 50, 10), "label": "HEADER_T1_PRODUCT", "page": 1},
        {"text": "Precio", "bbox": (60, 0, 100, 10), "label": "HEADER_T1_PRICE", "page": 1},
        {"text": "Manzana", "bbox": (0, 20, 50, 30), "label": "ITEM_T1_PRODUCT", "page": 1},
        {"text": "1.50", "bbox": (60, 20, 100, 30), "label": "ITEM_T1_PRICE", "page": 1},
        {"text": "Pera", "bbox": (0, 40, 50, 50), "label": "ITEM_T1_PRODUCT", "page": 1},
        {"text": "2.00", "bbox": (60, 40, 100, 50), "label": "ITEM_T1_PRICE", "page": 1},
    ]
    tables = extractor._build_tables(entities)
    assert len(tables) == 1
    table = tables[0]
    assert table["headers"] == ["Producto", "Precio"]
    assert len(table["rows"]) == 2

    row0 = table["rows"][0]
    assert row0[0]["value"] == "Manzana"
    assert row0[1]["value"] == pytest.approx(1.50)

    row1 = table["rows"][1]
    assert row1[0]["value"] == "Pera"
    assert row1[1]["value"] == pytest.approx(2.00)


def test_build_tables_empty_when_no_entities_match():
    extractor = make_extractor()
    assert extractor._build_tables([]) == []


# ---------------------------------------------------------------------------
# Formateo de valores
# ---------------------------------------------------------------------------

def test_format_value_parses_numeric_types():
    extractor = make_extractor()
    assert extractor._format_value("12.5", "AMOUNT") == pytest.approx(12.5)


def test_format_value_parses_date_types():
    from datetime import datetime
    extractor = make_extractor()
    assert extractor._format_value("15/03/2026", "DATE") == datetime(2026, 3, 15)


def test_format_value_falls_back_to_text_on_bad_number():
    extractor = make_extractor()
    assert extractor._format_value("no-es-numero", "AMOUNT") == "no-es-numero"


def test_format_value_falls_back_to_text_on_bad_date():
    extractor = make_extractor()
    assert extractor._format_value("no-es-fecha", "DATE") == "no-es-fecha"


def test_format_value_passthrough_for_plain_text_type():
    extractor = make_extractor()
    assert extractor._format_value("hola", "NAME") == "hola"


# ---------------------------------------------------------------------------
# Utilidades de orden de lineas
# ---------------------------------------------------------------------------

def test_sort_key_orders_by_page_then_row_then_column():
    e1 = {"bbox": (10, 0, 20, 10), "page": 1}
    e2 = {"bbox": (0, 0, 10, 10), "page": 1}
    assert sort_key(e2, 3.0) < sort_key(e1, 3.0)


def test_estimate_line_tolerance_default_with_few_lines():
    assert estimate_line_tolerance([{"bbox": (0, 0, 1, 1)}]) == 3.0


def test_estimate_line_tolerance_computes_from_gaps():
    lines = [
        {"bbox": (0, 0, 1, 1)},
        {"bbox": (0, 10, 1, 11)},
        {"bbox": (0, 20, 1, 21)},
    ]
    tolerance = estimate_line_tolerance(lines)
    assert tolerance == pytest.approx(5.0)
