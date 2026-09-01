import json
import math
from datetime import datetime
from pathlib import Path

from common.common_types import LayoutElement
from config.env import settings
from extract.layout_extractor import LayoutExtractor
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch

DEFAULT_LINE_TOLERANCE = 3.0

# LayoutLMv3 normaliza todas las bbox a una cuadricula fija de 0 a 1000,
# sin importar el tamano real de la pagina (ver layout_extractor.py).
# La maxima distancia euclidiana posible entre dos puntos en ese espacio
# es la diagonal del cuadrado, sqrt(2) * 1000.
NORMALIZED_BBOX_MAX = 1000
MAX_NORMALIZED_DISTANCE = math.sqrt(2) * NORMALIZED_BBOX_MAX

NUMERIC_VALUE_TYPES = {"AMOUNT", "PRICE", "QUANTITY", "DISCOUNT", "TOTAL", "SUBSIDY", "WITHOUT_SUBSIDY"}
DATE_VALUE_TYPES = {"DATE"}
DATE_FORMATS = ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y")


def sort_key(e, line_tolerance):
    x0, y0, x1, y1 = e["bbox"]

    y_group = int(y0 / line_tolerance)

    return (e["page"], y_group, x0)


def estimate_line_tolerance(lines, default=DEFAULT_LINE_TOLERANCE):
    y0_values = sorted(e["bbox"][1] for e in lines)
    if len(y0_values) < 2:
        return default

    deltas = sorted(b - a for a, b in zip(y0_values, y0_values[1:]) if b - a > 0.5)
    if not deltas:
        return default

    median_gap = deltas[len(deltas) // 2]
    return max(median_gap * 0.5, 1.0)

class KeyValueExtractor:
    if settings.use_remote_model:
        if not settings.hf_model_repo_id:
            raise ValueError("USE_REMOTE_MODEL esta activo pero HF_MODEL_REPO_ID no esta configurado.")
        MODEL_PATH = settings.hf_model_repo_id
    else:
        MODEL_PATH = settings.model_path

    FIELD_KEY_PREFIX = settings.field_key_prefix
    FIELD_VALUE_PREFIX = settings.field_value_prefix
    HEADER_PREFIX = settings.header_prefix
    ITEM_PREFIX = settings.item_prefix

    ROW_TOLERANCE = settings.row_tolerance
    EDGE_TOLERANCE = settings.edge_tolerance
    COLUMN_TOLERANCE = settings.column_tolerance
    PAGE_MAX_DISTANCE = MAX_NORMALIZED_DISTANCE
    AMBIGUITY_K = settings.ambiguity_k

    def __init__(
        self,
        field_key_prefix: str | None = None,
        field_value_prefix: str | None = None,
        header_prefix: str | None = None,
        item_prefix: str | None = None,
        row_tolerance: int | None = None,
        edge_tolerance: int | None = None,
        column_tolerance: int | None = None,
        ambiguity_k: float | None = None,
    ):
        self._configure_extraction_parameters(field_key_prefix,field_value_prefix,header_prefix,item_prefix,row_tolerance,
                                              edge_tolerance,column_tolerance,ambiguity_k,)

        self.layout_extractor = LayoutExtractor()
        self.processor = LayoutLMv3Processor.from_pretrained(self.MODEL_PATH,apply_ocr=False,)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(self.MODEL_PATH)
        self.model.eval()

    def _configure_extraction_parameters(
        self,
        field_key_prefix: str | None,
        field_value_prefix: str | None,
        header_prefix: str | None,
        item_prefix: str | None,
        row_tolerance: int | None,
        edge_tolerance: int | None,
        column_tolerance: int | None,
        ambiguity_k: float | None,
    ) -> None:
        self.FIELD_KEY_PREFIX = (settings.field_key_prefix if field_key_prefix is None else field_key_prefix)
        self.FIELD_VALUE_PREFIX = (settings.field_value_prefix if field_value_prefix is None else field_value_prefix)
        self.HEADER_PREFIX = (settings.header_prefix if header_prefix is None else header_prefix)
        self.ITEM_PREFIX = settings.item_prefix if item_prefix is None else item_prefix
        self.ROW_TOLERANCE = (settings.row_tolerance if row_tolerance is None else row_tolerance)
        self.EDGE_TOLERANCE = (settings.edge_tolerance if edge_tolerance is None else edge_tolerance)
        self.COLUMN_TOLERANCE = (settings.column_tolerance if column_tolerance is None else column_tolerance)
        self.PAGE_MAX_DISTANCE = MAX_NORMALIZED_DISTANCE
        self.AMBIGUITY_K = (settings.ambiguity_k if ambiguity_k is None else ambiguity_k)


    def predict(self, path: Path):
        lines, images = self.layout_extractor.extract_from_path(path)
        line_tolerance = estimate_line_tolerance(lines)
        lines.sort(key=lambda e: sort_key(e, line_tolerance))

        results = []
        for page_index, image in enumerate(images):
            current_page = page_index + 1
            words, boxes = self.extract_words_boxes_by_page(current_page, lines)
            if not words:
                results.append({"words": [], "boxes": [], "labels": [],"page": current_page})
                continue

            res = self.predict_page(words=words,boxes=boxes,image=image)
            res['page'] = current_page

    
            results.append(res)
        
        # Aqui debe ir la sección de la heuristica
        output = self.apply_heuristic(results)

        # print(json.dumps(output, indent=2, ensure_ascii=False))
        return output
    
    def extract_words_boxes_by_page(self,page_index: int, lines: list[LayoutElement]):
        page_elements = [l for l in lines if l["page"] == page_index]

        words = [e["text"] for e in page_elements]
        boxes = [e["normalized_bbox"] for e in page_elements]
        return (words, boxes)
    
    def predict_page(self,words,boxes,image):
        encoding = self.processor(
            images=image,
            text=words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**encoding)

        pred_ids = outputs.logits.argmax(-1).squeeze().tolist()


        word_ids = encoding.word_ids(batch_index=0)

        word_labels = []
        seen_words = set()
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx in seen_words:
                continue  
            seen_words.add(word_idx)
            word_labels.append(self.model.config.id2label[pred_ids[token_idx]])
        return {
            "words": words,
            "boxes": boxes,
            "labels": word_labels
        }

    def apply_heuristic(self, results):
        entities = []
        for page_result in results:
            page = page_result["page"]
            for text, box, label in zip(
                page_result["words"], page_result["boxes"], page_result["labels"]
            ):
                if label == "O":
                    continue
                entities.append({"text": text, "bbox": box, "label": label, "page": page})

        return {
            "form": self._build_form(entities),
            "tables": self._build_tables(entities),
        }

    def _format_value(self, text, value_type):
        if value_type in DATE_VALUE_TYPES:
            return self._parse_date(text)
        if value_type in NUMERIC_VALUE_TYPES:
            return self._parse_number(text)
        return text

    def _parse_date(self, text):
        for date_format in DATE_FORMATS:
            try:
                return datetime.strptime(text.strip(), date_format)
            except ValueError:
                continue
        return text

    def _parse_number(self, text):
        try:
            return float(text.strip())
        except ValueError:
            return text

    def _is_reliable(self, candidates):
        if len(candidates) == 1:
            return True

        scores = [c["score"] for c in candidates]
        best_score, second_score = sorted(scores, reverse=True)[:2]

        if best_score <= 0:
            return False

        relative_margin = (best_score - second_score) / best_score
        return relative_margin > self.AMBIGUITY_K

    def _score_value_candidate(self, key_bbox, value_bbox):
        kx0, ky0, kx1, ky1 = key_bbox
        vx0, vy0, vx1, vy1 = value_bbox

        key_center = ((kx0 + kx1) / 2, (ky0 + ky1) / 2)
        value_center = ((vx0 + vx1) / 2, (vy0 + vy1) / 2)

        same_row = abs(ky0 - vy0) <= self.ROW_TOLERANCE
        to_the_right = vx0 >= kx1 - self.EDGE_TOLERANCE

        # Alineados si los centros estan cerca, o si sus rangos horizontales se solapan/casi se tocan
        centered = abs(value_center[0] - key_center[0]) <= self.COLUMN_TOLERANCE
        horizontal_gap = max(0.0, max(kx0, vx0) - min(kx1, vx1))
        overlapping = horizontal_gap <= self.COLUMN_TOLERANCE

        aligned_below = vy0 >= ky1 - self.EDGE_TOLERANCE and (centered or overlapping)

        if same_row and to_the_right:
            tier = 1.2
        elif aligned_below:
            tier = 1
        else:
            return None

        distance = (
            (key_center[0] - value_center[0]) ** 2
            + (key_center[1] - value_center[1]) ** 2
        ) ** 0.5

        score = tier * self.PAGE_MAX_DISTANCE - distance
        return score

    def _build_form(self, entities):
        keys = [e for e in entities if e["label"].startswith(self.FIELD_KEY_PREFIX)]
        values = [e for e in entities if e["label"].startswith(self.FIELD_VALUE_PREFIX)]


        candidates_by_key = []
        for key in keys:
            suffix = key["label"][len(self.FIELD_KEY_PREFIX):]
            expected_label = self.FIELD_VALUE_PREFIX + suffix

            key_candidates = []
            for value_idx, value in enumerate(values):
                if value["page"] != key["page"] or value["label"] != expected_label:
                    continue
                score = self._score_value_candidate(key["bbox"], value["bbox"])
                if score is None:
                    continue

                key_candidates.append({"value_idx": value_idx, "value": value, "score": score})

            key_candidates.sort(key=lambda c: c["score"], reverse=True)
            candidates_by_key.append(key_candidates)

        global_pairs = []
        for key_id, key_candidates in enumerate(candidates_by_key):
            for candidate in key_candidates:
                global_pairs.append((key_id, candidate))

        global_pairs.sort(key=lambda pair: pair[1]["score"], reverse=True)


        assigned_by_key = {}
        used_values = set()
        for key_id, candidate in global_pairs:
            if key_id in assigned_by_key or candidate["value_idx"] in used_values:
                continue
            assigned_by_key[key_id] = candidate
            used_values.add(candidate["value_idx"])

  
        form = []
        for key_idx, key in enumerate(keys):
            best = assigned_by_key.get(key_idx)
            if best is None:
                continue


            available = [
                c for c in candidates_by_key[key_idx]
                if c["value_idx"] == best["value_idx"] or c["value_idx"] not in used_values
            ]

            suffix = key["label"][len(self.FIELD_KEY_PREFIX):]

            form.append({
                "field": key["text"],
                "value": self._format_value(best["value"]["text"], suffix),
                "value_type": best["value"]["label"],
                "is_reliable": self._is_reliable(available),
                "selected_by": "heuristic",
                "candidates": [
                    {"value": self._format_value(c["value"]["text"], suffix), "score": round(c["score"], 2)}
                    for c in available[:3]
                ],
            })

        return form

    def _build_tables(self, entities):
        header_entities = sorted(
            (e for e in entities if e["label"].startswith(self.HEADER_PREFIX)),
            key=lambda e: (e["page"], e["bbox"][0]),
        )
        item_entities = sorted(
            (e for e in entities if e["label"].startswith(self.ITEM_PREFIX)),
            key=lambda e: (e["page"], e["bbox"][1]),
        )

        headers_by_table = {}
        for header in header_entities:
            table_name, column = header["label"][len(self.HEADER_PREFIX):].split("_", 1)
           
            headers_by_table.setdefault(table_name, []).append((column, header))

        items_by_table = {}
        for item in item_entities:
            table_name, column = item["label"][len(self.ITEM_PREFIX):].split("_", 1)
  
            items_by_table.setdefault(table_name, []).append((column, item))

        tables = []
        for table_name, header_list in headers_by_table.items():
            tables.append(
                self._build_single_table(header_list, items_by_table.get(table_name, []))
            )

        return tables

    def _build_single_table(self, header_list, item_list):
        headers = []
        column_by_suffix = {}
        header_bboxes = []
        for column, header in header_list:
            if column in column_by_suffix:
                continue
            column_by_suffix[column] = len(headers)
            headers.append(header["text"])
            header_bboxes.append(header["bbox"])

        rows = []
        current_row = None
        row_page = None
        row_y = None

        for column, item in item_list:
            _, y0, _, _ = item["bbox"]

            if current_row is None or item["page"] != row_page or abs(y0 - row_y) > self.ROW_TOLERANCE:
                if current_row is not None:
                    rows.append(self._resolve_row(current_row))
                current_row = [[] for _ in headers]
                row_page = item["page"]
                row_y = y0

            column_index = column_by_suffix.get(column)
            if column_index is None:
                continue

            score = self._score_column_candidate(header_bboxes[column_index], item["bbox"])
            current_row[column_index].append({"item": item, "score": score, "column": column})

        if current_row is not None:
            rows.append(self._resolve_row(current_row))

        return {"headers": headers, "rows": rows}

    def _score_column_candidate(self, header_bbox, item_bbox):
        hx0, _, hx1, _ = header_bbox
        ix0, _, ix1, _ = item_bbox

        header_center = (hx0 + hx1) / 2
        item_center = (ix0 + ix1) / 2

        return -abs(header_center - item_center)

    def _resolve_row(self, column_candidates):
        row = []
        for candidates in column_candidates:
            if not candidates:
                row.append(None)
                continue

            candidates.sort(key=lambda c: c["score"], reverse=True)
            best = candidates[0]

            is_reliable = self._is_reliable(candidates)

            row.append({
                "value": self._format_value(best["item"]["text"], best["column"]),
                "value_type": best["item"]["label"],
                "is_reliable": is_reliable,
                "selected_by": "heuristic",
                "candidates": [
                    {"value": self._format_value(c["item"]["text"], c["column"]), "score": round(c["score"], 2)}
                    for c in candidates[:3]
                ],
            })

        return row

