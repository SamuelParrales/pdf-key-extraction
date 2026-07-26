import json
import statistics
from pathlib import Path

from common.common_types import LayoutElement
from extract.layout_extractor import LayoutExtractor
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
LINE_TOLERANCE = 3.0  
def sort_key(e):
    x0, y0, x1, y1 = e["bbox"]

    y_group = int(y0 / LINE_TOLERANCE)

    return (e["page"], y_group, x0)

model_path = "./model-output/final"
class KeyValueExtractor:
    FIELD_KEY_PREFIX = "FIELD_KEY_"
    FIELD_VALUE_PREFIX = "FIELD_VALUE_"
    HEADER_PREFIX = "HEADER_"
    ITEM_PREFIX = "ITEM_"

    ROW_TOLERANCE = 8
    EDGE_TOLERANCE = 8
    COLUMN_TOLERANCE = 50
    PAGE_MAX_DISTANCE = 2000  # cota superior de referencia (> diagonal maxima posible en bbox normalizado)
    AMBIGUITY_K = 1.5

    def __init__(self, model_path="./model-output/final"):
        self.layout_extractor = LayoutExtractor()
        self.processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        self.model.eval()

    def predict(self, path: Path):
        lines, images = self.layout_extractor.extract_from_path(path)
        lines.sort(key=sort_key)

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

        print(json.dumps(output, indent=2, ensure_ascii=False))
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

    def _is_reliable(self, candidates):
        if len(candidates) == 1:
            return True

        scores = [c["score"] for c in candidates]
        best_score, second_score = sorted(scores, reverse=True)[:2]

        margin = self.AMBIGUITY_K * statistics.pstdev(scores)
        return (best_score - second_score) > margin

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
            tier = 1.1
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

            form.append({
                "field": key["text"],
                "value": best["value"]["text"],
                "value_type": best["value"]["label"],
                "is_reliable": self._is_reliable(available),
                "selected_by": "heuristic",
                "candidates": [
                    {"value": c["value"]["text"], "score": round(c["score"], 2)}
                    for c in candidates_by_key[key_idx][:3]
                ],
            })

        return form

    def _build_tables(self, entities):
        header_entities = sorted(
            (e for e in entities if e["label"].startswith(self.HEADER_PREFIX)),
            key=lambda e: (e["page"], e["bbox"][0]),
        )

        if not header_entities:
            return []

        headers = []
        column_by_suffix = {}
        header_bboxes = []
        for header in header_entities:
            suffix = header["label"][len(self.HEADER_PREFIX):]
            if suffix in column_by_suffix:
                continue
            column_by_suffix[suffix] = len(headers)
            headers.append(header["text"])
            header_bboxes.append(header["bbox"])

        item_entities = sorted(
            (e for e in entities if e["label"].startswith(self.ITEM_PREFIX)),
            key=lambda e: (e["page"], e["bbox"][1]),
        )

        rows = []
        current_row = None
        row_page = None
        row_y = None

        for item in item_entities:
            _, y0, _, _ = item["bbox"]

            if current_row is None or item["page"] != row_page or abs(y0 - row_y) > self.ROW_TOLERANCE:
                if current_row is not None:
                    rows.append(self._resolve_row(current_row))
                current_row = [[] for _ in headers]
                row_page = item["page"]
                row_y = y0

            suffix = item["label"][len(self.ITEM_PREFIX):]
            column = column_by_suffix.get(suffix)
            if column is None:
                continue

            score = self._score_column_candidate(header_bboxes[column], item["bbox"])
            current_row[column].append({"item": item, "score": score})

        if current_row is not None:
            rows.append(self._resolve_row(current_row))

        return [{"headers": headers, "rows": rows}]

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
                "value": best["item"]["text"],
                "value_type": best["item"]["label"],
                "is_reliable": is_reliable,
                "selected_by": "heuristic",
                "candidates": [
                    {"value": c["item"]["text"], "score": round(c["score"], 2)}
                    for c in candidates[:3]
                ],
            })

        return row

