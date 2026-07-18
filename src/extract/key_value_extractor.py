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

        print(results)
    
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
    

