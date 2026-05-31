from pathlib import Path

import fitz
from PIL import Image
import numpy as np
from rapidocr_onnxruntime import RapidOCR
import io

from common.common_types import LayoutElement


class LayoutExtractor:

    def __init__(self):

        self.ocr_engine = RapidOCR()

    def extract_from_path(self,path: Path):
        doc:fitz.Document = fitz.open(path)

        return self.extract_from_doc(doc)

    def extract_from_doc(self, doc: fitz.Document):
        elements: list[LayoutElement] = []
        bboxes_to_remove: set[tuple] = set()

        for page in doc:
            text_dict = page.get_text("dict")

            for block in text_dict["blocks"]:
                if block["type"] != 0:
                    continue
                result, removed = self.extract_from_block(block, page)
                elements.extend(result)
                bboxes_to_remove.update(removed)

            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                img_rect = page.get_image_rects(xref)
                if not img_rect:
                    continue
                img_rect = img_rect[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                result = self.extract_from_img(img=pil_image, img_rect=img_rect, page=page)
                elements.extend(result)

        elements = [e for e in elements if tuple(e["bbox"]) not in bboxes_to_remove]
        return elements
    def spans_to_fragment(self,spans):
        span_boxes = []
        line_text = ""
        for span in spans:
            span_text = span["text"].strip()
            if span_text:
                line_text += span_text + " "
                x0, y0, x1, y1 = span["bbox"]
                span_boxes.append((x0, y0, x1, y1))

        x0 = min(box[0] for box in span_boxes)
        y0 = min(box[1] for box in span_boxes)
        x1 = max(box[2] for box in span_boxes)
        y1 = max(box[3] for box in span_boxes)
        return {
            "text": line_text.strip(),
            "bbox": [x0, y0, x1, y1],
        }

    def get_fragments_by_page_and_bbox(
        self,
        page: fitz.Page,
        bbox: tuple[float, float, float, float],
        y_tolerance: float = 1.7,
        x_tolerance: float = 9.2,
    ) -> list[dict]:
        x0, y0, x1, y1 = bbox
        fragments = []

        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # 0 = texto, 1 = imagen
                continue

            for i, line in enumerate(block.get("lines", [])):
                element = self.spans_to_fragment(line.get("spans", []))

                text = element.get("text", "").strip()
                if not text:
                    continue

                el_x0, el_y0, el_x1, el_y1 = element["bbox"]

                is_same_element = abs(el_x0 - x0) < 0.01 and abs(el_y1 - y1) < 0.01
                if is_same_element:
                    continue

                vertical_distance = abs(y1 - el_y0)
                horizontal_offset = abs(el_x0 - x0)

                if vertical_distance < y_tolerance and horizontal_offset < x_tolerance:
                    fragments.append({**element})

        return fragments
            
        
    def process_fragments(self, main,fragments):
        if(len(fragments)==0):
            return [main]
        
        main_text = main.get("text")
        lines_boxes = [] 
        lines_boxes.append(main.get("bbox"))
        for fragment in fragments:
            fragment_text = fragment.get("text")
            if not fragment_text:
                continue

            main_text += " " + fragment_text 
            lines_boxes.append(fragment.get("bbox"))
            x0 = min(box[0] for box in lines_boxes)
            y0 = min(box[1] for box in lines_boxes)
            x1 = max(box[2] for box in lines_boxes)
            y1 = max(box[3] for box in lines_boxes)
            
        new_fragments = [
            main,
            {
                'text': main_text, 
                "bbox": [x0, y0, x1, y1]
            }
        ] 
        # print(new_fragments)
      
     
        # sub_fragments = []
        # for fragment in new_fragments:
        #     text = fragment.get("text")
        #     if ':' in text:
        #         parts = text.split(':')
        #         if len(parts) == 2 and not parts[1].strip(): break
                
        #         for j, part in enumerate(parts[:-1]):
        #             if part.strip():
        #                 sub_fragments.append({'text': part.strip() + ':', 'bbox': fragment['bbox']})
        #         if parts[-1].strip():
        #             sub_fragments.append({'text': parts[-1].strip(), 'bbox': fragment['bbox']})
        # new_fragments.extend(sub_fragments)
        return new_fragments
        

    def extract_from_block(self, block, page: fitz.Page):
        page_number = page.number + 1
        page_width = page.rect.width
        page_height = page.rect.height
        elements: list[LayoutElement] = []
        bboxes_to_remove: set[tuple] = set()

        lines = block["lines"]

        i = 0
        while i < len(lines):
            spans = lines[i]["spans"]
            main_fragment = self.spans_to_fragment(spans)
            line_text = main_fragment["text"]

            if not line_text:
                i += 1
                continue

            fragments = self.get_fragments_by_page_and_bbox(page=page, bbox=main_fragment["bbox"])

            for fragment in fragments:
                bboxes_to_remove.add(tuple(fragment["bbox"]))

            final_fragments = self.process_fragments(main_fragment, fragments)
            for fragment in final_fragments:
                x0, y0, x1, y1 = fragment["bbox"]
                elements.append({
                    "text": fragment.get("text"),
                    "source": "digital",
                    "confidence": 1.0,
                    "page": page_number,
                    "bbox": [x0, y0, x1, y1],
                    "normalized_bbox": [
                        int((x0 / page_width) * 1000),
                        int((y0 / page_height) * 1000),
                        int((x1 / page_width) * 1000),
                        int((y1 / page_height) * 1000),
                    ]
                })

            i += 1

        return elements, bboxes_to_remove
        
    def extract_from_img(self,img: Image.Image, img_rect: fitz.Rect,page:fitz.Page):
        page_number = page.number + 1
        page_width = page.rect.width
        page_height = page.rect.height
        image_np = np.array(img)
        elements: list[LayoutElement] = []

        ocr_results, _ = self.ocr_engine(image_np)
        if not ocr_results:
            return elements

        # escala imagen → página
        scale_x = img_rect.width / img.width
        scale_y = img_rect.height / img.height

        for result in ocr_results:
            points, text, score = result
            if not text.strip():
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            # coords relativas a la imagen → coords en la página
            x0 = img_rect.x0 + min(xs) * scale_x
            y0 = img_rect.y0 + min(ys) * scale_y
            x1 = img_rect.x0 + max(xs) * scale_x
            y1 = img_rect.y0 + max(ys) * scale_y

            elements.append({
                "text": text,
                "source": "image_ocr",
                "confidence": float(score),
                "page": page_number,
                "bbox": [x0, y0, x1, y1],
                "normalized_bbox": [
                    int((x0 / page_width) * 1000),
                    int((y0 / page_height) * 1000),
                    int((x1 / page_width) * 1000),
                    int((y1 / page_height) * 1000),
                ]
            })
        return elements