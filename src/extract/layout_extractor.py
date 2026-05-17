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

    def extract_from_doc(self,doc:fitz.Document):
        elements:list[LayoutElement] = []
        for page in doc:
            text_dict = page.get_text("dict")
            for block in text_dict["blocks"]:
              
                if block["type"] != 0:
                    continue
                result = self.extract_from_block(block,page)
                elements.extend(result)
                

            image_list = page.get_images(full=True)
        
            # Imagenes
            for img in image_list:
                xref = img[0]
           
                img_rect = page.get_image_rects(xref)
                if not img_rect:
                    continue
                img_rect = img_rect[0] 

                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                result = self.extract_from_img(img=pil_image,img_rect=img_rect,page=page)
                elements.extend(result)
        return elements
    
    def extract_from_block(self, block, page:fitz.Page):
        page_number = page.number + 1
        page_width = page.rect.width
        page_height = page.rect.height
        elements:list[LayoutElement] = []
        
        for line in block["lines"]:
            line_text = ""
            span_boxes = []

            for span in line["spans"]:
                span_text = span["text"].strip()

                if not span_text:
                    continue

                line_text += span_text + " "

                x0, y0, x1, y1 = span["bbox"]

                span_boxes.append((x0, y0, x1, y1))

            line_text = line_text.strip()

            if not line_text:
                continue


            x0 = min(box[0] for box in span_boxes)
            y0 = min(box[1] for box in span_boxes)
            x1 = max(box[2] for box in span_boxes)
            y1 = max(box[3] for box in span_boxes)

            elements.append({
                "text": line_text,
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

        return elements
    
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