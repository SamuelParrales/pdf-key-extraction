import json
from pathlib import Path
import re

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
        images = []
        matrix = fitz.Matrix(2, 2)
        for page in doc:
            # Extraccion de la imagen completa
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes(
                "RGB",
                (pix.width, pix.height),
                pix.samples,
                )
            images.append(image)

            # Extracción de texto digital
            text_dict = page.get_text("dict")

            for block in text_dict["blocks"]:
                if block["type"] != 0:
                    continue
                result = self.extract_from_block(block, page,bboxes_to_remove=bboxes_to_remove)
                elements.extend(result)
  

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
        
        return elements, images
    def spans_to_fragments(self, spans):
        fragments = []

        current_text = ""
        current_boxes = []

        for span in spans:
            text = span["text"]

            # Dos o más espacios => nuevo fragmento
  
            if text.isspace() and len(text) >= 1:
                if current_boxes:
                    fragments.append({
                        "text": current_text.strip(),
                        "bbox": [
                            min(b[0] for b in current_boxes),
                            min(b[1] for b in current_boxes),
                            max(b[2] for b in current_boxes),
                            max(b[3] for b in current_boxes),
                        ],
                    })
                current_text = ""
                current_boxes = []
                continue

            if text.strip():
                current_text += text
                current_boxes.append(span["bbox"])

        if current_boxes:
            fragments.append({
                "text": current_text.strip(),
                "bbox": [
                    min(b[0] for b in current_boxes),
                    min(b[1] for b in current_boxes),
                    max(b[2] for b in current_boxes),
                    max(b[3] for b in current_boxes),
                ],
            })

        return fragments
    def print_blocks_text(self, block):
   
        if block["type"] == 0:
            print("--- BLOQUE ---")
            for line in block["lines"]:
                for span in line["spans"]:
                    print(f"  '{span['text']}'")
            print()

    def get_fragments_by_page_and_bbox(
        self,
        page: fitz.Page,
        bbox: tuple[float, float, float, float],
        y_tolerance: float = 1.7,
        # 9.2
        x_tolerance: float = 15,
    ) -> list[dict]:
        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        # Recolectar todas las líneas candidatas de la página una sola vez
        all_lines = []
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                fragments = self.spans_to_fragments(line.get("spans", []))
                
                for fragment in fragments:
                    text = fragment.get("text", "").strip()
                    if text:
                        all_lines.append(fragment)

        fragments = []
        current_bbox = list(bbox)  # bbox que se va expandiendo
        used_indices = set()

        changed = True
        while changed:
            changed = False
            x0, y0, x1, y1 = current_bbox

            for idx, element in enumerate(all_lines):
                if idx in used_indices:
                    continue

                el_x0, el_y0, el_x1, el_y1 = element["bbox"]

                is_same_element = abs(el_x0 - x0) < 0.01 and abs(el_y0 - y0) < 0.01
                if is_same_element:
                    continue

                vertical_distance = abs(y1 - el_y0)
                horizontal_offset = abs(el_x0 - x0)

                if vertical_distance < y_tolerance and horizontal_offset < x_tolerance:
                    fragments.append(element)
                    used_indices.add(idx)
                    # expandir el bbox actual para seguir buscando desde aquí
                    current_bbox = [
                        min(current_bbox[0], el_x0),
                        min(current_bbox[1], el_y0),
                        max(current_bbox[2], el_x1),
                        max(current_bbox[3], el_y1),
                    ]
                    changed = True
                    break  # reiniciar el loop con el bbox actualizado

        return fragments
            
        
    def process_fragments(self, main,fragments):
     
        if(len(fragments)==0):
            return False, main
            
        
        main_text = main.get("text")
        lines_boxes = [] 
        lines_boxes.append(main.get("bbox"))
        for fragment in fragments:
            fragment_text = fragment.get("text")
            if not fragment_text:
                continue
            # TODO: Aqui debemos usar un modelo de clasificacion
            
            if(main_text.isdigit() and fragment_text.isdigit()):
                main_text += fragment_text 
            else:
                main_text += " " + fragment_text 
            lines_boxes.append(fragment.get("bbox"))



        x0 = min(box[0] for box in lines_boxes)
        y0 = min(box[1] for box in lines_boxes)
        x1 = max(box[2] for box in lines_boxes)
        y1 = max(box[3] for box in lines_boxes)
        
        bbox = [x0, y0, x1, y1]
        final_fragment = {
                    'text': main_text, 
                    "bbox": bbox
        }
        
        is_valid = False
        if(len(fragments)>0):
            is_valid = self.is_valid_grouping(final_fragment)

 
        if(not is_valid):
            return False, main
     
        return is_valid, final_fragment
    
    def is_valid_grouping(self, fragment) -> bool:
     
        # Esta función deberia usar PLN para generalizar
        text = fragment.get('text')
        if text.strip() == "Identificación Fecha":
            return False
        
        id_date_pattern = re.compile(r"^\d{8,}\s+\d{1,2}/\d{1,2}/\d{2,4}$")
        if id_date_pattern.match(text.strip()):
            return False
        
     
        return True

    def extract_from_block(
        self,
        block,
        page: fitz.Page,
        bboxes_to_remove: set[tuple]
    ):
        page_number = page.number + 1
        page_width = page.rect.width
        page_height = page.rect.height

        elements: list[LayoutElement] = []

        for line in block["lines"]:
            main_fragments = self.spans_to_fragments(line["spans"])

            for main_fragment in main_fragments:

                line_text = main_fragment["text"]

                if not line_text:
                    continue

                if tuple(main_fragment["bbox"]) in bboxes_to_remove:
                    continue

                fragments = self.get_fragments_by_page_and_bbox(
                    page=page,
                    bbox=main_fragment["bbox"]
                )

                is_group, final_fragment = self.process_fragments(
                    main_fragment,
                    fragments
                )

                if is_group:
                    for fragment in fragments:
                        bboxes_to_remove.add(tuple(fragment["bbox"]))

                x0, y0, x1, y1 = final_fragment["bbox"]

                elements.append({
                    "text": final_fragment["text"],
                    "source": "digital",
                    "confidence": 1.0,
                    "page": page_number,
                    "bbox": [x0, y0, x1, y1],
                    "normalized_bbox": [
                        int((x0 / page_width) * 1000),
                        int((y0 / page_height) * 1000),
                        int((x1 / page_width) * 1000),
                        int((y1 / page_height) * 1000),
                    ],
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