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

    def get_fragments_by_block_and_bbox(self, block, bbox):
        x0, y0, x1, y1 = bbox
        y_factor = 1.7
          

        lines = block["lines"]
        fragments = []
        if len(lines) == 0:
            return None

        for i, line in enumerate(lines):
            spans = line["spans"]
            element = self.spans_to_fragment(spans)

            if not element.get("text"):
                continue

            element_x0, element_y0, element_x1, element_y1 = element["bbox"]
            if element_x0 == x0 and element_y1 == y1:
                continue

            factor = abs(y1 - element_y0)
     
             
            if x0 == element_x0 and factor < y_factor:
               
                element['index'] = i
                fragments.append(element)

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
            {
                'text': main_text, 
                "bbox": [x0, y0, x1, y1]
            }
        ] 
        sub_fragments = []
        for fragment in new_fragments:
            text = fragment.get("text")
            if ':' in text:
                parts = text.split(':')
                if len(parts) == 2 and not parts[1].strip(): break
                
                for j, part in enumerate(parts[:-1]):
                    if part.strip():
                        sub_fragments.append({'text': part.strip() + ':', 'bbox': fragment['bbox']})
                if parts[-1].strip():
                    sub_fragments.append({'text': parts[-1].strip(), 'bbox': fragment['bbox']})
        new_fragments.extend(sub_fragments)
        return new_fragments
        

    def extract_from_block(self, block, page: fitz.Page):
        page_number = page.number + 1
        page_width = page.rect.width
        page_height = page.rect.height
        elements: list[LayoutElement] = []

        lines = block["lines"]
        
        i = 0
        ignore_index: list[int] = [] 
        while i < len(lines):
            if(i in ignore_index):
                i += 1
                continue

            spans = lines[i]["spans"]
            main_fragment = self.spans_to_fragment(spans)
            line_text = main_fragment["text"]
           
            if not line_text:
                i += 1
                continue

            fragments = self.get_fragments_by_block_and_bbox(block=block, bbox=main_fragment["bbox"])
            for fragment in fragments:
                ignore_index.append(fragment['index'])

    
    
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