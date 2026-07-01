

import glob
import json
import os
from pathlib import Path
from PIL import Image
from common.common_types import LayoutElement
from config.env import settings


class DataStorage:
    base_path = Path(settings.data_path)

    @staticmethod
    def get_path(folder: str = None) -> Path:

        if folder:
            directory = DataStorage.base_path / folder 
            directory.mkdir(parents=True, exist_ok=True)
            return  directory
        
        return DataStorage.base_path

    @staticmethod
    def find_pdf_paths() -> list[Path]:
        dir = DataStorage.get_path('pdfs')
        patron_pdf = dir / "**" / "*.pdf"
        rutas_pdf = [Path(p) for p in glob.glob(str(patron_pdf), recursive=True)]
        
        return rutas_pdf
    
    @staticmethod
    def find_json_paths() -> list[Path]:
        dir = DataStorage.get_path('labeled')
        return list(Path(dir).rglob("*.json"))
    
    @staticmethod
    def get_images(folder_name: str) -> list[Image.Image]:
        directory = DataStorage.get_path("images") / folder_name

        images: list[Image.Image] = []

        for image_path in sorted(
            directory.glob("*.png"),
            key=lambda p: int(p.stem)
        ):
            images.append(Image.open(image_path).convert("RGB"))

        return images

    @staticmethod
    def save_layout_element(filename: str, element: LayoutElement) -> None:

        directory = DataStorage.get_path('unlabeled')

        with open(directory / Path(filename).with_suffix('.json'), "w", encoding="utf-8") as f:
            json.dump(element, f, ensure_ascii=False, indent=2)
   
    @staticmethod
    def save_images(filename: str, images: list) -> None:
        directory = DataStorage.get_path("images") / filename
        directory.mkdir(parents=True, exist_ok=True)

        for i, image in enumerate(images, start=1):
            image.save(directory / f"{i}.png", format="PNG")
    