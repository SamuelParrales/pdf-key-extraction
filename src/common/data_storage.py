

import glob
import json
import os
from pathlib import Path

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
    def save_layout_element(filename: str, element: LayoutElement) -> None:

        directory = DataStorage.get_path('unlabeled')

        with open(directory / Path(filename).with_suffix('.json'), "w", encoding="utf-8") as f:
            json.dump(element, f, ensure_ascii=False, indent=2)
   