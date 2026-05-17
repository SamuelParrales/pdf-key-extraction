from typing import Literal, TypedDict


class LayoutElement(TypedDict):
    text: str
    source: Literal["digital", "image_ocr"]
    confidence: float
    page: int
    bbox: tuple[float, float, float, float]
    normalized_bbox: tuple[int, int, int, int]