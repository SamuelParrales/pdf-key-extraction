from pydantic import BaseModel

__all__ = [
    "ExtractedFormField",
    "ExtractedTable",
    "ExtractedTableCell",
    "ExtractionCandidate",
]


class ExtractionCandidate(BaseModel):
    value: str
    score: float


class ExtractedFormField(BaseModel):
    field: str
    value: str
    value_type: str
    is_reliable: bool
    selected_by: str
    candidates: list[ExtractionCandidate]


class ExtractedTableCell(BaseModel):
    value: str
    value_type: str
    is_reliable: bool
    selected_by: str
    candidates: list[ExtractionCandidate]


class ExtractedTable(BaseModel):
    headers: list[str]
    rows: list[list[ExtractedTableCell | None]]
