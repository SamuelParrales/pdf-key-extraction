from pydantic import BaseModel, ConfigDict

from api.models.extract import ExtractedFormField, ExtractedTable

__all__ = ["InvoiceExtractionResponse"]


class InvoiceExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    form: list[ExtractedFormField]
    tables: list[ExtractedTable]
