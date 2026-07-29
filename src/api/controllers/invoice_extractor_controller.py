from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from api.models.dto import InvoiceDto
from api.models.response import InvoiceExtractionResponse
from api.services import (
    EmptyPdfError,
    invoice_extraction_service,
)

__all__ = ["router"]

router = APIRouter(prefix="/invoices", tags=["Invoice extraction"])


def _read_upload_in_chunks(upload: UploadFile) -> Iterator[bytes]:
    while chunk := upload.file.read(1024 * 1024):
        yield chunk


@router.post("/extract", response_model=InvoiceExtractionResponse, status_code=status.HTTP_200_OK,)
def extract_invoice(invoice_dto: InvoiceDto = Depends()) -> InvoiceExtractionResponse:
    file = invoice_dto.file
    filename = file.filename or ""
    if file.content_type != "application/pdf" and not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="El archivo debe ser un PDF.",
        )

    try:
        result = invoice_extraction_service.extract(_read_upload_in_chunks(file))
        return InvoiceExtractionResponse.model_validate(result)
    except EmptyPdfError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No se pudo procesar el PDF: {error}",
        ) from error
    finally:
        file.file.close()
