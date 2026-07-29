from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any

from extract.key_value_extractor import KeyValueExtractor

__all__ = [
    "EmptyPdfError",
    "InvoiceExtractionService",
    "invoice_extraction_service",
]


class EmptyPdfError(ValueError):
    """Raised when an uploaded PDF contains no data."""


class InvoiceExtractionService:
    """Runs the notebook inference flow for an uploaded PDF."""

    def __init__(self) -> None:
        self._extractor: KeyValueExtractor | None = None
        self._inference_lock = Lock()

    def extract(self, chunks: Iterable[bytes]) -> dict[str, Any]:
        temporary_path = self._write_temporary_pdf(chunks)

        try:
            with self._inference_lock:
                extractor = self._get_extractor()
                return extractor.predict(temporary_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _get_extractor(self) -> KeyValueExtractor:
        if self._extractor is None:
            self._extractor = KeyValueExtractor()
        return self._extractor

    @staticmethod
    def _write_temporary_pdf(chunks: Iterable[bytes]) -> Path:
        bytes_written = 0

        with NamedTemporaryFile(suffix=".pdf", delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
            for chunk in chunks:
                if not chunk:
                    continue
                temporary_file.write(chunk)
                bytes_written += len(chunk)

        if bytes_written == 0:
            temporary_path.unlink(missing_ok=True)
            raise EmptyPdfError("El archivo PDF está vacío.")

        return temporary_path


invoice_extraction_service = InvoiceExtractionService()
