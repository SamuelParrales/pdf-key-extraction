import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

API_URL = "http://127.0.0.1:8000/invoices/extract"
PDF_PATH = Path("data/test/Factura.pdf")


def build_multipart_body(pdf_path: Path, boundary: str) -> bytes:
    file_content = pdf_path.read_bytes()

    return b"".join(
        [
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{pdf_path.name}"\r\n'
            ).encode(),
            b"Content-Type: application/pdf\r\n\r\n",
            file_content,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )


def execute_endpoint() -> dict:
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"No se encontró la factura: {PDF_PATH}")

    boundary = f"----InvoiceExtractorBoundary{uuid4().hex}"
    request_body = build_multipart_body(PDF_PATH, boundary)
    request = Request(
        API_URL,
        data=request_body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(request_body)),
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8")
        raise RuntimeError(
            f"El endpoint respondió con HTTP {error.code}: {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"No se pudo conectar con el endpoint {API_URL}: {error.reason}"
        ) from error


if __name__ == "__main__":
    result = execute_endpoint()
    print(json.dumps(result, indent=2, ensure_ascii=False))
