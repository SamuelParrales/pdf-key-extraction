from fastapi import File, UploadFile

__all__ = ["InvoiceDto"]


class InvoiceDto:
    def __init__(self, file: UploadFile = File(...)) -> None:
        self.file = file
