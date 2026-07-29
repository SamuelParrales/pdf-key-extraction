from fastapi import FastAPI

from api.controllers import router

app = FastAPI(title="Invoice Extractor API")
app.include_router(router)
