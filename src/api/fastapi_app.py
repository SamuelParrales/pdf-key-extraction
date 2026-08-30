from fastapi import FastAPI
import os
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from extract.key_value_extractor import KeyValueExtractor

from api.controllers import router


extractor_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global extractor_instance
    print("--> Pre-loading ML Model into memory...")
    extractor_instance = KeyValueExtractor() 
    print("--> ML Model loaded and ready!")
    yield
    # Cleanu


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX_PATH = os.path.join(BASE_DIR, "index.html")



app = FastAPI(title="Invoice Extractor API", lifespan=lifespan)
app.include_router(router)



@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse(INDEX_PATH)
