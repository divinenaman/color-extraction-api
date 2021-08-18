from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from process_image import color_extraction_model

app = FastAPI()

class Image(BaseModel):
    image: str


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/api/process_image")
async def process_image(image: Image):
    res = color_extraction_model(image.image)
    return {"data": res}