from fastapi import FastAPI
from api.v1.sentiment import router as sentiment_router

app = FastAPI()

app.include_router(sentiment_router, prefix="/api/v1")




