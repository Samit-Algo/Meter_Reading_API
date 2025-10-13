from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from Api_endpoint.meter_reading import  meter_reading_router
from Api_endpoint.auth import auth_router

app = FastAPI(
    title="Electricity Bill Meter Reader API",
    description="API for processing electricity meter images and extracting readings using YOLO and OCR",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


app.include_router(meter_reading_router, prefix="/meter_reading_test", tags=["meter_reading_test"])
app.include_router(auth_router, prefix="/auth", tags=["authentication"])

@app.get("/")
async def root():
    return {"message": "Electricity Bill Meter Reader API is running"}