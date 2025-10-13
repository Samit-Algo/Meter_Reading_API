from fastapi import APIRouter, File, UploadFile, HTTPException, Form

from services.image_processing import MeterService

meter_reading_router = APIRouter()
meter_service = MeterService()

@meter_reading_router.post("/upload-meter-image")
async def upload_meter_image(
    file: UploadFile = File(...)
):
    """
    Upload an electricity meter image and extract the reading
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image content directly into memory
        content = await file.read()
        
        # Process the image directly from memory
        result = await meter_service.groq_process_meter_image_from_bytes(content)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")