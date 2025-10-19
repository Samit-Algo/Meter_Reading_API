# Ollama Setup Guide for Meter Reading API

This API now uses **Ollama** with the **llava:7b** vision model for meter reading extraction.

## Prerequisites

1. **Install Ollama**
   - Download and install Ollama from: https://ollama.ai/
   - For Windows: Download the Windows installer
   - For Linux: `curl -fsSL https://ollama.ai/install.sh | sh`
   - For macOS: Download from the website or use Homebrew

2. **Verify Ollama Installation**
   ```bash
   ollama --version
   ```

## Setup Steps

### 1. Start Ollama Service

The Ollama service should start automatically after installation. If not, start it manually:

**Windows:**
- Ollama runs as a background service automatically
- Check the system tray for the Ollama icon

**Linux/macOS:**
```bash
ollama serve
```

### 2. Pull the llava:7b Model

Once Ollama is running, pull the llava:7b vision model:

```bash
ollama pull llava:7b
```

This will download the model (approximately 4.7 GB). Wait for it to complete.

### 3. Verify the Model is Available

```bash
ollama list
```

You should see `llava:7b` in the list of available models.

### 4. Test the Model (Optional)

Test if the model works correctly:

```bash
ollama run llava:7b "What do you see?" --image path/to/test/image.jpg
```

## Configuration

### Environment Variables (Optional)

Create a `.env` file in the project root if you need to customize the Ollama connection:

```env
# Ollama API URL (default: http://localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434
```

If you're running Ollama on a different machine or port, update the `OLLAMA_BASE_URL` accordingly.

## Running the API

1. **Install Python Dependencies**
   ```bash
   pip install -r Requirements.txt
   ```

2. **Start the API Server**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Test the API**
   Visit http://localhost:8000/docs for the interactive API documentation.

## Troubleshooting

### Issue: "Connection refused" or "Ollama API error"

**Solution:**
- Ensure Ollama service is running
- Check if Ollama is accessible at http://localhost:11434
- Test with: `curl http://localhost:11434/api/tags`

### Issue: Model not found

**Solution:**
- Run `ollama pull llava:7b` to download the model
- Verify with `ollama list`

### Issue: Slow responses

**Solution:**
- llava:7b is a 7 billion parameter model and may take 10-30 seconds per request depending on your hardware
- Ensure you have sufficient RAM (8GB minimum, 16GB recommended)
- GPU acceleration will significantly improve performance if available

### Issue: Out of memory

**Solution:**
- Close other applications to free up RAM
- Consider using a smaller model if available
- Increase system swap/pagefile

## Performance Notes

- **First request** may be slower as the model loads into memory
- **Subsequent requests** will be faster (model stays in memory)
- **GPU acceleration** is automatically used if available (NVIDIA/AMD/Apple Silicon)
- **Expected response time**: 10-30 seconds per image on CPU, 2-5 seconds on GPU

## API Endpoints

### Upload Meter Image
```
POST /meter_reading_test/upload-meter-image
```

Upload an image of a utility meter to extract the reading.

**Request:**
- Content-Type: multipart/form-data
- Body: file (image file)

**Response:**
```json
{
  "reading": "12345.67 kWh",
  "confidence": "85%"
}
```

Or if not visible:
```json
{
  "reading": "Not visible",
  "confidence": "0%",
  "reason": "Meter display is blurry or obstructed"
}
```

## Model Information

- **Model**: llava:7b
- **Type**: Vision-Language Model
- **Size**: ~4.7 GB
- **Context**: 4096 tokens
- **Capabilities**: Image understanding, OCR, visual reasoning

## Additional Resources

- Ollama Documentation: https://github.com/ollama/ollama
- llava Model Card: https://ollama.ai/library/llava
- API Documentation: http://localhost:8000/docs (when running)

