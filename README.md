# Face Verification API

 ONNX YOLOv5n-face for detection with histogram-based embedding matching.


## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload

# Or with custom port
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build
docker build -t face-verification .

# Run
docker run -p 8000:8000 face-verification
```

## API Endpoints


### Verify Faces

```bash
POST /api/v1/verify
Content-Type: application/json

{
  "m1_url": "https://example.com/passport.jpg",
  "m2_urls": [
    "https://example.com/selfie1.jpg",
    "https://example.com/selfie2.jpg",
    "https://example.com/selfie3.jpg"
  ]
}
```

**Response:**

```json
{
  "match_score": 0.9893,
  "match_percentage": 98.9,
  "is_match": true,
  "m1_face_detected": true,
  "m2_best_frame_index": 2,
  "processing_ms": 2572
}
```

## Configuration

Environment variables (optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/yolov5n-face.onnx` | Path to ONNX model |
| `IMG_SIZE` | `320` | Input image size for detection |
| `CONFIDENCE_THRESHOLD` | `0.45` | Face detection confidence threshold |
| `MAX_IMAGE_SIZE` | `1024` | Max dimension for downloaded images |
| `DOWNLOAD_TIMEOUT` | `5` | Image download timeout (seconds) |



## Project Structure

```
face-detection/
├── app/
│   ├── main.py              # FastAPI app with lifespan
│   ├── config.py            # Settings management
│   ├── api/v1/verify.py     # POST /api/v1/verify
│   └── core/
│       ├── models.py        # ONNX YOLOv5n-face detector
│       ├── alignment.py     # Face alignment (5-point transform)
│       ├── matcher.py       # Histogram + LBP embedding
│       └── downloader.py    # Async image fetcher
├── models/
│   └── yolov5n-face.onnx    # Pre-converted ONNX model
├── requirements.txt
├── Dockerfile
├── railway.toml
└── Procfile
```

## Performance

| Metric | Typical | Notes |
|--------|---------|-------|
| Cold start | ~3-5s | Model loading |
| Warm response | 1-3s | Depends on image sizes |
| Face detection | ~100ms | Per image, 320px input |
| Face alignment | ~5ms | Per face |
| Embedding | ~50ms | Per face |

## Technical Details

- **Detector**: YOLOv5n-face ONNX (1.7M params, 5.6 GFLOPs)
- **Alignment**: 5-point similarity transform (cv2.estimateAffinePartial2D)
- **Embedding**: Histogram + LBP features (272-dim vector)
- **Matching**: Cosine similarity
- **Framework**: FastAPI + ONNX Runtime

## License

MIT