from fastapi import APIRouter, HTTPException, Depends
from app.schemas.request import VerifyRequest
from app.schemas.response import VerifyResponse
from app.core.models import ONNXDetector
from app.core.downloader import download_multiple
from app.core.alignment import align_face
from app.core.matcher import simple_embedding, match_faces
from app.config import get_settings
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter()

_detector: ONNXDetector = None


def get_detector() -> ONNXDetector:
    global _detector
    if _detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _detector


def set_detector(detector: ONNXDetector):
    global _detector
    _detector = detector


@router.post("/verify", response_model=VerifyResponse)
async def verify_faces(
    request: VerifyRequest,
    detector: ONNXDetector = Depends(get_detector)
):
    settings = get_settings()
    start_time = time.time()
    
    m1_imgs = await download_multiple([request.m1_url], timeout=settings.download_timeout, max_size=settings.max_image_size)
    
    if m1_imgs[0] is None:
        raise HTTPException(status_code=400, detail="Failed to download m1 image")
    
    m1_detections = detector.detect(m1_imgs[0], img_size=settings.img_size, conf_threshold=settings.confidence_threshold)
    
    if not m1_detections:
        raise HTTPException(status_code=400, detail="No face detected in m1 image")
    
    best_m1 = max(m1_detections, key=lambda x: x["confidence"])
    aligned_m1 = align_face(m1_imgs[0], best_m1["landmarks"])
    embedding_m1 = simple_embedding(aligned_m1)
    
    m2_imgs = await download_multiple(request.m2_urls, timeout=settings.download_timeout, max_size=settings.max_image_size)
    
    m2_embeddings = []
    for img in m2_imgs:
        if img is None:
            continue
        
        detections = detector.detect(img, img_size=settings.img_size, conf_threshold=settings.confidence_threshold)
        
        if not detections:
            continue
        
        best = max(detections, key=lambda x: x["confidence"])
        aligned = align_face(img, best["landmarks"])
        embedding = simple_embedding(aligned)
        m2_embeddings.append(embedding)
    
    if not m2_embeddings:
        raise HTTPException(status_code=400, detail="No face detected in any m2 image")
    
    match_score, best_idx = match_faces(embedding_m1, m2_embeddings)
    
    threshold = request.options.get("match_threshold", 0.65) if request.options else 0.65
    is_match = match_score >= threshold
    match_percentage = round(match_score * 100, 1)
    
    processing_ms = int((time.time() - start_time) * 1000)
    
    return VerifyResponse(
        match_score=round(match_score, 4),
        match_percentage=match_percentage,
        is_match=is_match,
        m1_face_detected=True,
        m2_best_frame_index=best_idx,
        processing_ms=processing_ms,
    )