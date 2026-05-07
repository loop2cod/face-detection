from pydantic import BaseModel, Field
from typing import Optional


class VerifyResponse(BaseModel):
    match_score: float = Field(..., description="Cosine similarity score")
    match_percentage: float = Field(..., description="Match score as percentage")
    is_match: bool = Field(..., description="Whether faces match")
    m1_face_detected: bool = Field(..., description="Face detected in m1")
    m2_best_frame_index: int = Field(..., description="Best matching frame index from m2")
    processing_ms: int = Field(..., description="Total processing time in milliseconds")
    model_versions: dict = Field(..., description="Model version information")