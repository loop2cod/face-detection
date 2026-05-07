from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    model_path: str = "models/yolov5n-face.onnx"
    img_size: int = 320
    confidence_threshold: float = 0.45
    max_image_size: int = 1024
    download_timeout: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()