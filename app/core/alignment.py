import cv2
import numpy as np
from typing import Tuple


REFERENCE_LANDMARKS = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]
], dtype=np.float32)


def align_face(img: np.ndarray, landmarks: list, output_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    src_landmarks = np.array(landmarks, dtype=np.float32)

    if src_landmarks.shape[0] != 5:
        raise ValueError(f"Expected 5 landmarks, got {src_landmarks.shape[0]}")

    dst_landmarks = REFERENCE_LANDMARKS.copy()

    scale_x = output_size[0] / 112.0
    scale_y = output_size[1] / 112.0
    dst_landmarks[:, 0] *= scale_x
    dst_landmarks[:, 1] *= scale_y

    tform = cv2.estimateAffinePartial2D(src_landmarks, dst_landmarks)

    if tform[0] is None:
        raise ValueError("Failed to compute similarity transform")

    M = tform[0]

    aligned = cv2.warpAffine(img, M, output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))

    return aligned