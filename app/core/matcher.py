import numpy as np
import cv2
from typing import List


def cosine_similarity(embed1: np.ndarray, embed2: np.ndarray) -> float:
    embed1 = embed1.flatten()
    embed2 = embed2.flatten()

    dot_product = np.dot(embed1, embed2)
    norm1 = np.linalg.norm(embed1)
    norm2 = np.linalg.norm(embed2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def match_faces(embedding1: np.ndarray, embeddings2: List[np.ndarray]) -> tuple[float, int]:
    if not embeddings2:
        return 0.0, -1

    best_score = -1.0
    best_idx = -1

    for idx, emb2 in enumerate(embeddings2):
        if emb2 is None or len(emb2) == 0:
            continue
        score = cosine_similarity(embedding1, emb2)
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_score, best_idx


def simple_embedding(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        return np.zeros(272, dtype=np.float32)

    try:
        if len(img.shape) == 3 and img.shape[2] == 3:
            resized = cv2.resize(img, (64, 64))
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.resize(img, (64, 64))

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist / hist_sum
        else:
            hist = np.zeros(256, dtype=np.float32)

        mean_std = []
        for i in range(0, 64, 8):
            region = gray[:, i:i+8]
            if region.size > 0:
                mean_std.append(float(region.mean()) / 255.0)
                std_val = float(region.std()) / 255.0
                mean_std.append(std_val if std_val > 0 else 0.001)
            else:
                mean_std.extend([0.5, 0.001])

        while len(mean_std) < 16:
            mean_std.extend([0.5, 0.001])

        embedding = np.concatenate([hist, np.array(mean_std[:16], dtype=np.float32)])
        return embedding.astype(np.float32)
    except Exception as e:
        return np.zeros(272, dtype=np.float32)