import onnxruntime as ort
import numpy as np
from typing import Tuple, List, Optional
import logging
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class ONNXDetector:
    def __init__(self, model_path: str, providers: List[str] = None):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.strides = [8, 16, 32]
        logger.info(f"ONNX model loaded from {model_path}")
        logger.info(f"Input: {self.input_name}, Outputs: {self.output_names}")

    def detect(self, img: np.ndarray, img_size: int = 320, conf_threshold: float = 0.45) -> List[dict]:
        h, w = img.shape[:2]

        input_img, ratio, pad = self._letterbox_resize(img, (img_size, img_size))

        input_img = input_img.transpose(2, 0, 1)
        input_img = np.ascontiguousarray(input_img)
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.expand_dims(input_img, axis=0)

        outputs = self.session.run(self.output_names, {self.input_name: input_img})

        detections = self._postprocess(outputs, conf_threshold, h, w, ratio, pad)
        return detections

    def _letterbox_resize(self, img: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w = img.shape[:2]
        target_h, target_w = target_size

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        result = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        dy = (target_h - new_h) // 2
        dx = (target_w - new_w) // 2
        result[dy:dy+new_h, dx:dx+new_w] = resized

        return result, scale, (dx, dy)

    def _postprocess(self, outputs, conf_threshold: float, orig_h: int, orig_w: int, scale: float, pad: Tuple[int, int]) -> List[dict]:
        all_dets = []

        for stride_idx, output in enumerate(outputs):
            stride = self.strides[stride_idx]

            output = output[0]

            for anchor in range(output.shape[0]):
                for grid_y in range(output.shape[1]):
                    for grid_x in range(output.shape[2]):
                        det = output[anchor, grid_y, grid_x]

                        conf = 1.0 / (1.0 + np.exp(-det[4]))
                        if conf < conf_threshold:
                            continue

                        cx = (1.0 / (1.0 + np.exp(-det[0])) * 2 - 0.5 + grid_x) * stride
                        cy = (1.0 / (1.0 + np.exp(-det[1])) * 2 - 0.5 + grid_y) * stride
                        bw = (det[2] * 2) ** 2
                        bh = (det[3] * 2) ** 2

                        x1 = cx - bw / 2
                        y1 = cy - bh / 2
                        x2 = cx + bw / 2
                        y2 = cy + bh / 2

                        x1 = (x1 - pad[0]) / scale
                        y1 = (y1 - pad[1]) / scale
                        x2 = (x2 - pad[0]) / scale
                        y2 = (y2 - pad[1]) / scale

                        x1 = max(0, min(x1, orig_w))
                        y1 = max(0, min(y1, orig_h))
                        x2 = max(0, min(x2, orig_w))
                        y2 = max(0, min(y2, orig_h))

                        landmarks = []
                        for i in range(5):
                            lx = (1.0 / (1.0 + np.exp(-det[6 + i * 2])) * 2 - 0.5 + grid_x) * stride
                            ly = (1.0 / (1.0 + np.exp(-det[6 + i * 2 + 1])) * 2 - 0.5 + grid_y) * stride

                            lx = (lx - pad[0]) / scale
                            ly = (ly - pad[1]) / scale
                            lx = max(0, min(lx, orig_w))
                            ly = max(0, min(ly, orig_h))

                            landmarks.append([lx, ly])

                        all_dets.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": conf,
                            "landmarks": landmarks
                        })

        if not all_dets:
            return []

        dets = self._nms(all_dets)

        return dets

    def _nms(self, detections: List[dict], iou_threshold: float = 0.45) -> List[dict]:
        if not detections:
            return []

        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            remaining = []
            for det in detections:
                iou = self._iou(best["bbox"], det["bbox"])
                if iou < iou_threshold:
                    remaining.append(det)

            detections = remaining

        return keep

    def _iou(self, box1: List[float], box2: List[float]) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0

        return inter_area / union_area