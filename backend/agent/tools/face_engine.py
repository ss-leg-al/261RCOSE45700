"""InsightFace face detection, embedding, and DBSCAN identity clustering."""
from __future__ import annotations

import numpy as np

_app = None


def _get_app():
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis
        _app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app


def detect_faces(image_path: str) -> list[dict]:
    """Returns [{bbox_xyxy, embedding, score}] for each face in the image."""
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return []
    return [
        {
            "bbox_xyxy":  face.bbox.tolist(),
            "embedding":  face.normed_embedding.tolist(),
            "score":      float(face.det_score),
        }
        for face in _get_app().get(img)
    ]


def embed_crop(image_path: str, bbox_xyxy: list[float]) -> list[float] | None:
    """Get InsightFace embedding from a pre-cropped face region (for Phase 2 matching)."""
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return None
    x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
    h, w = img.shape[:2]
    crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return None
    faces = _get_app().get(crop)
    return faces[0].normed_embedding.tolist() if faces else None


def cluster_embeddings(embeddings: list[list[float]], eps: float = 0.4) -> np.ndarray:
    from sklearn.cluster import DBSCAN
    if not embeddings:
        return np.array([], dtype=int)
    return DBSCAN(eps=eps, min_samples=1, metric="cosine").fit_predict(np.array(embeddings))


def find_best_cluster(
    embedding: list[float],
    cluster_embs: dict[int, list],
    threshold: float,
) -> int | None:
    emb = np.array(embedding)
    best_sim, best_id = -1.0, None
    for cid, embs in cluster_embs.items():
        for ref in embs:
            sim = float(np.dot(emb, np.array(ref)))
            if sim > best_sim:
                best_sim, best_id = sim, cid
    return best_id if best_sim >= threshold else None


def bbox_iou(a: list[float], b: list[float]) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0
