from __future__ import annotations

import asyncio
import mimetypes
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import numpy as np

from .agent.job_store import get_store, list_stores
from .agent.pipeline import run_detection_phase, run_masking_phase
from .agent.profile_store import delete_profile, get_profile, list_profiles, save_profile
from .agent.report_builder import build_final_report, write_report
from .agent.tools.sam3_modes import normalize_sam3_mode
from .config import settings
from .models.sam3_loader import get_load_error, is_available, load_sam3
from .schemas import (
    ApplyProfileResponse,
    CandidatesResponse,
    GuidelineResponse,
    JobCreated,
    JobStatus,
    JobSummary,
    ProfileSummary,
    SaveProfileRequest,
    SelectionRequest,
)

app = FastAPI(title="SafeVlog3 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

settings.upload_path.mkdir(parents=True, exist_ok=True)
settings.output_path.mkdir(parents=True, exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=str(settings.upload_path)), name="thumbnails")


@app.on_event("startup")
async def startup():
    load_sam3(settings.SAM3_CHECKPOINT)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "sam3": "loaded" if is_available() else "unavailable",
        "sam3_error": get_load_error(),
    }


@app.get("/api/jobs", response_model=list[JobSummary])
def list_jobs():
    return [
        JobSummary(
            job_id=s.job_id,
            status=s.status,
            scene_type=s.scene_type,
            face_count=len(s.face_clusters),
            pii_count=len(s.pii_candidates),
        )
        for s in list_stores()
    ]


@app.post("/api/jobs", response_model=JobCreated)
async def create_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > settings.MAX_VIDEO_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"파일이 너무 큽니다 (최대 {settings.MAX_VIDEO_SIZE_MB}MB)")

    job_id = str(uuid.uuid4())
    job_dir = settings.upload_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    video_path = job_dir / f"input{suffix}"
    video_path.write_bytes(content)

    store = get_store(job_id)
    store.video_path = str(video_path)

    background_tasks.add_task(run_detection_phase, job_id)
    return JobCreated(job_id=job_id)


@app.get("/api/jobs/{job_id}/status", response_model=JobStatus)
def get_status(job_id: str):
    store = get_store(job_id)
    return JobStatus(status=store.status, error=store.error)


@app.get("/api/jobs/{job_id}/candidates", response_model=CandidatesResponse)
def get_candidates(job_id: str):
    store = get_store(job_id)
    if store.status not in ("awaiting_selection", "masking", "done"):
        raise HTTPException(400, "아직 탐지가 완료되지 않았습니다")
    return CandidatesResponse(
        scene_type=store.scene_type or "unknown",
        face_clusters=[
            {
                "cluster_id":    c.cluster_id,
                "thumbnail_url": f"/thumbnails/{job_id}/thumbnails/{c.thumbnail}",
                "count":         c.count,
            }
            for c in store.face_clusters
        ],
        pii_candidates=[
            {
                "object_id":     p.object_id,
                "pii_type":      p.pii_type,
                "thumbnail_url": f"/thumbnails/{job_id}/thumbnails/{p.thumbnail}",
                "confidence":    p.confidence,
                "frame_index":   p.frame_index,
            }
            for p in store.pii_candidates
        ],
        scene_analysis=store.scene_analysis or None,
    )


@app.get("/api/jobs/{job_id}/guideline", response_model=GuidelineResponse)
def get_guideline(job_id: str):
    store = get_store(job_id)
    return GuidelineResponse(items=store.guideline)


@app.post("/api/jobs/{job_id}/skip")
def skip_job(job_id: str):
    from .agent.log_emitter import write_status
    store = get_store(job_id)
    if store.status != "awaiting_selection":
        raise HTTPException(400, f"현재 상태({store.status})에서는 스킵할 수 없습니다")
    store.output_video_path = store.video_path
    store.mask_preview_video_path = None
    store.mask_preview_frames_dir = None
    store.report = build_final_report(
        store=store,
        job_id=job_id,
        total_faces_blurred=0,
        total_pii_masked=0,
        output_video_path=store.video_path,
        skipped=True,
    )
    store.report.update({
        "detection_pii_types":           store.detection_pii_types or [],
        "deterministic_pii_types_added": store.deterministic_pii_types_added or [],
        "masked_pii_object_ids":         [],
        "selected_pii_object_count":     0,
        "colored_mask_enabled":          False,
        "colored_mask_preview_enabled":  False,
        "debug_mask_overlay_enabled":    False,
        "mask_preview_video_path":       None,
        "mask_colors":                   {},
    })
    write_report(store.report, store.video_path)
    store.status = "done"
    write_status(job_id, "done")
    return {"message": "편집 없이 완료"}


@app.post("/api/jobs/{job_id}/selection")
def submit_selection(
    job_id: str,
    body: SelectionRequest,
    background_tasks: BackgroundTasks,
):
    store = get_store(job_id)
    if store.status != "awaiting_selection":
        raise HTTPException(400, f"현재 상태({store.status})에서는 선택할 수 없습니다")
    store.protected_face_cluster_ids = body.protected_face_cluster_ids
    store.masked_pii_object_ids = list(dict.fromkeys(body.masked_pii_object_ids or []))
    store.masked_pii_types = list(dict.fromkeys(body.masked_pii_types))
    try:
        store.sam3_mode = normalize_sam3_mode(body.sam3_mode or settings.SAM3_MODE)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    background_tasks.add_task(run_masking_phase, job_id)
    return {"message": "마스킹을 시작합니다"}


@app.get("/api/profiles", response_model=list[ProfileSummary])
def list_profiles_endpoint():
    return [
        ProfileSummary(
            profile_id=p.profile_id,
            name=p.name,
            masked_pii_types=p.masked_pii_types,
            face_count=len(p.protected_face_embeddings),
        )
        for p in list_profiles()
    ]


@app.post("/api/jobs/{job_id}/save-profile", response_model=ProfileSummary)
def save_profile_endpoint(job_id: str, body: SaveProfileRequest):
    store = get_store(job_id)
    # Compute mean embedding per protected cluster as the saved face signature
    protected_embs: list[list[float]] = []
    for cid in body.protected_face_cluster_ids:
        embs = store.cluster_embeddings.get(cid, [])
        if embs:
            mean_emb = np.mean(embs, axis=0)
            mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)
            protected_embs.append(mean_emb.tolist())
    profile = save_profile(body.name, body.masked_pii_types, protected_embs)
    return ProfileSummary(
        profile_id=profile.profile_id,
        name=profile.name,
        masked_pii_types=profile.masked_pii_types,
        face_count=len(profile.protected_face_embeddings),
    )


@app.get("/api/jobs/{job_id}/apply-profile/{profile_id}", response_model=ApplyProfileResponse)
def apply_profile_endpoint(job_id: str, profile_id: str):
    store   = get_store(job_id)
    profile = get_profile(profile_id)
    if not profile:
        raise HTTPException(404, "프로필을 찾을 수 없습니다")

    protected_ids: list[int] = []
    for cid, embs in store.cluster_embeddings.items():
        if not embs:
            continue
        cluster_mean = np.mean(embs, axis=0)
        cluster_mean = cluster_mean / (np.linalg.norm(cluster_mean) + 1e-9)
        for saved_emb in profile.protected_face_embeddings:
            sim = float(np.dot(cluster_mean, np.array(saved_emb)))
            if sim >= settings.FACE_SIMILARITY_THRESHOLD:
                protected_ids.append(cid)
                break

    # PII types in profile but not detected in this job
    detected_pii = {p.pii_type for p in store.pii_candidates}
    unmatched_pii = [t for t in profile.masked_pii_types if t not in detected_pii]

    return ApplyProfileResponse(
        protected_face_cluster_ids=protected_ids,
        masked_pii_types=profile.masked_pii_types,
        matched_face_count=len(protected_ids),
        profile_face_count=len(profile.protected_face_embeddings),
        unmatched_pii_types=unmatched_pii,
    )


@app.delete("/api/profiles/{profile_id}", status_code=204)
def delete_profile_endpoint(profile_id: str):
    if not delete_profile(profile_id):
        raise HTTPException(404, "프로필을 찾을 수 없습니다")


@app.get("/api/jobs/{job_id}/stream")
async def stream_logs(job_id: str):
    async def generator():
        log_path = settings.upload_path / job_id / "logs.jsonl"
        sent = 0
        while True:
            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8").splitlines()
                for line in lines[sent:]:
                    yield f"data: {line}\n\n"
                sent = len(lines)
            store = get_store(job_id)
            if store.status in ("done", "failed"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/report")
def get_report(job_id: str):
    store = get_store(job_id)
    if not store.report:
        raise HTTPException(404, "리포트가 없습니다")
    return store.report


@app.get("/api/jobs/{job_id}/original")
def download_original(job_id: str):
    store = get_store(job_id)
    if not store.video_path:
        raise HTTPException(404, "원본 영상이 없습니다")
    path = Path(store.video_path)
    mime = mimetypes.guess_type(str(path))[0] or "video/mp4"
    return FileResponse(str(path), media_type=mime)


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str):
    store = get_store(job_id)
    if not store.output_video_path:
        raise HTTPException(404, "결과 영상이 없습니다")
    return FileResponse(store.output_video_path, media_type="video/mp4", filename="output.mp4")

@app.get("/api/jobs/{job_id}/mask-preview")
def mask_preview(job_id: str):
    store = get_store(job_id)
    if not store.mask_preview_video_path:
        raise HTTPException(404, "Colored mask preview video is not available")
    return FileResponse(
        store.mask_preview_video_path,
        media_type="video/mp4",
        filename="mask-preview.mp4",
    )
