from pydantic import BaseModel


class JobCreated(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    status: str
    error: str | None = None


class JobSummary(BaseModel):
    job_id: str
    status: str
    scene_type: str | None = None
    face_count: int = 0
    pii_count: int = 0


class FaceClusterItem(BaseModel):
    cluster_id: int
    thumbnail_url: str
    count: int


class PIICandidateItem(BaseModel):
    object_id: int
    pii_type: str
    thumbnail_url: str
    confidence: float
    frame_index: int | None = None


class CandidatesResponse(BaseModel):
    scene_type: str
    face_clusters: list[FaceClusterItem]
    pii_candidates: list[PIICandidateItem]
    scene_analysis: dict | None = None


class GuidelineItem(BaseModel):
    level: str   # "warning" | "info"
    category: str  # "face" | "pii" | "scene"
    message: str


class GuidelineResponse(BaseModel):
    items: list[GuidelineItem]


class SelectionRequest(BaseModel):
    protected_face_cluster_ids: list[int]
    masked_pii_types: list[str]
    masked_pii_object_ids: list[int] | None = None


class ProfileSummary(BaseModel):
    profile_id: str
    name: str
    masked_pii_types: list[str]
    face_count: int  # number of saved protected-face embeddings


class SaveProfileRequest(BaseModel):
    name: str
    protected_face_cluster_ids: list[int]
    masked_pii_types: list[str]


class ApplyProfileResponse(BaseModel):
    protected_face_cluster_ids: list[int]
    masked_pii_types: list[str]
    matched_face_count: int       # profile faces found in this video
    profile_face_count: int       # total faces saved in profile
    unmatched_pii_types: list[str]  # profile PII types not detected in this video
