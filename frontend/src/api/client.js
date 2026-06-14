import axios from "axios";

const http = axios.create({ baseURL: "" });

export const api = {
  async listJobs() {
    return (await http.get("/api/jobs")).data;
  },

  async createJob(file) {
    const form = new FormData();
    form.append("file", file);
    return (await http.post("/api/jobs", form)).data;
  },

  async deleteJob(jobId) {
    await http.delete(`/api/jobs/${jobId}`);
  },

  async getStatus(jobId) {
    return (await http.get(`/api/jobs/${jobId}/status`)).data;
  },

  async getCandidates(jobId) {
    return (await http.get(`/api/jobs/${jobId}/candidates`)).data;
  },

  async getGuideline(jobId) {
    return (await http.get(`/api/jobs/${jobId}/guideline`)).data;
  },

  async listProfiles() {
    return (await http.get("/api/profiles")).data;
  },

  async saveProfile(jobId, name, protectedFaceClusterIds, maskedPiiTypes) {
    return (await http.post(`/api/jobs/${jobId}/save-profile`, {
      name,
      protected_face_cluster_ids: protectedFaceClusterIds,
      masked_pii_types: maskedPiiTypes,
    })).data;
  },

  async applyProfile(jobId, profileId) {
    return (await http.get(`/api/jobs/${jobId}/apply-profile/${profileId}`)).data;
  },

  async deleteProfile(profileId) {
    await http.delete(`/api/profiles/${profileId}`);
  },

  async skipJob(jobId) {
    return (await http.post(`/api/jobs/${jobId}/skip`)).data;
  },

  async submitSelection(jobId, protectedFaceIds, maskedPiiTypes, maskedPiiObjectIds = [], sam3Mode = "normal") {
    return (
      await http.post(`/api/jobs/${jobId}/selection`, {
        protected_face_cluster_ids: protectedFaceIds,
        masked_pii_types: maskedPiiTypes,
        masked_pii_object_ids: maskedPiiObjectIds,
        sam3_mode: sam3Mode,
      })
    ).data;
  },

  downloadUrl:  (jobId) => `/api/jobs/${jobId}/download`,
  originalUrl:  (jobId) => `/api/jobs/${jobId}/original`,
  maskPreviewUrl: (jobId) => `/api/jobs/${jobId}/mask-preview`,

  async getReport(jobId) {
    return (await http.get(`/api/jobs/${jobId}/report`)).data;
  },
};
