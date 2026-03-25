import { getToken } from "./authToken";

const fallbackBaseUrl =
  typeof window !== "undefined" ? window.location.origin : "http://localhost:8000";
const uploadBaseUrl = import.meta.env.VITE_UPLOAD_BASE_URL || fallbackBaseUrl;

const buildAuthHeaders = () => {
  const token = getToken();
  if (!token) {
    return {};
  }
  return { Authorization: `Bearer ${token}` };
};

export const listUploads = async () => {
  const response = await fetch(`${uploadBaseUrl}/uploads`, {
    headers: {
      ...buildAuthHeaders(),
    },
  });
  if (!response.ok) {
    throw new Error("Failed to fetch uploads");
  }
  const data = await response.json();
  return Array.isArray(data.files) ? data.files : [];
};

export const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${uploadBaseUrl}/upload`, {
    method: "POST",
    headers: {
      ...buildAuthHeaders(),
    },
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Upload failed");
  }

  return response.json();
};

export const uploadFiles = async (files) => {
  const entries = Array.from(files || []);
  const results = [];

  for (const file of entries) {
    try {
      const payload = await uploadFile(file);
      results.push({ file, ok: true, payload });
    } catch (error) {
      results.push({ file, ok: false, error });
    }
  }

  return results;
};

export const deleteUpload = async (filename) => {
  const response = await fetch(`${uploadBaseUrl}/upload/${encodeURIComponent(filename)}`, {
    method: "DELETE",
    headers: {
      ...buildAuthHeaders(),
    },
  });

  if (!response.ok) {
    throw new Error("Delete failed");
  }

  return response.json();
};
