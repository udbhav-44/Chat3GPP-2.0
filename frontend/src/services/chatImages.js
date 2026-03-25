import { getToken } from "./authToken";

const fallbackBaseUrl =
  typeof window !== "undefined" ? window.location.origin : "http://localhost:8000";
const chatUploadBaseUrl =
  import.meta.env.VITE_CHAT_UPLOAD_BASE_URL ||
  import.meta.env.VITE_UPLOAD_BASE_URL ||
  fallbackBaseUrl;

const buildAuthHeaders = () => {
  const token = getToken();
  if (!token) {
    return {};
  }
  return { Authorization: `Bearer ${token}` };
};

const buildTokenQuery = () => {
  const token = getToken();
  if (!token) {
    return "";
  }
  return `token=${encodeURIComponent(token)}`;
};

export const getChatImageUrl = (threadId, filename) => {
  if (!threadId || !filename) {
    return "";
  }
  const tokenQuery = buildTokenQuery();
  const suffix = tokenQuery ? `?${tokenQuery}` : "";
  return `${chatUploadBaseUrl}/chat-upload/${encodeURIComponent(
    threadId
  )}/${encodeURIComponent(filename)}${suffix}`;
};

export const listChatImages = async (threadId) => {
  const response = await fetch(
    `${chatUploadBaseUrl}/chat-uploads?thread_id=${encodeURIComponent(threadId)}`,
    {
      headers: {
        ...buildAuthHeaders(),
      },
    }
  );
  if (!response.ok) {
    throw new Error("Failed to fetch chat images");
  }
  const data = await response.json();
  return Array.isArray(data.files) ? data.files : [];
};

export const uploadChatImage = async (file, threadId) => {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(
    `${chatUploadBaseUrl}/chat-upload?thread_id=${encodeURIComponent(threadId)}`,
    {
      method: "POST",
      headers: {
        ...buildAuthHeaders(),
      },
      body: formData,
    }
  );
  if (!response.ok) {
    throw new Error("Image upload failed");
  }
  return response.json();
};

export const uploadChatImages = async (files, threadId) => {
  const entries = Array.from(files || []);
  const results = [];

  for (const file of entries) {
    try {
      const payload = await uploadChatImage(file, threadId);
      results.push({ file, ok: true, payload });
    } catch (error) {
      results.push({ file, ok: false, error });
    }
  }

  return results;
};

export const deleteChatImage = async (threadId, filename) => {
  const response = await fetch(
    `${chatUploadBaseUrl}/chat-upload/${encodeURIComponent(
      threadId
    )}/${encodeURIComponent(filename)}`,
    {
      method: "DELETE",
      headers: {
        ...buildAuthHeaders(),
      },
    }
  );
  if (!response.ok) {
    throw new Error("Delete failed");
  }
  return response.json();
};
