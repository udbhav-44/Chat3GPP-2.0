import { getToken } from "./authToken";
import { getApiBaseUrl } from "./auth";

const request = async (path, options = {}) => {
  const token = getToken();
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
  const response = await fetch(`${getApiBaseUrl() || ""}${path}`, {
    ...options,
    headers,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = data.error || data.message || "Request failed";
    throw new Error(message);
  }
  return data;
};

export const listChats = () => request("/api/chats", { method: "GET" });

export const createChat = (title) =>
  request("/api/chats", {
    method: "POST",
    body: JSON.stringify({ title }),
  });

export const getChat = (threadId) =>
  request(`/api/chats/${threadId}`, { method: "GET" });

export const getChatResults = (threadId) =>
  request(`/api/chats/${threadId}/results`, { method: "GET" });

export const addChatMessage = (threadId, payload) =>
  request(`/api/chats/${threadId}/messages`, {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const deleteChat = (threadId) =>
  request(`/api/chats/${threadId}`, { method: "DELETE" });
