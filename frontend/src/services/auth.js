import { getToken } from "./authToken";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

const request = async (path, options = {}) => {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const message = data.error || data.message || "Request failed";
    throw new Error(message);
  }

  return data;
};

export const signup = (payload) =>
  request("/api/auth/signup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

export const login = (payload) =>
  request("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

export const forgotPassword = (payload) =>
  request("/api/auth/forgot", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

export const resetPassword = (payload) =>
  request("/api/auth/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

export const getCurrentUser = async () => {
  const token = getToken();
  if (!token) {
    return null;
  }

  const data = await request("/api/auth/me", {
    headers: { Authorization: `Bearer ${token}` },
  });
  return data.user;
};

export const getApiBaseUrl = () => API_BASE_URL;
