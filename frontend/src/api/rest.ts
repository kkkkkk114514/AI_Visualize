import type { ApiErrorPayload } from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly payload: ApiErrorPayload | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { Accept: "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    const payload = await readErrorPayload(response);
    throw new ApiError(
      response.status,
      payload?.detail ?? payload?.message_key ?? `${response.status} ${response.statusText}`,
      payload,
    );
  }
  return (await response.json()) as T;
}

export async function apiGet<T>(path: string): Promise<T> {
  return request<T>(path);
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function apiDelete<T>(path: string): Promise<T> {
  return request<T>(path, { method: "DELETE" });
}

async function readErrorPayload(response: Response): Promise<ApiErrorPayload | null> {
  try {
    const payload = (await response.json()) as { error?: ApiErrorPayload };
    return payload.error ?? null;
  } catch {
    return null;
  }
}
