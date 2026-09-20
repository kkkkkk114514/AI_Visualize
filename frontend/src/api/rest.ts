export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    throw new ApiError(response.status, await readErrorDetail(response));
  }
  return (await response.json()) as T;
}

async function readErrorDetail(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { error?: { detail?: string; message_key?: string } };
    return payload.error?.detail ?? payload.error?.message_key ?? `${response.status} ${response.statusText}`;
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}
