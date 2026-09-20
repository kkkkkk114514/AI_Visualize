import { apiGet } from "./rest";
import type { ProbeKind, SnapshotPayload, SnapshotsResponse } from "./types";

/** 按 id 缓存的 payload 上限：超出后淘汰最久未用的（消费侧只看当前步 + 相邻步）。 */
const CACHE_LIMIT = 64;

const payloads = new Map<string, SnapshotPayload>();
const etags = new Map<string, string>();
const inflight = new Map<string, Promise<SnapshotPayload>>();

function keyOf(runId: string, snapshotId: string): string {
  return `${runId}:${snapshotId}`;
}

function touch(key: string, payload: SnapshotPayload): void {
  payloads.delete(key);
  payloads.set(key, payload);
  while (payloads.size > CACHE_LIMIT) {
    const oldest = payloads.keys().next().value;
    if (oldest === undefined) break;
    payloads.delete(oldest);
  }
}

export function snapshotUrl(runId: string, snapshotId: string): string {
  return `/api/runs/${encodeURIComponent(runId)}/snapshots/${encodeURIComponent(snapshotId)}`;
}

export function listSnapshots(
  runId: string,
  filters: { step?: number; nodeId?: string; kind?: ProbeKind } = {},
): Promise<SnapshotsResponse> {
  const params = new URLSearchParams();
  if (filters.step !== undefined) params.set("step", String(filters.step));
  if (filters.nodeId) params.set("node_id", filters.nodeId);
  if (filters.kind) params.set("kind", filters.kind);
  const query = params.toString();
  return apiGet<SnapshotsResponse>(
    `/api/runs/${encodeURIComponent(runId)}/snapshots${query ? `?${query}` : ""}`,
  );
}

/**
 * 快照一次写定、内容不可变（docs/02 §6.3）：同 id 直接命中本地缓存；
 * 并发请求共用同一个 promise；缓存被淘汰时带 `If-None-Match` 复用 304。
 */
export function fetchSnapshot(runId: string, snapshotId: string): Promise<SnapshotPayload> {
  const key = keyOf(runId, snapshotId);
  const cached = payloads.get(key);
  if (cached) {
    touch(key, cached);
    return Promise.resolve(cached);
  }
  const pending = inflight.get(key);
  if (pending) return pending;

  const task = load(runId, snapshotId, key).finally(() => inflight.delete(key));
  inflight.set(key, task);
  return task;
}

async function load(runId: string, snapshotId: string, key: string): Promise<SnapshotPayload> {
  const url = snapshotUrl(runId, snapshotId);
  const known = etags.get(key);
  const response = await request(url, known);
  if (response.status === 304) {
    // 本地已把 payload 淘汰、服务端确认内容未变：去掉条件头补取一次
    const retry = await fetch(url, { headers: { Accept: "application/json" } });
    if (!retry.ok) throw new Error(`${retry.status} ${retry.statusText}`);
    return store(key, retry);
  }
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return store(key, response);
}

function request(url: string, etag?: string): Promise<Response> {
  return fetch(url, {
    headers: {
      Accept: "application/json",
      ...(etag ? { "If-None-Match": etag } : {}),
    },
  });
}

async function store(key: string, response: Response): Promise<SnapshotPayload> {
  const etag = response.headers.get("ETag");
  if (etag) etags.set(key, etag);
  const payload = (await response.json()) as SnapshotPayload;
  touch(key, payload);
  return payload;
}

/** 换 run 时清空：快照 id 全局唯一，但仍避免跨 run 长期驻留。 */
export function clearSnapshotCache(): void {
  payloads.clear();
  etags.clear();
}
