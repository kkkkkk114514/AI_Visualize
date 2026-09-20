import type { ServerEvent } from "./types";

export type WsStatus = "connecting" | "connected" | "disconnected";

type EventListener = (event: ServerEvent) => void;
type StatusListener = (status: WsStatus) => void;
type ReconnectListener = () => void;

const MAX_BACKOFF_MS = 30_000;

class WsClient {
  private socket: WebSocket | null = null;
  private retryCount = 0;
  private reconnectTimer: number | null = null;
  private everConnected = false;
  private readonly eventListeners = new Set<EventListener>();
  private readonly statusListeners = new Set<StatusListener>();
  private readonly reconnectListeners = new Set<ReconnectListener>();
  private readonly subscriptions = new Set<string>();

  connect(): void {
    if (
      this.socket &&
      (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${window.location.host}/ws`);
    this.socket = socket;
    this.setStatus("connecting");

    socket.onopen = () => {
      this.retryCount = 0;
      this.setStatus("connected");
      // 断线期间服务端不重放事件：重连后由订阅方重新拉 REST 全量
      if (this.everConnected) {
        this.reconnectListeners.forEach((listener) => listener());
      }
      this.everConnected = true;
      this.subscriptions.forEach((runId) => this.send({ type: "subscribe", run_id: runId }));
    };
    socket.onmessage = (event: MessageEvent<string>) => {
      this.handleMessage(event.data);
    };
    socket.onclose = () => {
      this.socket = null;
      this.setStatus("disconnected");
      this.scheduleReconnect();
    };
    socket.onerror = () => {
      socket.close();
    };
  }

  onEvent(listener: EventListener): () => void {
    this.eventListeners.add(listener);
    return () => this.eventListeners.delete(listener);
  }

  onStatus(listener: StatusListener): () => void {
    this.statusListeners.add(listener);
    listener(this.currentStatus());
    return () => this.statusListeners.delete(listener);
  }

  onReconnect(listener: ReconnectListener): () => void {
    this.reconnectListeners.add(listener);
    return () => this.reconnectListeners.delete(listener);
  }

  /** 订阅某个 run 的事件（服务端只向订阅者推送 run 级事件）；重复订阅幂等。 */
  subscribe(runId: string): void {
    if (this.subscriptions.has(runId)) return;
    this.subscriptions.add(runId);
    this.send({ type: "subscribe", run_id: runId });
  }

  unsubscribe(runId: string): void {
    if (!this.subscriptions.delete(runId)) return;
    this.send({ type: "unsubscribe", run_id: runId });
  }

  private currentStatus(): WsStatus {
    if (this.socket?.readyState === WebSocket.OPEN) return "connected";
    if (this.socket?.readyState === WebSocket.CONNECTING) return "connecting";
    return "disconnected";
  }

  private setStatus(status: WsStatus): void {
    this.statusListeners.forEach((listener) => listener(status));
  }

  private handleMessage(data: string): void {
    let message: ServerEvent;
    try {
      message = JSON.parse(data) as ServerEvent;
    } catch {
      return;
    }
    if (message.type === "ping") {
      this.send({ type: "pong" });
      return;
    }
    this.eventListeners.forEach((listener) => listener(message));
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer !== null) return;
    const delay = Math.min(1000 * 2 ** this.retryCount, MAX_BACKOFF_MS);
    this.retryCount += 1;
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }

  private send(payload: Record<string, unknown>): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(payload));
    }
  }
}

export const wsClient = new WsClient();
