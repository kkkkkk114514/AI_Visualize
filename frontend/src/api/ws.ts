import type { ServerEvent } from "./types";

export type WsStatus = "connecting" | "connected" | "disconnected";

type EventListener = (event: ServerEvent) => void;
type StatusListener = (status: WsStatus) => void;

const MAX_BACKOFF_MS = 30_000;

class WsClient {
  private socket: WebSocket | null = null;
  private retryCount = 0;
  private reconnectTimer: number | null = null;
  private readonly eventListeners = new Set<EventListener>();
  private readonly statusListeners = new Set<StatusListener>();

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
