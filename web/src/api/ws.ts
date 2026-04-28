export function connectJsonWebSocket(path: string, onMessage: (value: unknown) => void): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${window.location.host}${path}`);
  socket.onmessage = (event) => {
    try {
      onMessage(JSON.parse(event.data));
    } catch {
      onMessage(event.data);
    }
  };
  return socket;
}
