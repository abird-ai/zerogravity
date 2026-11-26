// No provider credentials or inference bearer on this transport.
const STORAGE_KEY = "zg-control-session";
export const validKey = value => typeof value === "string" && /^[0-9a-f]{64}$/.test(value);

export function takeKey(location, history, storage) {
  // The private URL fragment never crosses the HTTP request boundary.
  const fragment = new URLSearchParams(location.hash.slice(1));
  if (fragment.has("control")) {
    const key = fragment.get("control");
    history.replaceState(null, "", location.pathname);
    try {
      if (validKey(key)) storage.setItem(STORAGE_KEY, key);
      else storage.removeItem(STORAGE_KEY);
    } catch { /* storage may be unavailable; in-memory use still works */ }
    return validKey(key) ? key : "";
  }
  try {
    const key = storage.getItem(STORAGE_KEY);
    return validKey(key) ? key : "";
  } catch { return ""; }
}

export function storeKey(key) {
  try {
    if (validKey(key)) sessionStorage.setItem(STORAGE_KEY, key);
    else sessionStorage.removeItem(STORAGE_KEY);
  } catch { /* private browsing may disable storage */ }
}

export class ControlError extends Error {
  constructor(status) {
    super(status === 401 ? "Control credential is invalid or expired."
      : status === 409 ? "An action is busy or the selected identity is stale. Refresh and try explicitly."
      : status === 400 ? "The control action or input was refused."
      : status === 404 ? "This action result is no longer retained."
      : "Control request failed (" + status + ").");
    this.status = status;
  }
}

async function request(path, key, signal, body) {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: { Authorization: "Bearer " + key, ...(body === undefined ? {} : {"Content-Type": "application/json"}) },
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: "no-store",
    credentials: "omit",
    redirect: "error",
    signal,
  });
  if (!response.ok) throw new ControlError(response.status);
  return response;
}

export async function snapshot(key, signal) {
  const response = await request("/zg/control/snapshot", key, signal);
  const value = await response.json();
  if (value.schema !== "zerogravity.control.v1") throw new Error("Unsupported control schema.");
  return value;
}

// Bounded framing; events invalidate observations rather than mutate a
// browser-owned runtime model. No bearer in an EventSource URL.
export function eventDecoder(onRefresh) {
  let pending = "";
  return chunk => {
    pending += chunk;
    if (pending.length > 16 * 1024) throw new Error("Control event frame exceeds bound.");
    let end;
    while ((end = pending.indexOf("\n\n")) >= 0) {
      const frame = pending.slice(0, end);
      pending = pending.slice(end + 2);
      if (frame.startsWith("event: refresh\n") || frame.startsWith("event: resync\n")) onRefresh();
    }
  };
}

export async function events(key, signal, onRefresh, onConnected) {
  const response = await request("/zg/events", key, signal);
  if (!response.body || !response.headers.get("content-type")?.startsWith("text/event-stream")) {
    throw new Error("Unexpected control event response.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const receive = eventDecoder(onRefresh);
  onConnected();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return;
      receive(decoder.decode(value, { stream: true }));
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

export async function submit(key, signal, action) {
  const response = await request("/zg/control/actions", key, signal, action);
  return response.json();
}

export async function actionResult(key, signal, id) {
  if (!/^(0|[1-9][0-9]{0,19})$/.test(id)) throw new Error("Invalid action identity.");
  const response = await request("/zg/control/actions/" + id, key, signal);
  return response.json();
}
