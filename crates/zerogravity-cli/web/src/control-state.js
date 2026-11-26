import { createEffect, createSignal, onCleanup } from "solid-js";
import * as api from "./api.js";

export const live = value => !!value && (
  value.action != null ||
  value.runtime.status.loaded_owners > 0 ||
  !["empty", "closed"].includes(value.runtime.status.phase) ||
  ["listening", "transitioning"].includes(value.serving.state) ||
  (value.download && !value.download.terminal)
);

// Exactly one in-flight snapshot, coalesced invalidations, and one stream.
// Reconnect and resync always reread authoritative owners.
export function createControl(key, unauthorized) {
  const [value, setValue] = createSignal(null);
  const [status, setStatus] = createSignal("locked");
  const [eventStatus, setEventStatus] = createSignal("disconnected");
  const [updated, setUpdated] = createSignal(null);
  let manual = () => {};
  createEffect(() => {
    const credential = key();
    setValue(null);
    setUpdated(null);
    setEventStatus("disconnected");
    if (!api.validKey(credential)) {
      setStatus("locked");
      manual = () => {};
      return;
    }
    setStatus("connecting");
    const abort = new AbortController();
    let inFlight = false;
    let pending = false;
    let reconnect;
    let closed = false;

    function failure(error) {
      if (closed || abort.signal.aborted) return;
      if (error.status === 401) {
        unauthorized();
      } else {
        setStatus("disconnected");
      }
    }
    async function refresh() {
      if (closed) return;
      if (inFlight) { pending = true; return; }
      inFlight = true;
      try {
        const next = await api.snapshot(credential, abort.signal);
        if (!closed) {
          setValue(next);
          setUpdated(new Date());
          setStatus("connected");
        }
      } catch (error) {
        failure(error);
      } finally {
        inFlight = false;
        if (!closed && pending) { pending = false; void refresh(); }
      }
    }

    async function stream() {
      if (closed) return;
      setEventStatus("connecting");
      try {
        await api.events(credential, abort.signal, refresh, () => setEventStatus("connected"));
      } catch (error) {
        failure(error);
      }
      if (!closed) {
        setEventStatus("reconnecting");
        reconnect = setTimeout(stream, 1000);
      }
    }

    // Direct HTTP inference can change counters without a control event.
    // Idle state follows events, not a continuous inventory poll.
    const timer = setInterval(() => {
      if (document.visibilityState === "visible" && (live(value()) || status() === "disconnected")) refresh();
    }, 2000);
    const visible = () => { if (document.visibilityState === "visible") refresh(); };
    document.addEventListener("visibilitychange", visible);
    manual = refresh;
    refresh();
    stream();
    onCleanup(() => {
      closed = true;
      abort.abort();
      clearInterval(timer);
      clearTimeout(reconnect);
      document.removeEventListener("visibilitychange", visible);
      manual = () => {};
    });
  });
  return { value, status, eventStatus, updated, refresh: () => manual() };
}
