// Cold observation controller: one in-flight read for the visible shelf,
// coalesced relevant owner invalidations and explicit refresh. No cache owner.
import { createEffect, createSignal, onCleanup } from "solid-js";
import * as api from "./api.js";

export function createInventory(key, control, view, unauthorized, configTarget) {
  const [values, setValues] = createSignal({});
  const [state, setState] = createSignal("idle");
  const [notice, setNotice] = createSignal("");

  createEffect(() => {
    key();
    setValues({});
    setState("idle");
    setNotice("");
  });

  createEffect(() => {
    const credential = key();
    const kind = {Models:"models", Devices:"devices", Profiles:"profiles", Config:"config", Logs:"logs"}[view()];
    const target = kind === "config" ? configTarget() : null;
    if (!api.validKey(credential) || !kind) return;
    const abort = new AbortController();
    if (kind === "config") setValues(previous => ({...previous, config:undefined}));
    let inFlight = false;
    let pending = false;
    async function refresh() {
      if (abort.signal.aborted) return;
      if (inFlight) { pending = true; return; }
      inFlight = true;
      setState("loading");
      setNotice("");
      try {
        // The host bounds concurrent cold reads. A previous shelf's aborted
        // connection may still be retiring; retry only a refused read, never
        // an action. Keep retries finite and scoped to the visible shelf.
        const read = () => kind === "config" ? api.config(credential, abort.signal, target) : api.inventory(credential, abort.signal, kind);
        let data;
        for (let attempt = 0; ; attempt++) {
          try { data = await read(); break; }
          catch (error) {
            if (abort.signal.aborted || error.status !== 503 || attempt === 2) throw error;
            await new Promise(resolve => setTimeout(resolve, 200));
            if (abort.signal.aborted) return;
          }
        }
        if (!abort.signal.aborted) {
          setValues(previous => ({...previous, [kind]:data}));
          setState("ready");
        }
      } catch (error) {
        if (!abort.signal.aborted) {
          if (error.status === 401) unauthorized();
          else {
            setState("failed");
            setNotice(error.message);
          }
        }
      } finally {
        inFlight = false;
        if (!abort.signal.aborted && pending) {
          pending = false;
          void refresh();
        }
      }
    }
    createEffect(() => {
      control.inventoryRevision(kind);
      void refresh();
    });
    onCleanup(() => abort.abort());
  });
  return { value: kind => values()[kind], state, notice };
}
