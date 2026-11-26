// Cold observation controller: one in-flight read for the visible shelf,
// coalesced relevant owner invalidations and explicit refresh. No cache owner.
import { createEffect, createSignal, onCleanup } from "solid-js";
import * as api from "./api.js";

export function createInventory(key, control, view, unauthorized) {
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
    const kind = {Models:"models", Devices:"devices"}[view()];
    if (!api.validKey(credential) || !kind) return;
    const abort = new AbortController();
    let inFlight = false;
    let pending = false;
    async function refresh() {
      if (abort.signal.aborted) return;
      if (inFlight) { pending = true; return; }
      inFlight = true;
      setState("loading");
      setNotice("");
      try {
        const data = await api.inventory(credential, abort.signal, kind);
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
