// One host action slot, one bounded latest Web receipt. No optimistic model
// state and no retry of an ambiguously accepted effect.
import { createEffect, createMemo, createSignal, onCleanup } from "solid-js";
import * as api from "./api.js";

export function newerId(a, b) {
  // IDs are opaque canonical decimal u64 strings, never JS numbers.
  if (!a) return b;
  if (!b) return a;
  return a.length === b.length ? (a > b ? a : b) : (a.length > b.length ? a : b);
}

export function createWork(key, control, unauthorized) {
  const [submitted, setSubmitted] = createSignal(null);
  const [result, setResult] = createSignal(null);
  const [pending, setPending] = createSignal(false);
  const [notice, setNotice] = createSignal("");
  const latest = createMemo(() => newerId(submitted(), control.value()?.web_action));
  let session;

  createEffect(() => {
    const credential = key();
    session = { credential, abort: new AbortController() };
    const owned = session;
    setSubmitted(null);
    setResult(null);
    setNotice("");
    setPending(false);
    onCleanup(() => owned.abort.abort());
  });

  createEffect(() => {
    const credential = key();
    const id = latest();
    setResult(null);
    if (!api.validKey(credential) || !id) return;
    const abort = new AbortController();
    let timer;
    const schedule = (delay) => {
      if (!abort.signal.aborted) timer = setTimeout(read, delay);
    };
    async function read() {
      try {
        const value = await api.actionResult(credential, abort.signal, id);
        if (abort.signal.aborted) return;
        if (value.id !== id) throw new Error("Mismatched action result.");
        setResult(value);
        if (value.state === "running") {
          schedule(document.visibilityState === "visible" ? 200 : 1000);
        } else {
          control.refresh();
        }
      } catch (error) {
        if (abort.signal.aborted) return;
        if (error.status === 401) unauthorized();
        else if (error.status === 404) {
          setNotice(error.message);
          if (submitted() === id) setSubmitted(null);
          control.refresh();
        } else {
          // Read-only observations may reconnect; never repeat POST.
          schedule(1000);
        }
      }
    }
    read();
    onCleanup(() => { abort.abort(); clearTimeout(timer); });
  });

  async function dispatch(action) {
    const owned = session;
    if (!owned || !api.validKey(owned.credential) || pending()) return;
    setPending(true);
    setNotice("");
    try {
      const receipt = await api.submit(owned.credential, owned.abort.signal, action);
      if (owned.abort.signal.aborted || session !== owned) return;
      if (action.kind !== "cancel") setSubmitted(receipt.id);
      control.refresh();
    } catch (error) {
      if (owned.abort.signal.aborted || session !== owned) return;
      if (error.status === 401) unauthorized();
      else {
        setNotice(error.status ? error.message : "Action acceptance is uncertain; refresh before retrying.");
        control.refresh();
      }
    } finally {
      if (session === owned) setPending(false);
    }
  }
  const busy = () => pending() || control.value()?.action != null || result()?.state === "running" || (submitted() != null && latest() === submitted() && result()?.id !== submitted());
  const ready = () => control.status() === "connected" && !!control.value() && !busy();
  return {
    result, pending, notice, ready, busy, dispatch,
    cancel: id => dispatch({kind: "cancel", id}),
  };
}
