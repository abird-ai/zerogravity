// Original resident action grammar and exact observed IDs; the server
// validates all source and execution policy again at admission.
import { createEffect, createMemo, createSignal, For, Show } from "solid-js";

const textBytes = value => new TextEncoder().encode(value).length;
const promptValid = value => !!value.trim() && textBytes(value) <= 32768 && !value.includes("\0");

function ConfirmButton(props) {
  const [intent, setIntent] = createSignal(null);
  return <div class="confirm-action">
    <Show when={intent()} fallback={
      <button class="quiet danger" disabled={!props.work.ready() || props.disabled} onClick={() => setIntent({request: props.request(), label: props.label})}>{props.children}</button>
    }>{pending =>
      <div class="confirmation" role="group" aria-label="Confirm action">
        <p>{pending().label}</p>
        <div class="form-actions">
          <button class="danger" disabled={!props.work.ready()} onClick={() => {
            const request = pending().request;
            setIntent(null);
            props.work.dispatch(request);
          }}>Confirm</button>
          <button class="quiet" onClick={() => setIntent(null)}>Cancel</button>
        </div>
      </div>
    }</Show>
  </div>;
}

function PromptField(props) {
  const [prompt, setPrompt] = createSignal("");
  const [chat, setChat] = createSignal(true);
  const configuredChat = createMemo(() => props.defaultChat);
  createEffect(() => setChat(configuredChat()));
  return <>
    <label class="field-label" for={props.id}>Prompt</label>
    <textarea id={props.id} rows="4" placeholder="Enter a prompt…" value={prompt()} onInput={event => setPrompt(event.currentTarget.value)}/>
    <div class="form-actions">
      <label class="check"><input type="checkbox" checked={chat()} onChange={event => setChat(event.currentTarget.checked)}/> Chat template</label>
      <span class="subtle">{textBytes(prompt()).toLocaleString()} / 32,768 bytes</span>
      <button class="primary" disabled={!props.work.ready() || !promptValid(prompt()) || props.disabled} onClick={() => props.submit(prompt(), chat())}>{props.label}</button>
    </div>
  </>;
}

export function QuickRun(props) {
  const [source, setSource] = createSignal("");
  const ready = createMemo(() => props.value.runtime.models.filter(model => model.lifecycle === "ready"));
  const chosen = createMemo(() => ready().find(model => model.registration_id === props.selection()) ?? ready()[0]);
  createEffect(() => {
    if (chosen() && props.selection() !== chosen().registration_id) props.select(chosen().registration_id);
  });
  const empty = () => props.value.runtime.status.phase === "empty";
  return <section class="panel">
    <div class="panel-title"><h2>Quick run</h2><span class="subtle">Local / cached only</span></div>
    <Show when={empty()} fallback={
      <Show when={chosen()} fallback={<p class="subtle">No ready registration. Wait for the current route transition or retire it first.</p>}>{model =>
        <>
          <label for="quick-owner" class="field-label">Registered owner</label>
          <select id="quick-owner" value={model().registration_id} onChange={event => props.select(event.currentTarget.value)}>
            <For each={ready()}>{owner => <option value={owner.registration_id}>{owner.aliases.join(", ")} · #{owner.registration_id}</option>}</For>
          </select>
          <PromptField id="quick-prompt" work={props.work} defaultChat={props.value.configuration.chat} label="Run" submit={(prompt, chat) => {
            const owner = model();
            props.work.dispatch({kind:"invoke", alias:owner.aliases[0], registration_id:owner.registration_id, prompt, chat});
          }}/>
        </>
      }</Show>
    }>
      <label for="quick-source" class="field-label">Model source</label>
      <input id="quick-source" value={source()} onInput={event => setSource(event.currentTarget.value)} placeholder="Cached model, sha256:… or ID=MODEL"/>
      <p class="field-hint">Leave blank to load the selected configuration's routes. No automatic download or arbitrary browser file paths.</p>
      <div class="form-actions"><button disabled={!props.work.ready() || textBytes(source()) > 4096} onClick={() => props.work.dispatch({kind:"load",source:source()})}>Load model</button></div>
      <PromptField id="quick-prompt" work={props.work} defaultChat={props.value.configuration.chat} label="Load and run" disabled={textBytes(source()) > 4096} submit={(prompt, chat) => props.work.dispatch({kind:"run_source",source:source(),prompt,chat})}/>
    </Show>
  </section>;
}

export function OwnerActions(props) {
  const request = () => ({alias:props.model.aliases[0],registration_id:props.model.registration_id});
  return <Show when={props.model.lifecycle === "ready"}>
    <PromptField id="owner-prompt" work={props.work} defaultChat={props.chat} label="Run" submit={(prompt, chat) => props.work.dispatch({kind:"invoke",...request(),prompt,chat})}/>
    <ConfirmButton work={props.work} request={() => ({kind:"unload",...request(),confirm:true})}
      label={"Retire registration " + props.model.registration_id + " and all aliases (" + props.model.aliases.join(", ") + ")? The inference listener stays unchanged."}>Unload owner</ConfirmButton>
  </Show>;
}

export function ServeActions(props) {
  const [address, setAddress] = createSignal("");
  return <Show when={props.value.serving.state === "idle"} fallback={
    <Show when={["listening","finished"].includes(props.value.serving.state)}>
      <ConfirmButton work={props.work} request={() => ({kind:"stop",generation:props.value.serving.generation,confirm:true})}
        label={"Stop listener generation " + props.value.serving.generation + " at " + props.value.serving.address + "? Loaded models are retained."}>Stop inference API</ConfirmButton>
    </Show>
  }>
    <label class="field-label" for="listen-address">Listen address</label>
    <input id="listen-address" value={address()} onInput={event => setAddress(event.currentTarget.value)} placeholder="127.0.0.1:8000 (or blank for profile)"/>
    <p class="field-hint">Numeric loopback only. Port 0 selects a free port. Inference authentication comes from the host, never this browser.</p>
    <div class="form-actions">
      <button class="primary" disabled={!props.work.ready() || props.value.runtime.status.loaded_owners === 0 || textBytes(address()) > 128} onClick={() => props.work.dispatch({kind:"start",address:address()})}>Start inference API</button>
      <Show when={props.value.runtime.status.loaded_owners === 0}><span class="subtle">Load a model first.</span></Show>
    </div>
  </Show>;
}

export function ActionPanel(props) {
  const result = props.work.result;
  const active = () => props.value.action;
  return <Show when={active() || result() || props.work.notice() || props.work.pending()}>
    <section class="panel action-panel" aria-label="Control action" data-action-id={active()?.id ?? result()?.id}>
      <div class="panel-title">
        <h2>{active()?.title ?? result()?.title ?? "Control action"}</h2>
        <span class="subtle" role="status">{active() ? active().origin + " #" + active().id + " · " + (active().cancel_requested ? "cancellation requested" : "in progress") : result() ? "Web #" + result().id + " · " + result().state : props.work.pending() ? "Submitting" : "No active action"}</span>
      </div>
      <Show when={active()?.origin === "web"}>
        <button class="quiet" disabled={props.work.pending() || active().cancel_requested} onClick={() => props.work.cancel(active().id)}>Request cancellation</button>
      </Show>
      <Show when={active()?.origin === "native"}><p class="note">The native client owns this action. Its private response is not displayed here.</p></Show>
      <Show when={props.work.notice()}><p class="notice" role="alert">{props.work.notice()}</p></Show>
      <Show when={active() && result() && active().id !== result().id}><div class="eyebrow response-label">Previous Web action #{result().id}</div></Show>
      <Show when={result()?.message}><p class="action-message">{result().message}</p></Show>
      <Show when={result()?.response}>{response =>
        <>
          <div class="eyebrow response-label">{response().owner ? "Response · " + response().owner.alias + " · #" + response().owner.registration_id : "Awaiting admitted output"}</div>
          <pre class="response-text">{response().text}</pre>
          <Show when={response().clipped}><p class="note">Display prefix is bounded; the original output stream may contain more text.</p></Show>
          <Show when={response().usage}>{usage => <p class="subtle">{usage().completion_tokens} output tokens · {usage().prompt_tokens} prompt tokens · {usage().finish_reason}</p>}</Show>
        </>
      }</Show>
    </section>
  </Show>;
}
