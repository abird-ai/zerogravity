import { render } from "solid-js/web";
import { createEffect, createMemo, createSignal, For, onCleanup, Show, Switch, Match } from "solid-js";
import { takeKey, storeKey, validKey } from "./api.js";
import { createControl } from "./control-state.js";
import "./styles.css";

const views = [
  ["Home", "Overview", "◉"], ["Running", "Registered models", "▤"], ["Serve", "Inference API", "↗"],
  ["Models", "Cached sources", "◇"], ["Downloads", "Transfer owner", "↓"], ["Devices", "Hardware", "▣"],
  ["Profiles", "Selected document", "▱"], ["Config", "Configuration", "⚙"], ["Logs", "Operations", "≡"],
];
const count = value => typeof value === "number" ? value.toLocaleString() : "—";
const human = value => value == null ? "Unknown" : String(value).replaceAll("_", " ");
const bytes = value => {
  if (value == null) return "Unknown";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let index = 0;
  while (value >= 1024 && index < units.length - 1) { value /= 1024; index++; }
  return value.toLocaleString(undefined, { maximumFractionDigits: 1 }) + " " + units[index];
};
function State(props) {
  return <span class="state"><span class={"dot " + (props.good ? "good" : props.warn ? "warn" : "")}/>{human(props.text)}</span>;
}
function Stat(props) {
  return <div class="stat"><div class="eyebrow">{props.label}</div><div class="number">{props.value}</div><div class="subtle">{props.detail}</div></div>;
}
function Empty(props) {
  return <div class="empty"><h2>{props.title}</h2><p>{props.children}</p></div>;
}
function Facts(props) {
  return <dl class="facts"><For each={props.items}>{([name, value]) => <><dt>{name}</dt><dd>{value}</dd></>}</For></dl>;
}
function Serve(props) {
  const serving = () => props.value.serving;
  return <section class="panel">
    <div class="panel-title"><h2>OpenAI-compatible API</h2><State text={serving().state} good={serving().state === "listening"}/></div>
    <Show when={serving().address} fallback={<p class="subtle">{serving().state === "unavailable" ? "Inference serving is not compiled into this binary." : "No listening endpoint is reported."}</p>}>
      <div class="endpoint">http://{serving().address}/v1</div>
      <Facts items={[["Listener generation", serving().generation], ["Accepted requests", count(serving().accepted_requests)], ["Connections", count(serving().active_connections)]]}/>
    </Show>
    <p class="note">The inference API is independent of this authenticated control listener.</p>
  </section>;
}
function Download(props) {
  return <section class="panel">
    <div class="panel-title"><h2>Transfer</h2><Show when={props.value.download}>{row => <State text={row().status} good={row().status === "durable"} warn={row().status === "durability_unknown"}/>}</Show></div>
    <Show when={props.value.download} fallback={<p class="subtle">{props.value.capabilities.acquisition ? "No download operation in this runtime." : "Acquisition is not compiled into this binary."}</p>}>{row =>
      <><Facts items={[["Operation", row().id], ["Phase", human(row().phase)], ["Files", count(row().completed_files) + " / " + count(row().selected_files)], ["Accepted", bytes(row().total_bytes)], ["Expected", bytes(row().total_expected_bytes)], ["Cached / resumed", bytes(row().cached_bytes) + " / " + bytes(row().resumed_bytes)], ["Accepted logical rate", row().logical_bytes_per_second == null ? "Unknown" : bytes(row().logical_bytes_per_second) + "/s"]]}/>
      <Show when={row().total_expected_bytes > 0}><progress aria-label="Accepted transfer bytes" value={Math.min(row().total_bytes, row().total_expected_bytes)} max={row().total_expected_bytes}/></Show>
      <Show when={row().cancel_requested}><p class="note">Cancellation requested; terminal outcome not inferred.</p></Show>
      <Show when={row().snapshot}><p class="mono wrap">{row().snapshot}</p></Show></>
    }</Show>
  </section>;
}
function Home(props) {
  const status = () => props.value.runtime.status;
  const scheduler = () => props.value.runtime.scheduler;
  return <>
    <div class="stats">
      <Stat label="Runtime" value={human(status().phase)} detail={status().failed ? "Owner reports failure" : "Local process"}/>
      <Stat label="Model owners" value={count(status().loaded_owners)} detail={count(status().loaded_models) + " registered aliases"}/>
      <Stat label="Active / queued" value={scheduler() ? count(scheduler().active) + " / " + count(scheduler().queued) : "—"} detail={scheduler() ? "Admitted scheduler" : "No scheduler installed"}/>
      <Stat label="Generated tokens" value={count(scheduler()?.output_tokens)} detail="Observed runtime counter"/>
    </div>
    <div class="overview"><Serve value={props.value}/><Download value={props.value}/></div>
    <section class="panel">
      <div class="panel-title"><h2>Registered models</h2><button class="quiet" onClick={() => props.navigate("Running")}>Inspect</button></div>
      <Show when={props.value.runtime.models.length} fallback={<p class="subtle">No registered model owners. Load through the native client to observe it here.</p>}>
        <For each={props.value.runtime.models}>{model => <div class="model-line"><span>{model.aliases.join(", ")}</span><State text={model.lifecycle} good={model.lifecycle === "ready"}/></div>}</For>
      </Show>
    </section>
  </>;
}
function Running(props) {
  const models = () => props.value.runtime.models;
  const chosen = createMemo(() => models().find(model => model.registration_id === props.selection()));
  createEffect(() => {
    if (!chosen()) props.select(models()[0]?.registration_id ?? "");
  });
  return <Show when={models().length} fallback={<Empty title="No model owners">The runtime has not registered a model. Native model actions remain available in the TUI.</Empty>}>
    <div class="split">
      <section class="panel entries"><h2>Owners</h2><For each={models()}>{model =>
        <button classList={{ row: true, selected: model.registration_id === props.selection() }} onClick={() => props.select(model.registration_id)}>
          <span class="row-main">{model.aliases.join(", ")}</span><State text={model.lifecycle} good={model.lifecycle === "ready"}/><span class="subtle">Registration {model.registration_id}</span>
        </button>
      }</For></section>
      <Show when={chosen()}>{model => <section class="panel">
        <h2>{model().aliases.join(", ")}</h2>
        <Facts items={[["Registration", model().registration_id], ["Lifecycle", human(model().lifecycle)], ["Admitted requests", count(model().admitted_requests)], ["Provider", model().capabilities?.provider ?? "Not reported"], ["Family", model().capabilities?.family ?? "Not reported"], ["Context capacity", count(model().capabilities?.context)], ["Host source reservation", bytes(model().capabilities?.source_host_reserved_bytes)]]}/>
      </section>}</Show>
    </div>
  </Show>;
}
function Palette(props) {
  const [query, setQuery] = createSignal("");
  const [cursor, setCursor] = createSignal(0);
  const matches = createMemo(() => views.filter(([name, description]) => (name + " " + description).toLowerCase().includes(query().toLowerCase())));
  let input;
  queueMicrotask(() => input?.focus());
  const choose = name => { props.navigate(name); props.close(); };
  return <div class="scrim" onClick={props.close}><div class="palette" role="dialog" aria-modal="true" aria-label="Command palette" onClick={event => event.stopPropagation()}>
    <input ref={input} aria-label="Search views" placeholder="Go to a view…" value={query()} onInput={event => { setQuery(event.currentTarget.value); setCursor(0); }} onKeyDown={event => {
      if (event.key === "ArrowDown") { event.preventDefault(); setCursor(Math.min(cursor() + 1, Math.max(0, matches().length - 1))); }
      else if (event.key === "ArrowUp") { event.preventDefault(); setCursor(Math.max(0, cursor() - 1)); }
      else if (event.key === "Enter" && matches()[cursor()]) { event.preventDefault(); choose(matches()[cursor()][0]); }
      else if (event.key === "Tab") event.preventDefault();
    }}/>
    <div class="palette-list"><For each={matches()}>{([name, description, glyph], index) => <button tabindex="-1" classList={{ selected: cursor() === index() }} onClick={() => choose(name)}><span aria-hidden="true">{glyph}</span><span>{name}<small>{description}</small></span></button>}</For>
    <Show when={!matches().length}><p class="subtle">No matching view.</p></Show></div>
    <div class="subtle palette-footer">↑ ↓ select · Enter open · Esc close</div>
  </div></div>;
}
function App() {
  const [key, setKey] = createSignal(takeKey(location, history, sessionStorage));
  const [input, setInput] = createSignal("");
  const [view, setView] = createSignal("Home");
  const [palette, setPalette] = createSignal(false);
  const [nav, setNav] = createSignal(false);
  const [selection, select] = createSignal("");
  const [light, setLight] = createSignal(false);
  let previousFocus;
  const lock = () => { storeKey(""); setKey(""); setInput(""); };
  const control = createControl(key, lock);
  const navigate = name => { setView(name); setNav(false); };
  const openPalette = () => { previousFocus = document.activeElement; setPalette(true); };
  const closePalette = () => { setPalette(false); queueMicrotask(() => previousFocus?.focus()); };
  const keyboard = event => {
    if (event.key === "F1" || (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === "p")) {
      event.preventDefault();
      if (palette()) closePalette(); else openPalette();
    } else if (event.key === "Escape") {
      if (palette()) { event.preventDefault(); closePalette(); }
      else setNav(false);
    }
  };
  const newLaunch = () => {
    // Reopening the same private URL can be a same-document fragment
    // navigation. Consume it exactly as on initial launch.
    if (new URLSearchParams(location.hash.slice(1)).has("control")) {
      setInput("");
      setKey(takeKey(location, history, sessionStorage));
    }
  };
  document.addEventListener("keydown", keyboard);
  window.addEventListener("hashchange", newLaunch);
  onCleanup(() => {
    document.removeEventListener("keydown", keyboard);
    window.removeEventListener("hashchange", newLaunch);
  });
  const observation = control.value;
  return <div classList={{ shell: true, light: light() }}>
    <header>
      <button class="brand" onClick={() => navigate("Home")}><span class="mark">Z</span>ZeroGravity</button>
      <button class="nav-toggle" onClick={() => setNav(!nav())} aria-expanded={nav()} aria-label="Toggle navigation">☰</button>
      <div class="header-context"><span class="subtle">Local control</span><Show when={observation()}>{row => <span class="profile">{row().configuration.profile ?? "Portable config"}{row().configuration.offline ? " · offline" : ""}</span>}</Show></div>
      <div class="header-actions"><button class="quiet" onClick={openPalette} aria-label="Open command palette">⌕ <span>Go to</span></button><button class="quiet" onClick={() => setLight(!light())} aria-label="Toggle theme">{light() ? "◐" : "☼"}</button><State text={control.status()} good={control.status() === "connected"}/></div>
    </header>
    <aside classList={{ rail: true, "nav-open": nav() }} aria-label="Workspace navigation">
      <div class="eyebrow rail-heading">Workspace</div>
      <nav><For each={views}>{([name, description, glyph]) =>
        <button title={name} aria-current={view() === name ? "page" : undefined} classList={{ selected: view() === name }} onClick={() => navigate(name)}>
          <span class="glyph" aria-hidden="true">{glyph}</span><span class="nav-label">{name}<small>{description}</small></span>
        </button>
      }</For></nav>
      <div class="rail-bottom"><span class="dot"/> Same-process runtime</div>
    </aside>
    <main>
      <div class="workspace-title"><div><div class="eyebrow">ZeroGravity / {view()}</div><h1>{view()}</h1></div><button class="quiet" onClick={control.refresh} disabled={!key()}>Refresh</button></div>
      <Show when={key()} fallback={
        <section class="panel unlock"><h2>Unlock local control</h2><p class="subtle">Open the private URL printed by the ZeroGravity host, or enter its 64-character control credential. Inference API keys do not grant control access.</p>
          <form onSubmit={event => { event.preventDefault(); if (validKey(input().trim())) { const token = input().trim(); storeKey(token); setKey(token); setInput(""); } }}>
            <label for="credential">Control credential</label><input id="credential" type="password" autocomplete="off" value={input()} onInput={event => setInput(event.currentTarget.value)}/><button type="submit" disabled={!validKey(input().trim())}>Unlock</button>
          </form>
        </section>}>
        <Show when={observation()} fallback={<Empty title={control.status() === "disconnected" ? "Control host unavailable" : "Connecting"}>Waiting for the original runtime observation. Refresh or reopen the host's private URL.</Empty>}>{row =>
          <><Show when={control.status() !== "connected"}><div role="status" class="notice">Connection interrupted — showing the last observation, not live state.</div></Show>
          <Switch>
            <Match when={view() === "Home"}><Home value={row()} navigate={navigate}/></Match>
            <Match when={view() === "Running"}><Running value={row()} selection={selection} select={select}/></Match>
            <Match when={view() === "Serve"}><Serve value={row()}/></Match>
            <Match when={view() === "Downloads"}><Download value={row()}/></Match>
            <Match when={view() === "Profiles" || view() === "Config"}>
              <section class="panel"><h2>Active configuration</h2><Facts items={[["Selection", row().configuration.profile ?? "Portable file"], ["Acquisition", row().configuration.offline ? "Offline" : "Explicit consent required"]]}/><p class="note">Editing and profile management are available in the native client. This browser shell does not expose raw documents.</p></section>
            </Match>
            <Match when={view() === "Models"}><Empty title="Model management">{row().capabilities.cache ? "Cached source inventory and mutations are not yet exposed in this browser slice. Use Models in the native client." : "Cache support is not compiled into this binary."}</Empty></Match>
            <Match when={view() === "Devices"}><Empty title="Device inventory">Physical device management remains in the native client. No backend or execution eligibility is inferred here.</Empty></Match>
            <Match when={view() === "Logs"}><Empty title="Operational logs">The native client's bounded operational log is not yet exposed to this browser. Model output is never used as log history.</Empty></Match>
          </Switch></>
        }</Show>
      </Show>
    </main>
    <footer><span>Read-only control shell · no implicit execution</span><a class="license" href="/ui/licenses.txt" target="_blank" rel="noopener noreferrer">Licenses</a><span class="footer-detail">Events: {control.eventStatus()} · {control.updated() ? control.updated().toLocaleTimeString() : "No observation"}</span><Show when={key()}><button class="quiet" onClick={lock}>Lock</button></Show></footer>
    <Show when={palette()}><Palette navigate={navigate} close={closePalette}/></Show>
  </div>;
}
render(() => <App/>, document.getElementById("root"));
