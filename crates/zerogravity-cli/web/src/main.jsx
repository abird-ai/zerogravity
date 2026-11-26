import { render } from "solid-js/web";
import { createEffect, createMemo, createSignal, For, onCleanup, Show, Switch, Match } from "solid-js";
import { takeKey, storeKey, validKey } from "./api.js";
import { createControl } from "./control-state.js";
import { createWork } from "./work-state.js";
import { createInventory } from "./inventory-state.js";
import { Models, Devices } from "./library.jsx";
import { Profiles, Config, Logs } from "./settings.jsx";
import { QuickRun, OwnerActions, ServeActions, ActionPanel } from "./resident.jsx";
import { Performance } from "./performance.jsx";
import { modelName, providerLabel } from "./performance.js";
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
    <Show when={props.work && props.value.capabilities.serve}><ServeActions value={props.value} work={props.work}/></Show>
    <p class="note">The inference API is independent of this authenticated control listener.</p>
  </section>;
}
function Download(props) {
  const eta = row => {
    if (row.terminal || row.status !== "running" || row.logical_bytes_per_second <= 0 || !row.logical_bytes_per_second || row.total_expected_bytes == null) return "Unknown";
    const seconds = Math.ceil(Math.max(0, row.total_expected_bytes - row.total_bytes) / row.logical_bytes_per_second);
    return seconds < 60 ? seconds + " s" : seconds < 3600 ? Math.ceil(seconds/60) + " min" : (seconds/3600).toFixed(1) + " h";
  };
  const enabled = () => props.work?.connected() && !props.work.pending();
  const command = (row, command) => props.work.dispatch({kind:"download",operation_id:row.id,command});
  return <section class="panel">
    <div class="panel-title"><h2>Transfer</h2><Show when={props.value.download}>{row => <State text={row().status} good={row().status === "durable"} warn={row().status === "durability_unknown" || row().status === "failed"}/>}</Show></div>
    <Show when={props.value.download} fallback={<p class="subtle">{props.value.capabilities.acquisition ? "No download operation in this runtime." : "Acquisition is not compiled into this binary."}</p>}>{row =>
      <><Facts items={[["Operation", row().id], ["Phase", human(row().phase)], ["Files", count(row().completed_files) + " / " + count(row().selected_files)], ["Accepted", bytes(row().total_bytes)], ["Expected", bytes(row().total_expected_bytes)], ["Maximum", bytes(row().total_maximum_bytes)], ["Cached / resumed", bytes(row().cached_bytes) + " / " + bytes(row().resumed_bytes)], ["Observed checkpoint", bytes(row().checkpoint_bytes)], ["Accepted logical rate", row().logical_bytes_per_second == null ? "Unknown" : bytes(row().logical_bytes_per_second) + "/s"], ["Estimated remaining",eta(row())]]}/>
      <Show when={row().total_expected_bytes > 0}><progress aria-label="Accepted transfer bytes" value={Math.min(row().total_bytes, row().total_expected_bytes)} max={row().total_expected_bytes}/></Show>
      <Show when={props.work && !row().terminal}>
        <div class="form-actions">
          <button disabled={!enabled() || row().cancel_requested || row().phase === "publishing" || row().status !== "running"} onClick={() => command(row(),"pause")}>Pause download</button>
          <button disabled={!enabled() || row().cancel_requested || !["pause_requested","paused"].includes(row().status)} onClick={() => command(row(),"resume")}>Resume download</button>
          <button class="danger" disabled={!enabled() || row().cancel_requested} onClick={() => command(row(),"cancel")}>Cancel download</button>
        </div>
        <p class="field-hint">Commands target the observed transfer, not a new action. Pause/cancel requests are not completion acknowledgements.</p>
      </Show>
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
    <QuickRun value={props.value} work={props.work} selection={props.selection} select={props.select}/>
    <ActionPanel value={props.value} work={props.work}/>
    <div class="overview"><Serve value={props.value}/><Download value={props.value}/></div>
    <section class="panel">
      <div class="panel-title"><h2>Registered models</h2><button class="quiet" onClick={() => props.navigate("Running")}>Inspect</button></div>
      <Show when={props.value.runtime.models.length} fallback={<p class="subtle">No registered model owners. Load a model above.</p>}>
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
  return <Show when={models().length} fallback={<Empty title="No model owners">The runtime has not registered a model. Use Home to load or run a cached source.</Empty>}>
    <div class="split running-split">
      <section class="panel entries"><h2>Owners</h2><For each={models()}>{model =>
        <button classList={{ row: true, selected: model.registration_id === props.selection() }} onClick={() => props.select(model.registration_id)}>
          <span class="row-main">{modelName(model)}</span><State text={model.lifecycle} good={model.lifecycle === "ready"}/><span class="subtle">{providerLabel(model.capabilities?.provider)}<Show when={model.admitted_requests > 0}> · {count(model.admitted_requests)} admitted</Show></span>
        </button>
      }</For></section>
      <Show when={chosen()}>{model => <section class="panel">
        <h2>{modelName(model())}</h2>
        <Performance model={model()} bytes={bytes}/>
        <OwnerActions model={model()} chat={props.value.configuration.chat} work={props.work}/>
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
  const [previewTheme, setPreviewTheme] = createSignal(null);
  const [appearanceRequest, setAppearanceRequest] = createSignal(false);
  const systemTheme = window.matchMedia("(prefers-color-scheme: light)");
  const [systemLight, setSystemLight] = createSignal(systemTheme.matches);
  const appearanceChanged = event => setSystemLight(event.matches);
  systemTheme.addEventListener("change", appearanceChanged);
  onCleanup(() => systemTheme.removeEventListener("change", appearanceChanged));
  const deviceSource = createSignal("");
  const configTarget = createSignal("@active");
  let previousFocus;
  const lock = () => { storeKey(""); setKey(""); setInput(""); deviceSource[1](""); configTarget[1]("@active"); };
  const control = createControl(key, lock);
  const work = createWork(key, control, lock);
  const inventory = createInventory(key, control, view, lock, configTarget[0]);
  const navigate = name => { setView(name); setNav(false); };
  const openConfig = name => { setAppearanceRequest(false); configTarget[1](name); navigate("Config"); };
  const openAppearance = () => { openConfig("@active"); setAppearanceRequest(true); };
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
  const mode = () => previewTheme() ?? observation()?.configuration.theme ?? "auto";
  const light = () => mode() === "light" || (mode() === "auto" && systemLight());
  return <div classList={{ shell: true, light: light() }}>
    <header>
      <button class="brand" onClick={() => navigate("Home")}><span class="mark">Z</span>ZeroGravity</button>
      <button class="nav-toggle" onClick={() => setNav(!nav())} aria-expanded={nav()} aria-label="Toggle navigation">☰</button>
      <div class="header-context"><span class="subtle">Local control</span><Show when={observation()}>{row => <span class="profile">{row().configuration.profile ?? "Portable config"}{row().configuration.offline ? " · offline" : ""}</span>}</Show></div>
      <div class="header-actions"><button class="quiet" onClick={openPalette} aria-label="Open command palette">⌕ <span>Go to</span></button><button class="quiet theme-control" onClick={openAppearance} disabled={!observation()} aria-label="Appearance settings" title="Open appearance settings"><span>Theme: {mode() === "auto" ? "Auto" : mode() === "light" ? "Light" : "Dark"}</span></button><State text={control.status()} good={control.status() === "connected"}/></div>
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
      <div class="workspace-title"><div><div class="eyebrow">ZeroGravity / {view()}</div><h1>{view()}</h1></div><button class="quiet" onClick={() => { control.refresh(); control.refreshInventory(); }} disabled={!key()}>Refresh</button></div>
      <Show when={key()} fallback={
        <section class="panel unlock"><h2>Unlock local control</h2><p class="subtle">Open the private URL printed by the ZeroGravity host, or enter its 64-character control credential. Inference API keys do not grant control access.</p>
          <form onSubmit={event => { event.preventDefault(); if (validKey(input().trim())) { const token = input().trim(); storeKey(token); setKey(token); setInput(""); } }}>
            <label for="credential">Control credential</label><input id="credential" type="password" autocomplete="off" value={input()} onInput={event => setInput(event.currentTarget.value)}/><button type="submit" disabled={!validKey(input().trim())}>Unlock</button>
          </form>
        </section>}>
        <Show when={observation()} fallback={<Empty title={control.status() === "disconnected" ? "Control host unavailable" : "Connecting"}>Waiting for the original runtime observation. Refresh or reopen the host's private URL.</Empty>}>{row =>
          <><Show when={control.status() !== "connected"}><div role="status" class="notice">Connection interrupted — showing the last observation, not live state.</div></Show>
          <Switch>
            <Match when={view() === "Home"}><Home value={row()} navigate={navigate} work={work} selection={selection} select={select}/></Match>
            <Match when={view() === "Running"}><Running value={row()} selection={selection} select={select} work={work}/></Match>
            <Match when={view() === "Serve"}><Serve value={row()} work={work}/></Match>
            <Match when={view() === "Downloads"}><Download value={row()} work={work}/><Show when={row().capabilities.acquisition}><button class="quiet" onClick={() => navigate("Models")}>Acquire or resume from Models</button></Show></Match>
            <Match when={view() === "Profiles"}><Profiles inventory={inventory} work={work} open={openConfig}/></Match>
            <Match when={view() === "Config"}><Config inventory={inventory} work={work} target={configTarget[0]} open={openConfig} appearanceRequest={appearanceRequest} clearAppearance={()=>setAppearanceRequest(false)} previewTheme={setPreviewTheme}/></Match>
            <Match when={view() === "Models"}><Models value={row()} inventory={inventory} work={work} navigate={navigate}/></Match>
            <Match when={view() === "Devices"}><Devices value={row()} inventory={inventory} work={work} source={deviceSource}/></Match>
            <Match when={view() === "Logs"}><Logs inventory={inventory}/></Match>
          </Switch>
          <Show when={["Running","Serve","Models","Downloads","Devices","Profiles","Config"].includes(view())}><ActionPanel value={row()} work={work}/></Show></>
        }</Show>
      </Show>
    </main>
    <footer><span>Explicit local actions · no implicit acquisition</span><a class="license" href="/ui/licenses.txt" target="_blank" rel="noopener noreferrer">Licenses</a><span class="footer-detail">Events: {control.eventStatus()} · {control.updated() ? control.updated().toLocaleTimeString() : "No observation"}</span><Show when={key()}><button class="quiet" onClick={lock}>Lock</button></Show></footer>
    <Show when={palette()}><Palette navigate={navigate} close={closePalette}/></Show>
  </div>;
}
render(() => <App/>, document.getElementById("root"));
