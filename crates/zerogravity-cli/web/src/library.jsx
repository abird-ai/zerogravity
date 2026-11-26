// Model/cache/device presentation over original, separately observed owners.
// Selection and drafts are client-local; all effects use the shared slot.
import { createEffect, createMemo, createSignal, For, Show, Switch, Match } from "solid-js";
import { ConfirmButton, PromptField } from "./resident.jsx";

const digestValid = value => /^[0-9a-f]{64}$/.test(value);
const aliasValid = value => /^[a-z0-9](?:[a-z0-9.-]{0,238}[a-z0-9])?$/.test(value);
const textBytes = value => new TextEncoder().encode(value).length;
const amount = value => value == null ? "Unknown" : value.toLocaleString() + " bytes";
function Facts(props) {
  return <dl class="facts"><For each={props.items}>{([name, value]) => <><dt>{name}</dt><dd>{value}</dd></>}</For></dl>;
}
function InventoryNotice(props) {
  return <>
    <Show when={props.inventory.state() === "loading"}><p class="subtle inventory-notice" role="status">Refreshing owner observations…</p></Show>
    <Show when={props.inventory.notice()}><p class="notice" role="alert">{props.inventory.notice()} — refresh to observe again.</p></Show>
  </>;
}
function CachedActions(props) {
  const [alias, setAlias] = createSignal("");
  const [previewId, setPreviewId] = createSignal("");
  const [execution, setExecution] = createSignal(null);
  const choiceKey = choice => JSON.stringify(choice.execution);
  const choices = createMemo(() => {
    const rows = props.executionChoices ?? [{label:"Use profile default",execution:null}];
    const selected = execution();
    return selected && !rows.some(row => choiceKey(row) === choiceKey(selected))
      ? [...rows, {...selected,label:selected.label+" · not currently observed"}] : rows;
  });
  const current = () => execution() ? choiceKey(execution()) : "null";
  const placement = () => execution()?.execution ? {execution:execution().execution} : {};
  const ready = () => props.ready && props.work.ready();
  const empty = () => props.value.runtime.status.phase === "empty";
  const digest = () => props.digest;
  const source = () => "sha256:" + digest();
  const request = kind => ({kind, digest:digest()});
  const preview = () => props.work.result()?.id === previewId() && props.work.result()?.state === "completed" && !props.work.result()?.cancel_requested;
  return <>
    <div class="execution-field">
      <label for="model-execution">Execution</label>
      <select id="model-execution" disabled={!ready() || !empty()} value={current()} onChange={event => {
        const choice = choices().find(row => choiceKey(row) === event.currentTarget.value);
        if (choice) setExecution(choice.execution === null ? null : choice);
      }}>
        <For each={choices()}>{choice => <option value={choiceKey(choice)}>{choice.execution === null ? props.value.execution_default ?? "Use profile" : choice.label}</option>}</For>
      </select>
    </div>
    <div class="form-actions">
      <button disabled={!ready()} onClick={() => props.work.dispatch(request("inspect"))}>Inspect</button>
      <button disabled={!ready()} onClick={() => props.work.dispatch(request("verify"))}>Verify payloads</button>
      <button disabled={!ready() || !empty()} onClick={() => props.work.dispatch({kind:"load",source:source(),...placement()})}>Load</button>
      <button disabled={!ready() || !empty() || !props.value.capabilities.serve} onClick={async () => {
        const receipt = await props.work.dispatch({kind:"load",source:source(),...placement()});
        if (receipt) props.navigate("Serve");
      }}>Load for serving</button>
    </div>
    <Show when={!empty()}><p class="field-hint">Retire the current route set or invoke its exact owner from Running.</p></Show>
    <PromptField id="cached-prompt" work={props.work} defaultChat={props.value.configuration.chat} label="Load and run" disabled={!props.ready || !empty()} submit={(prompt,chat) => props.work.dispatch({kind:"run_source",source:source(),prompt,chat,...placement()})}/>
    <label class="field-label" for="cache-alias">Add a by-name alias</label>
    <input id="cache-alias" placeholder="my-model" value={alias()} onInput={event => setAlias(event.currentTarget.value)}/>
    <div class="form-actions">
      <button disabled={!ready() || !aliasValid(alias())} onClick={() => props.work.dispatch({...request("alias"),name:alias()})}>Create alias</button>
      <span class="subtle">Existing aliases are never retargeted.</span>
    </div>
    <div class="form-actions">
      <button disabled={!ready()} onClick={async () => {
        setPreviewId("");
        const receipt = await props.work.dispatch(request("removal_preview"));
        if (receipt) setPreviewId(receipt.id);
      }}>Preview removal</button>
    </div>
    <Show when={preview()}>
      <p class="field-hint">Preview completed. Removal must revalidate this exact digest and its original cache readers/references.</p>
      <ConfirmButton work={props.work} disabled={!props.ready || !preview()} request={() => ({...request("remove"),confirm:true})}
        label={"Remove exact cached snapshot sha256:" + digest() + "? Active readers and references refuse removal. No references are deleted."}>Remove snapshot</ConfirmButton>
    </Show>
  </>;
}
function PullForm(props) {
  const [selection, setSelection] = createSignal("");
  const [preview, setPreview] = createSignal(null);
  const [source, setSource] = props.source;
  // An explicitly chosen catalog recipe fills only the draft.
  createEffect(() => {
    props.recipeRevision();
    setSelection("");
    setPreview(null);
  });
  const ready = () => props.work.ready() && !props.value.configuration.offline;
  const valid = () => !!source().trim() && textBytes(source()) <= 4096 && textBytes(selection()) <= 65536;
  const matching = () => preview() && props.work.result()?.id === preview().id && props.work.result()?.state === "completed" && !props.work.result()?.cancel_requested;
  return <section class="panel">
    <div class="panel-title"><h2>Acquire a model</h2><span class="subtle">{props.value.configuration.offline ? "Offline" : "Explicit consent"}</span></div>
    <Show when={props.value.capabilities.acquisition} fallback={<p class="subtle">Acquisition is not compiled into this binary.</p>}>
      <label class="field-label" for="pull-source">Pinned source or approved recipe</label>
      <input id="pull-source" placeholder="hf:OWNER/REPO@FULL_COMMIT or catalog target" value={source()} onInput={event => { setSource(event.currentTarget.value); setPreview(null); }}/>
      <label class="field-label" for="pull-selection">Exact files or quantization</label>
      <textarea id="pull-selection" rows="4" placeholder={"model.gguf\nconfig.json\n—or quant:Q4_K_M"} value={selection()} onInput={event => { setSelection(event.currentTarget.value); setPreview(null); }}/>
      <p class="field-hint">For an approved recipe leave selection empty. For a provider, use one exact relative file per line or quant:NAME, with the full immutable commit. No implicit download, quota increase, eviction or load.</p>
      <div class="form-actions"><button disabled={!props.work.ready() || !valid()} onClick={async () => {
        const intent = {source:source().trim(),selection:selection()};
        setPreview(null);
        const receipt = await props.work.dispatch({kind:"pull_preview",...intent});
        // If the draft changed while the preview was running, do not offer
        // consent for either the new draft or an unseen old one.
        if (receipt && source().trim() === intent.source && selection() === intent.selection) setPreview({id:receipt.id,intent});
      }}>Preview pull</button></div>
      <Show when={matching()}>
        <div class="confirmation" role="group" aria-label="Confirm acquisition">
          <pre class="intent-preview">{props.work.result().message}</pre>
          <div class="form-actions">
            <button class="primary" disabled={!ready()} onClick={async () => {
              const intent = preview().intent;
              setPreview(null);
              const receipt = await props.work.dispatch({kind:"pull",...intent,confirm:true});
              if (receipt) props.navigate("Downloads");
            }}>Confirm pull</button>
            <button class="quiet" onClick={() => setPreview(null)}>Cancel</button>
          </div>
          <Show when={props.value.configuration.offline}><p class="field-hint">The host is offline. Acquisition cannot be admitted.</p></Show>
        </div>
      </Show>
    </Show>
  </section>;
}
export function Models(props) {
  const [shelf,setShelf] = createSignal("cached");
  const [selected,setSelected] = createSignal("");
  const [query,setQuery] = createSignal("");
  const source = createSignal("");
  const [recipeRevision,bumpRecipe] = createSignal(0);
  const inventory = () => props.inventory.value("models");
  const key = row => row.digest ?? row.slug ?? row.key;
  const title = row => row.slug ?? row.repository ?? "Partial checkpoint";
  const rows = createMemo(() => (inventory()?.[shelf()] ?? []).filter(row => {
    const needle = query().trim().toLowerCase();
    return !needle || (title(row) + " " + key(row) + " " + (row.repository ?? "")).toLowerCase().includes(needle);
  }));
  const chosen = createMemo(() => rows().find(row => key(row) === selected()));
  createEffect(() => {
    if (!chosen()) setSelected(rows()[0] ? key(rows()[0]) : "");
  });
  const observed = () => props.inventory.state() === "ready" && !!inventory();
  return <>
    <InventoryNotice inventory={props.inventory}/>
    <Show when={inventory()?.warning}><p class="notice">{inventory().warning}</p></Show>
    <div class="split">
      <section class="panel entries">
        <div class="panel-title"><h2>Model inventory</h2></div>
        <div class="shelves" role="group" aria-label="Model shelf"><For each={["cached","catalog","partial"]}>{name => <button classList={{selected:shelf() === name}} aria-pressed={shelf() === name} onClick={() => { setShelf(name); setSelected(""); setQuery(""); }}>{name[0].toUpperCase()+name.slice(1)}</button>}</For></div>
        <label class="field-label" for="model-search">Search sources</label>
        <input id="model-search" value={query()} onInput={event => setQuery(event.currentTarget.value)} placeholder="Name or exact identity"/>
        <Show when={rows().length} fallback={<p class="field-hint">{inventory() ? "No matching observations." : "Waiting for cold model inventory."}</p>}>
          <For each={rows()}>{row =>
            <button classList={{row:true,selected:selected() === key(row)}} onClick={() => setSelected(key(row))} aria-pressed={selected() === key(row)}>
              <span class="row-main">{title(row)}</span>
              <span class="subtle mono wrap">{row.digest ?? row.key ?? row.source_status}</span>
              <Show when={row.bytes != null}><span class="subtle">{amount(row.bytes)}{row.total != null ? " / " + amount(row.total) : ""}</span></Show>
            </button>
          }</For>
        </Show>
      </section>
      <Show when={chosen() && selected()} keyed>{identity =>
        <section class="panel">
          <h2>{title(chosen())}</h2>
          <Switch>
            <Match when={shelf() === "cached"}>
              <Facts items={[["Snapshot", "sha256:"+identity],["Provider", chosen().provider],["Repository",chosen().repository],["Revision",chosen().revision],["Files",chosen().files],["Bytes",amount(chosen().bytes)],["Verification","Metadata only; payloads not verified by listing"]]}/>
              <CachedActions digest={identity} ready={observed()} value={props.value} work={props.work} navigate={props.navigate} executionChoices={inventory()?.execution_choices}/>
            </Match>
            <Match when={shelf() === "catalog"}>
              <Facts items={[["Target",identity],["Task",chosen().task],["Repository",chosen().repository ?? "Unverified"],["Source layout",chosen().source_layout],["Source status",chosen().source_status],["Execution",chosen().execution_status]]}/>
              <p class="note">{chosen().payload_hold ? "Payload hold; no catalog acquisition." : chosen().note}</p>
              <Show when={chosen().recipe}>{recipe =>
                <><Facts items={[["Approved snapshot",recipe().snapshot],["Bytes",amount(recipe().bytes)],["Pinned revision",recipe().revision]]}/>
                <button disabled={!observed()} onClick={() => { source[1](identity); bumpRecipe(n=>n+1); }}>Use approved recipe</button></>
              }</Show>
            </Match>
            <Match when={shelf() === "partial"}>
              <Facts items={[["Checkpoint key",identity],["Observed prefix",amount(chosen().bytes)],["Expected",amount(chosen().total)]]}/>
              <p class="note">Checkpoint metadata is not a verified or inactive attempt. Repeat the original pinned pull intent below to revalidate and resume; the key itself cannot authorize a transfer.</p>
            </Match>
          </Switch>
        </section>
      }</Show>
    </div>
    <PullForm value={props.value} work={props.work} source={source} recipeRevision={recipeRevision} navigate={props.navigate}/>
  </>;
}
export function Devices(props) {
  const [selected,setSelected] = createSignal("");
  const [source,setSource] = props.source;
  const inventory = () => props.inventory.value("devices");
  const rows = () => inventory()?.devices ?? [];
  const chosen = createMemo(() => rows().find(row => row.id === selected()));
  createEffect(() => { if (!chosen()) setSelected(rows()[0]?.id ?? ""); });
  const exact = () => source().trim().replace(/^sha256:/,"");
  const ready = () => props.value.capabilities.cache && props.inventory.state() === "ready" && props.work.ready() && digestValid(exact());
  const request = kind => ({kind,digest:exact(),device:selected()});
  return <>
    <InventoryNotice inventory={props.inventory}/>
    <div class="split">
      <section class="panel entries">
        <div class="panel-title"><h2>Physical inventory</h2></div>
        <Show when={inventory()}><p class="subtle">Observed {new Date(Number(inventory().observed_unix_ms)).toLocaleString()}</p></Show>
        <For each={rows()}>{row =>
          <button classList={{row:true,selected:selected() === row.id}} aria-pressed={selected() === row.id} onClick={() => setSelected(row.id)}>
            <span class="row-main">{row.id}</span><span class="subtle">{row.class} · {row.vendor}</span>
          </button>
        }</For>
        <Show when={!rows().length}><p class="field-hint">No physical observations reported.</p></Show>
      </section>
      <Show when={chosen()}>{row =>
        <section class="panel">
          <h2>{row().id}</h2>
          <Facts items={[["Class",row().class],["Vendor",row().vendor],["Backend observations",row().backends.join(", ") || "None"],["Capacity",amount(row().ceiling)],["Available",amount(row().available)],["Provenance",row().provenance],["Exclusion",row().exclusion ?? "No observed hardware exclusion; not a source-qualified capability"]]}/>
          <label class="field-label" for="device-source">Exact cached snapshot digest</label>
          <input id="device-source" placeholder="sha256:…" value={source()} onInput={event=>setSource(event.currentTarget.value)}/>
          <p class="field-hint">Choose a cached model from Models or enter its full digest. The selected profile and original resident planner determine eligibility; no implicit fallback or persistent device change.</p>
          <Show when={!props.value.capabilities.cache}><p class="field-hint">Cache-backed device actions are unavailable in this build.</p></Show>
          <div class="form-actions">
            <button disabled={!ready()} onClick={()=>props.work.dispatch(request("check_fit"))}>Check source / device</button>
            <button disabled={!ready() || props.value.runtime.status.phase !== "empty"} onClick={()=>props.work.dispatch(request("load_on_device"))}>Load on device</button>
          </div>
        </section>
      }</Show>
    </div>
  </>;
}
