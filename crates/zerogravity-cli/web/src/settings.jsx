// Cold documents and bounded host observations; no browser-owned policy store.
import { createEffect, createMemo, createSignal, For, Show, onCleanup } from "solid-js";
import { ConfirmButton } from "./resident.jsx";

const nameValid = name => /^[a-z0-9._-]{1,128}$/.test(name) && ![".","..","default"].includes(name);
const bytes = value => new TextEncoder().encode(value).length;
const settings = action => ({kind:"settings", action});
function Notice(props) {
  return <Show when={props.inventory.notice()}><p class="notice" role="status">{props.inventory.notice()}</p></Show>;
}

export function Profiles(props) {
  const data = () => props.inventory.value("profiles");
  const [name, setName] = createSignal("");
  const [search, setSearch] = createSignal("");
  const rows = createMemo(() => (data()?.profiles ?? []).filter(name => name.includes(search().trim().toLowerCase())));
  const create = async () => {
    if (await props.work.dispatch(settings({command:"create",name:name()}))) setName("");
  };
  return <section class="panel">
    <div class="panel-title"><h2>Profiles</h2><span class="subtle">One configuration schema</span></div>
    <p class="note">Open a document to edit, validate, copy, activate or remove it. Activation affects future actions only, not admitted models or listeners.</p>
    <Notice inventory={props.inventory}/>
    <Show when={data()} fallback={<p class="subtle">Waiting for profile inventory…</p>}>
      <label for="profile-search" class="field-label">Search profiles</label>
      <input id="profile-search" value={search()} onInput={event=>setSearch(event.currentTarget.value)}/>
      <Show when={data().active == null}><button class="row" onClick={()=>props.open("@active")}>Portable configuration · active</button></Show>
      <div class="entries"><For each={rows()}>{name=><button class="row" onClick={()=>props.open(name)}>
        <span class="row-main">{name}</span><span class="subtle">{name === "default" ? "config.yaml" : "Named profile"}{data().active === name ? " · active" : ""}</span>
      </button>}</For></div>
      <Show when={!rows().length}><p class="subtle">No matching profiles.</p></Show>
      <label class="field-label" for="profile-name">New profile name</label>
      <input id="profile-name" value={name()} onInput={event=>setName(event.currentTarget.value)} placeholder="workstation"/>
      <div class="form-actions"><button class="primary" disabled={!nameValid(name()) || !props.work.ready()} onClick={create}>Create profile</button></div>
    </Show>
  </section>;
}

export function Config(props) {
  const data = () => props.inventory.value("config");
  const ready = () => props.work.ready() && props.inventory.state() === "ready";
  const [search,setSearch] = createSignal("");
  const [copy,setCopy] = createSignal("");
  const [draft,setDraft] = createSignal(null);
  const [saving,setSaving] = createSignal(null);
  const rows = createMemo(() => (data()?.rows ?? []).filter(row => (row.key + " " + (row.value ?? "") + " " + row.help).toLowerCase().includes(search().toLowerCase())));
  const stale = () => draft() && draft().snapshot !== data()?.snapshot;
  const request = (command, extra={}) => settings({command,snapshot:data().snapshot,...extra});
  const openDraft = row => setDraft({snapshot:data().snapshot, key:row?.key,
    value:row?.key === "ui.theme" ? (row.value?.trim() || "auto") : row ? (row.explicit ? row.value ?? "" : "") : data().raw});
  createEffect(() => {
    const editing = draft();
    props.previewTheme(editing?.key === "ui.theme" ? editing.value : null);
  });
  onCleanup(() => props.previewTheme(null));
  const save = async () => {
    const selected = draft();
    if (!selected || stale()) return;
    const result = await props.work.dispatch(settings(selected.key ?
      {command:"set",snapshot:selected.snapshot,key:selected.key,value:selected.value} :
      {command:"replace",snapshot:selected.snapshot,yaml:selected.value}));
    if (result) setSaving(result.id);
  };
  createEffect(() => {
    const target = props.target();
    void target;
    setDraft(null); setSaving(null); setSearch(""); setCopy("");
  });
  createEffect(() => {
    if (props.appearanceRequest()) {
      setSearch("ui.theme");
      props.clearAppearance();
    }
  });
  createEffect(() => {
    const result = props.work.result();
    if (saving() && result?.id === saving() && result.state !== "running") {
      setSaving(null);
      if (result.state === "completed") setDraft(null);
    }
  });
  return <section class="panel">
    <div class="panel-title"><h2>{data()?.profile ?? (props.target() === "@active" ? "Active configuration" : props.target())}</h2><button class="quiet" onClick={()=>props.open("@active")}>Open active</button></div>
    <Notice inventory={props.inventory}/>
    <Show when={data()} fallback={<p class="subtle">Waiting for document observation…</p>}>
      <p class="note">{data().active ? "Active for future actions." : "Editing only; not active."} {data().exists ? "Stored document." : "Built-in defaults; no file created yet."}</p>
      <Show when={data().warning}><p class="notice">{data().warning}</p></Show>
      <div class="form-actions">
        <button disabled={!ready() || !data().valid} onClick={()=>props.work.dispatch(request("validate"))}>Validate</button>
        <button disabled={!ready() || !data().valid || data().active} onClick={()=>props.work.dispatch(request("activate"))}>Activate profile</button>
        <ConfirmButton work={props.work} disabled={!ready() || !data().removable} request={()=>request("remove",{confirm:true})}
          label={"Remove profile " + data().profile + "? Existing runtime owners remain unchanged."}>Remove profile</ConfirmButton>
      </div>
      <label class="field-label" for="copy-profile">Copy to a named profile</label>
      <input id="copy-profile" value={copy()} onInput={event=>setCopy(event.currentTarget.value)}/>
      <div class="form-actions"><button disabled={!nameValid(copy()) || !data().valid || !ready()} onClick={async()=>{if(await props.work.dispatch(request("copy",{name:copy()})))setCopy("");}}>Copy profile</button></div>
      <p class="field-hint">Local paths, model graphs, prompt text and unrestricted YAML are native-only. Protected values are not disclosed. Browser-safe raw YAML is normalized without comments.</p>
      <Show when={data().raw != null}><div class="form-actions"><button disabled={!ready() || !!draft()} onClick={()=>openDraft(null)}>Edit YAML</button></div></Show>
      <Show when={draft()}>{editing=>
        <div class="confirmation" role="group" aria-label="Configuration draft">
          <label class="field-label" for="config-draft">{editing().key ?? "Normalized YAML"}</label>
          <Show when={editing().key === "ui.theme"} fallback={
            <textarea id="config-draft" rows={editing().key ? 3 : 12} value={editing().value} onInput={event=>setDraft(previous=>({...previous,value:event.currentTarget.value}))}/>
          }>
            <select id="config-draft" value={editing().value} onChange={event=>setDraft(previous=>({...previous,value:event.currentTarget.value}))}>
              <option value="auto">Auto · follow system appearance</option>
              <option value="dark">Dark</option>
              <option value="light">Light</option>
            </select>
            <p class="field-hint">Preview only until Save. Editing an inactive profile does not change the active host appearance.</p>
          </Show>
          <Show when={stale()}><p class="notice">Document observation changed. Cancel and reopen the draft before saving.</p></Show>
          <div class="form-actions">
            <button class="primary" disabled={!ready() || !!saving() || stale() || !editing().value.trim() || bytes(editing().value) > (editing().key ? 8192 : 128*1024)} onClick={save}>Save configuration</button>
            <button class="quiet" onClick={()=>{setDraft(null);setSaving(null);}}>Cancel draft</button>
          </div>
        </div>
      }</Show>
      <label class="field-label" for="config-search">Search settings</label>
      <input id="config-search" value={search()} onInput={event=>setSearch(event.currentTarget.value)}/>
      <div class="config-rows">
        <For each={rows()}>{row=><div class="config-row">
          <div><strong class="mono wrap">{row.key}</strong><div class="field-hint">{row.help}</div><div class="subtle">{row.explicit ? "Explicit" : "Inherited / built-in"} · {row.origin}</div></div>
          <div class="mono wrap">{row.value == null ? "Native-only" : row.value.trim()}</div>
          <div class="form-actions">
            <button disabled={!row.editable || !ready() || !!draft()} onClick={()=>openDraft(row)}>Edit</button>
            <Show when={row.explicit && row.editable}><ConfirmButton work={props.work} disabled={!ready() || !!draft()} request={()=>request("unset",{key:row.key,confirm:true})} label={"Unset " + row.key + " and restore inherited policy?"}>Unset</ConfirmButton></Show>
          </div>
        </div>}</For>
      </div>
    </Show>
  </section>;
}

export function Logs(props) {
  const data = () => props.inventory.value("logs");
  const [search,setSearch] = createSignal("");
  const [level,setLevel] = createSignal("");
  const [component,setComponent] = createSignal("");
  const [paused,setPaused] = createSignal(null);
  const rows = createMemo(() => (data()?.entries ?? []).filter(row =>
    (!paused() || row.sequence.length < paused().length || (row.sequence.length === paused().length && row.sequence < paused())) &&
    (!level() || row.level === level()) && (!component() || row.component === component()) &&
    [row.message,row.correlation,row.component,row.level,row.sequence,row.stage].join(" ").toLowerCase().includes(search().toLowerCase())));
  return <section class="panel">
    <div class="panel-title"><h2>Operational logs</h2><button disabled={!data()} onClick={()=>setPaused(paused() ? null : data().revision)}>{paused() ? "Follow logs" : "Pause logs"}</button></div>
    <p class="note">Recent safe host observations. Full operational history remains in the local log destination. {paused() ? "Paused view; collection continues." : "Following."} {data()?.dropped ?? "0"} older rows dropped.</p>
    <Notice inventory={props.inventory}/>
    <Show when={data()?.sink_available === false}><p class="note">Local log destination unavailable; inspect host diagnostics.</p></Show>
    <div class="form-actions">
      <input aria-label="Search logs" placeholder="Search logs" value={search()} onInput={event=>setSearch(event.currentTarget.value)}/>
      <select aria-label="Log level" value={level()} onChange={event=>setLevel(event.currentTarget.value)}><option value="">All levels</option><For each={["INFO","WARN","ERROR"]}>{name=><option>{name}</option>}</For></select>
      <select aria-label="Log component" value={component()} onChange={event=>setComponent(event.currentTarget.value)}><option value="">All components</option><For each={["action","models","requests","serve","cache","download","control"]}>{name=><option>{name}</option>}</For></select>
    </div>
    <div class="log-rows"><For each={rows()}>{row=><div class="log-row"><span class="mono subtle" title={new Date(row.timestamp_unix_ms).toISOString()}>{new Date(row.timestamp_unix_ms).toISOString().slice(11,23)}Z</span><span class="mono">{row.level}</span><span>{row.component}</span><span class="wrap">{row.message}<small class="subtle"> · {row.correlation}{row.stage ? ` · ${row.stage}` : ""}{row.elapsed_ms == null ? "" : ` · ${row.elapsed_ms} ms`}</small><Show when={row.details}>{detail=><details><summary class="subtle">Details</summary><pre class="mono wrap">{JSON.stringify(detail(),null,2)}</pre></details>}</Show></span></div>}</For></div>
    <Show when={!rows().length}><p class="subtle">{data() ? "No matching operational observations." : "Waiting for operational observations…"}</p></Show>
  </section>;
}
