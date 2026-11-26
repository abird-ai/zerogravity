import {createEffect, createSignal, For, Show} from "solid-js";
import {history,tps,latency,uptime,deviceLabel} from "./performance.js";

export function Performance(props) {
  const [rates,setRates]=createSignal([]);
  const sample=history();
  createEffect(()=>setRates(sample(props.model)));
  const p=()=>props.model.performance;
  const placement=()=>props.model.capabilities?.layer_placement;
  const memory=()=>props.model.capabilities?.memory;
  const peak=()=>Math.max(0,...rates());
  const current=()=>rates().at(-1);
  const count=n=>typeof n==="number" ? n.toLocaleString() : "—";
  const spanLabel=span=>`${span.start}–${span.end-1} · ${deviceLabel(span.execution)}`;
  return <Show when={p()}>{p=><div class="model-performance">
    <div class="subtle">Since loaded · {uptime(p().loaded_ms)}</div>
    <div class="performance-values">
      <div><div class="metric-number">{tps(p().avg_decode_tps)}<small> tok/s</small></div><div class="subtle">Avg decode</div></div>
      <div><div class="metric-number">{latency(p().avg_ttft_ms)}</div><div class="subtle">Avg TTFT</div></div>
    </div>
    <Show when={rates().length}>
      <div class="performance-caption"><span>Recent aggregate output</span><strong>{tps(current())} tok/s</strong></div>
      <svg class="performance-trend" viewBox="0 0 240 38" role="img" aria-label={`Recent sampled output, latest ${tps(current())} tokens per second, peak ${tps(peak())}`}>
        <For each={rates()}>{(value,i)=><rect x={(24-rates().length+i())*10} y={37-(peak()?value/peak()*34:0)} width="7" height={Math.max(1,peak()?value/peak()*34:0)} rx="1" fill="currentColor" opacity={i()===rates().length-1?1:0.45}/>}</For>
      </svg>
    </Show>
    <div class="subtle">{count(p().completed_requests)} completed · {count(p().output_tokens)} output tokens</div>
    <Show when={placement()}>{map=><div class="layer-map">
      <div class="performance-caption"><span>Layer execution</span><span>{map().total_layers} layers</span></div>
      <div class="layer-track"><For each={map().spans}>{span=><div style={{flex:span.end-span.start}} title={spanLabel(span)} aria-label={spanLabel(span)}/>}</For></div>
      <Show when={map().spans.length===1}><div class="subtle">{deviceLabel(map().spans[0].execution)}</div></Show>
    </div>}</Show>
    <details class="performance-details">
      <summary>Performance details</summary>
      <dl class="facts">
        <dt>Registration</dt><dd>{props.model.registration_id}</dd>
        <dt>Aliases</dt><dd>{props.model.aliases.join(", ")}</dd>
        <Show when={props.model.capabilities?.source_host_reserved_bytes!=null}><dt>Source host reservation</dt><dd>{props.bytes(props.model.capabilities.source_host_reserved_bytes)}</dd></Show>
        <dt>First-token samples</dt><dd>{count(p().ttft_samples)}</dd>
        <dt>Decode intervals</dt><dd>{count(p().decode_intervals)}</dd>
        <Show when={p().last_completed}>{last=><>
          <dt>Last completed · decode</dt><dd>{tps(last().decode_tps)} tok/s</dd>
          <dt>Last completed · TTFT</dt><dd>{latency(last().ttft_ms)}</dd>
          <dt>Last completed · output</dt><dd>{count(last().output_tokens)} tokens</dd>
        </>}</Show>
        <Show when={placement()} fallback={<><dt>Layer execution</dt><dd>Not reported</dd></>}>{map=><For each={map().spans}>{span=><><dt>Layers {span.start}–{span.end-1}</dt><dd>{deviceLabel(span.execution)}</dd></>}</For>}</Show>
        <Show when={memory()?.device}><dt>Resident device</dt><dd>{memory().device.backend} · {memory().device.device}</dd></Show>
        <For each={[["Resident bytes","resident_bytes"],["Resident weights","resident_weight_bytes"],["State reserve","state_reserve_bytes"],["Admitted model capacity","model_capacity_bytes"],["Context device budget","context_device_budget_bytes"],["Host state","host_state_bytes"],["Host scratch","host_scratch_bytes"]]}>{([label,key])=><Show when={memory()?.[key]!=null}><dt>{label}</dt><dd>{props.bytes(memory()[key])}</dd></Show>}</For>
      </dl>
      <p class="note">TTFT: engine request acquisition to first committed non-EOS token, including queue and prefill. Decode averages use within-request commit intervals, including active and ended runs. Recent output uses sampled committed tokens per elapsed wall time, not per-request decode speed. Layer placement describes compute, not weight residency; memory is the original lease/admission observation.</p>
    </details>
  </div>}</Show>;
}
