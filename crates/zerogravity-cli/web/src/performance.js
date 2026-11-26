// Selected registration only; no additional fetch/timer or persisted history.
export function history() {
  let registration, baseline, rates = [];
  return model => {
    const id = model?.registration_id;
    const p = model?.performance;
    if (id !== registration) {
      registration = id;
      baseline = undefined;
      rates = [];
    }
    if (!p || !Number.isFinite(p.loaded_ms) || !Number.isFinite(p.output_tokens)) {
      baseline = undefined;
      rates = [];
      return rates;
    }
    if (baseline) {
      const dt = p.loaded_ms - baseline.loaded_ms;
      const tokens = p.output_tokens - baseline.output_tokens;
      if (dt < 0 || tokens < 0 || dt > 6000) rates = [];
      else if (dt < 750) return rates;
      else rates = [...rates.slice(-23), tokens * 1000 / dt];
    }
    baseline = {loaded_ms:p.loaded_ms,output_tokens:p.output_tokens};
    return rates;
  };
}
export const tps = v => Number.isFinite(v) ? v.toFixed(1) : "—";
export const latency = v => !Number.isFinite(v) ? "—" : v >= 1000 ? (v/1000).toFixed(2)+" s" : Math.round(v)+" ms";
export function uptime(ms) {
  const s = Math.floor(ms/1000);
  return s >= 3600 ? Math.floor(s/3600)+"h "+Math.floor(s/60)%60+"m" :
    s >= 60 ? Math.floor(s/60)+"m "+s%60+"s" : s+"s";
}
export const deviceLabel = execution => execution?.kind === "host_cpu" ? "Host CPU" :
  execution?.device ? execution.device.backend+" · "+execution.device.device : "Not reported";

export function modelName(model) {
  const aliases = model.aliases ?? [];
  const named = aliases.find(alias=> !/^[a-f0-9]{64}$/i.test(alias));
  if (named) return named;
  const family = model.capabilities?.family ?? "Model";
  return aliases.length ? family+" · "+aliases[0].slice(0,8)+"…" : family;
}

// Complete, ordered compute ranges only. Compress without inventing an owner
// for any cell spanning different devices. Same device retains its identity.
export function layerStrip(placement, maxCells = 48) {
  const total = placement?.total_layers;
  if (!Number.isSafeInteger(total) || total <= 0 || !Array.isArray(placement.spans)) return null;
  const devices = [], spans = [];
  let end = 0;
  for (const span of placement.spans) {
    if (span.start !== end || !Number.isSafeInteger(span.end) || span.end <= end || span.end > total) return null;
    const execution = span.execution;
    const key = execution?.kind === "host_cpu" ? "host_cpu" :
      execution?.kind === "device" && execution.device ? JSON.stringify([execution.device.backend, execution.device.device]) : null;
    if (key === null) return null;
    let owner = devices.findIndex(device => device.key === key);
    if (owner < 0) {
      owner = devices.length;
      devices.push({key, label:deviceLabel(execution)});
    }
    spans.push({start:span.start,end:span.end,owner});
    end = span.end;
  }
  const count = Math.min(total,maxCells);
  if (end !== total || count < 1) return null;
  const boundary = i => Math.floor(i * total / count);
  const cells = Array.from({length:count}, (_,i) => {
    const start = boundary(i), end = boundary(i+1);
    const owners = spans.filter(span => span.start < end && span.end > start);
    const owner = owners.every(span => span.owner === owners[0].owner) ? owners[0].owner : null;
    return {start,end,owner};
  });
  return {cells,devices};
}
export const deviceMark = index => "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[index] ?? "?";
export const providerLabel = provider => ({Cpu:"CPU",Cuda:"CUDA",Hip:"ROCm",Vulkan:"Vulkan",GraphOnly:"Graph only",Remote:"Remote"})[provider] ?? "Not reported";
