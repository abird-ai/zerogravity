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
