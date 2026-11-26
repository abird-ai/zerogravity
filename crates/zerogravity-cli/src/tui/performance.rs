//! Client-local visible trend over the existing registration counters.
use super::{format, theme::Theme};
use ratatui::text::{Line, Span};
use std::collections::VecDeque;
use zerogravity::models::{LayerDevice, ModelPerformance, ModelSnapshot};

const SAMPLES: usize = 24;

#[derive(Default)]
pub(super) struct History {
    registration: Option<u64>,
    last: Option<(u64, u64)>,
    rates: VecDeque<f64>,
}
impl History {
    pub fn observe(&mut self, model: Option<&ModelSnapshot>) {
        let Some(model) = model else {
            *self = Self::default();
            return;
        };
        self.sample(model.registration_id, model.performance);
    }
    fn sample(&mut self, registration: u64, performance: ModelPerformance) {
        if self.registration != Some(registration) {
            *self = Self {
                registration: Some(registration),
                ..Default::default()
            };
        }
        let next = (performance.loaded_ms, performance.output_tokens);
        if let Some((ms, tokens)) = self.last {
            if next.0 < ms || next.1 < tokens || next.0 - ms > 6000 {
                self.rates.clear();
            } else if next.0 - ms < 750 {
                return;
            } else {
                if self.rates.len() == SAMPLES {
                    self.rates.pop_front();
                }
                self.rates
                    .push_back((next.1 - tokens) as f64 * 1000.0 / (next.0 - ms) as f64);
            }
        }
        self.last = Some(next);
    }
    fn trend(&self, registration: u64) -> Option<(String, f64)> {
        if self.registration != Some(registration) {
            return None;
        }
        let latest = *self.rates.back()?;
        let peak = self.rates.iter().copied().fold(0.0, f64::max);
        let bars = self
            .rates
            .iter()
            .map(|value| {
                if *value == 0.0 {
                    '·'
                } else {
                    let level = ((value / peak) * 8.0).ceil().clamp(1.0, 8.0) as usize;
                    ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'][level - 1]
                }
            })
            .collect();
        Some((bars, latest))
    }
}
fn tps(rate: Option<f64>) -> String {
    rate.map_or_else(|| "—".into(), |rate| format!("{rate:.1} tok/s"))
}
fn latency(ms: Option<f64>) -> String {
    ms.map_or_else(
        || "—".into(),
        |ms| {
            if ms >= 1000.0 {
                format!("{:.2} s", ms / 1000.0)
            } else {
                format!("{ms:.0} ms")
            }
        },
    )
}
fn uptime(ms: u64) -> String {
    let seconds = ms / 1000;
    if seconds >= 3600 {
        format!("{}h {}m", seconds / 3600, (seconds / 60) % 60)
    } else if seconds >= 60 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}
fn target(device: &LayerDevice) -> String {
    match device {
        LayerDevice::HostCpu => "Host CPU".into(),
        LayerDevice::Device { device } => format!("{} · {}", device.backend, device.device),
    }
}

pub(super) fn model_name(model: &ModelSnapshot) -> String {
    let digest = |alias: &str| alias.len() == 64 && alias.bytes().all(|c| c.is_ascii_hexdigit());
    if let Some(alias) = model.aliases.iter().find(|alias| !digest(alias)) {
        return alias.clone();
    }
    let family = model
        .capabilities
        .as_ref()
        .and_then(|cap| cap.details.artifact.as_ref())
        .map_or("Model", |artifact| artifact.family.as_str());
    model.aliases.first().map_or_else(
        || family.into(),
        |alias| format!("{family} · {}…", &alias[..8]),
    )
}

pub(super) fn summary(
    model: &ModelSnapshot,
    history: &History,
    theme: Theme,
) -> Vec<Line<'static>> {
    let p = model.performance;
    let mut lines = vec![
        Line::from(Span::styled(
            format!("Since loaded · {}", uptime(p.loaded_ms)),
            theme.muted(),
        )),
        Line::default(),
        Line::from(vec![
            Span::styled("Avg decode  ", theme.muted()),
            Span::styled(tps(p.decode_tps()), theme.accent()),
        ]),
        Line::from(vec![
            Span::styled("Avg TTFT    ", theme.muted()),
            Span::styled(latency(p.mean_ttft_ms()), theme.accent()),
        ]),
    ];
    if let Some((bars, latest)) = history.trend(model.registration_id) {
        lines.push(Line::default());
        lines.push(Line::from(vec![
            Span::styled(bars, theme.accent()),
            Span::raw(format!("  {latest:.1} tok/s")),
        ]));
        lines.push(Line::from(Span::styled(
            "Recent aggregate output",
            theme.muted(),
        )));
    }
    lines.push(Line::from(Span::styled(
        format!(
            "{} completed · {} output tokens",
            p.completed_requests, p.output_tokens
        ),
        theme.muted(),
    )));
    if let Some(placement) = model
        .capabilities
        .as_ref()
        .and_then(|cap| cap.details.layer_placement.as_ref())
    {
        lines.push(Line::default());
        lines.push(Line::from(vec![
            Span::styled("Layers ", theme.muted()),
            Span::raw(format!("{}", placement.total_layers)),
        ]));
        if let [span] = placement.spans.as_slice() {
            lines.push(Line::from(vec![
                Span::styled("━".repeat(placement.total_layers.min(24)), theme.info()),
                Span::raw(format!("  {}", target(&span.execution))),
            ]));
        } else {
            lines.push(Line::from("See layer placement in Details"));
        }
    }
    lines
}

pub(super) fn details(model: &ModelSnapshot) -> Vec<String> {
    let p = model.performance;
    let mut lines = vec![
        format!(
            "Registration {} · since loaded {}",
            model.registration_id,
            uptime(p.loaded_ms)
        ),
        format!("Aliases: {}", model.aliases.join(", ")),
        String::new(),
        format!("Average decode: {}", tps(p.decode_tps())),
        format!(
            "Average TTFT: {} · {} first-token samples",
            latency(p.mean_ttft_ms()),
            p.ttft_samples
        ),
        format!(
            "Decode intervals: {} · committed output tokens: {}",
            p.decode_intervals, p.output_tokens
        ),
        format!("Completed requests: {}", p.completed_requests),
        "Engine acquisition to committed first non-EOS token; includes queue/prefill.".into(),
        "Decode: token intervals / sum of within-request commit intervals; not wire throughput."
            .into(),
        "Recent output: sampled total committed tokens / elapsed wall time.".into(),
    ];
    if let Some(last) = p.last_completed {
        lines.extend([
            String::new(),
            "Last completed request".into(),
            format!(
                "Decode: {} · TTFT: {}",
                tps(last.decode_tps()),
                latency(last.ttft_ms())
            ),
            format!("Output: {} tokens", last.output_tokens),
        ]);
    }
    if let Some(cap) = &model.capabilities {
        lines.extend([
            String::new(),
            "Prepared execution".into(),
            format!("Context capacity: {}", cap.max_context),
            format!("Vocabulary: {}", cap.vocabulary_size),
            format!("Prefill: {:?}", cap.prefill),
        ]);
        if let Some(execution) = &cap.details.execution {
            lines.push(format!(
                "Provider: {:?} · {:?}",
                execution.selected.provider, execution.selected.mode
            ));
            if let Some(program) = &execution.program {
                lines.push(format!(
                    "Recorded program: {} nodes · {} state domains",
                    program.node_count, program.state_domain_count
                ));
            }
        }
        if let Some(placement) = &cap.details.layer_placement {
            lines.extend([
                String::new(),
                format!(
                    "Layer execution · {} layers (zero-based IDs)",
                    placement.total_layers
                ),
            ]);
            for span in &placement.spans {
                if span.start < span.end {
                    lines.push(format!(
                        "{}–{}  {}  {}",
                        span.start,
                        span.end - 1,
                        "━".repeat((span.end - span.start).min(24)),
                        target(&span.execution)
                    ));
                }
            }
            lines.push(
                "Compute placement only; not a weight-residency or utilization measurement.".into(),
            );
        } else {
            lines.push("Layer placement: not reported by this adapter.".into());
        }
        if let Some(bytes) = cap.details.source_host_reserved_bytes {
            lines.push(format!("Source host reservation: {}", format::bytes(bytes)));
        }
        if let Some(memory) = &cap.details.memory {
            use zerogravity::models::ModelMemory;
            lines.push(String::new());
            match memory {
                ModelMemory::Host {
                    session_state_bytes,
                    session_scratch_bytes,
                } => {
                    lines.push(format!(
                        "Host state {} · scratch {}",
                        format::bytes(*session_state_bytes),
                        format::bytes(*session_scratch_bytes)
                    ));
                }
                ModelMemory::Resident(memory) => {
                    lines.push(format!(
                        "Device: {} · resident {}",
                        memory.device.device,
                        format::bytes(memory.resident_bytes as u64)
                    ));
                    lines.push(format!(
                        "Model capacity: {}",
                        format::bytes(memory.model_capacity_bytes)
                    ));
                }
                ModelMemory::ResidentText {
                    admission,
                    resident_bytes,
                } => {
                    lines.push(format!(
                        "Device: {} · resident {}",
                        admission.device.device,
                        format::bytes(*resident_bytes as u64)
                    ));
                    lines.push(format!(
                        "Resident weights: {}",
                        format::bytes(admission.resident_weight_bytes as u64)
                    ));
                    lines.push(format!(
                        "State reserve: {}",
                        format::bytes(admission.state_reserve_bytes as u64)
                    ));
                    lines.push(format!(
                        "Admitted model capacity: {}",
                        format::bytes(admission.admitted_model_capacity_bytes as u64)
                    ));
                    lines.push(format!(
                        "Context device budget: {}",
                        format::bytes(admission.context_device_budget_bytes as u64)
                    ));
                }
            }
            lines.push(
                "Original lease / admission observations, not total device VRAM usage.".into(),
            );
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sampled_history_uses_elapsed_owner_time_and_resets_on_replacement() {
        let mut history = History::default();
        let p = |ms, tokens| ModelPerformance {
            loaded_ms: ms,
            output_tokens: tokens,
            ..Default::default()
        };
        history.sample(1, p(1000, 10));
        assert!(history.trend(1).is_none());
        history.sample(1, p(1200, 12));
        history.sample(1, p(2000, 50));
        assert_eq!(history.trend(1).unwrap().1, 40.0);
        history.sample(1, p(3000, 50));
        assert_eq!(history.trend(1).unwrap().1, 0.0);
        history.sample(1, p(10000, 50));
        assert!(history.trend(1).is_none());
        history.sample(1, p(11000, 70));
        assert_eq!(history.trend(1).unwrap().1, 20.0);
        history.sample(2, p(0, 0));
        assert!(history.trend(1).is_none() && history.trend(2).is_none());
    }
}
