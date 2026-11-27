//! Client-local visible trend over the existing registration counters.
use super::{format, theme::Theme};
use ratatui::text::{Line, Span};
use std::collections::VecDeque;
use zerogravity::models::{LayerDevice, LayerPlacement, ModelPerformance, ModelSnapshot};
use zerogravity::provider::ProviderKind;

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
    fn trend(&self, registration: u64) -> Option<(&VecDeque<f64>, f64)> {
        if self.registration != Some(registration) {
            return None;
        }
        Some((&self.rates, *self.rates.back()?))
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

/// Actual selected provider, independent of the device inventory or requested backend.
pub(super) fn provider(model: &ModelSnapshot) -> &'static str {
    match model
        .capabilities
        .as_ref()
        .and_then(|cap| cap.details.execution.as_ref())
        .map(|e| e.selected.provider)
    {
        Some(ProviderKind::Cpu) => "CPU",
        Some(ProviderKind::Cuda) => "CUDA",
        Some(ProviderKind::Hip) => "ROCm",
        Some(ProviderKind::Vulkan) => "Vulkan",
        Some(ProviderKind::GraphOnly) => "Graph only",
        Some(ProviderKind::Remote) => "Remote",
        None => "Not reported",
    }
}

fn paired(left: Span<'static>, right: Span<'static>, column: usize) -> Line<'static> {
    let padding = column.saturating_sub(left.width());
    Line::from(vec![left, Span::raw(" ".repeat(padding)), right])
}

fn chart(rates: &VecDeque<f64>, width: usize, tall: bool, theme: Theme) -> Vec<Line<'static>> {
    let count = rates.len().min(width);
    let rates = rates
        .iter()
        .skip(rates.len() - count)
        .copied()
        .collect::<Vec<_>>();
    let peak = rates.iter().copied().fold(0.0, f64::max);
    let gap = width >= count * 2;
    let height = if tall { 2usize } else { 1 };
    let blocks = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    (0..height)
        .rev()
        .map(|row| {
            let mut spans = Vec::with_capacity(count);
            for (i, rate) in rates.iter().enumerate() {
                let level = if *rate == 0.0 {
                    0
                } else {
                    ((*rate / peak) * (height * 8) as f64)
                        .ceil()
                        .clamp(1.0, (height * 8) as f64) as usize
                };
                let mark = if row == 0 && level == 0 {
                    '·'
                } else {
                    blocks[level.saturating_sub(row * 8).min(8)]
                };
                let style = if i + 4 >= count {
                    theme.accent()
                } else {
                    theme.muted()
                };
                spans.push(Span::styled(
                    if gap {
                        format!("{mark} ")
                    } else {
                        mark.to_string()
                    },
                    style,
                ));
            }
            Line::from(spans)
        })
        .collect()
}

/// A compact rendering of complete reported spans. Each cell covers an exact
/// interval; a compressed cell crossing different devices has no single owner.
struct LayerStrip<'a> {
    devices: Vec<&'a LayerDevice>,
    cells: Vec<Option<usize>>,
}
fn layer_strip(placement: &LayerPlacement, width: usize) -> Option<LayerStrip<'_>> {
    let total = placement.total_layers;
    let mut end = 0;
    let mut devices = Vec::new();
    for span in &placement.spans {
        if span.start != end || span.end <= end || span.end > total {
            return None;
        }
        end = span.end;
        if !devices.contains(&&span.execution) {
            devices.push(&span.execution);
        }
    }
    let count = total.min(width).min(48);
    if end != total || count == 0 {
        return None;
    }
    let boundary = |i: usize| (i as u128 * total as u128 / count as u128) as usize;
    let cells = (0..count)
        .map(|i| {
            let (start, end) = (boundary(i), boundary(i + 1));
            let mut owners = placement
                .spans
                .iter()
                .filter(|span| span.start < end && span.end > start);
            let first = &owners.next()?.execution;
            if owners.all(|span| &span.execution == first) {
                devices.iter().position(|device| *device == first)
            } else {
                None
            }
        })
        .collect();
    Some(LayerStrip { devices, cells })
}
fn device_mark(index: usize) -> char {
    "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        .chars()
        .nth(index)
        .unwrap_or('?')
}
fn layer_style(index: usize, theme: Theme) -> ratatui::style::Style {
    match index % 4 {
        0 => theme.accent(),
        1 => theme.info(),
        2 => theme.warning(),
        _ => theme.muted(),
    }
}

pub(super) fn summary(
    model: &ModelSnapshot,
    history: &History,
    theme: Theme,
    width: usize,
    height: u16,
) -> Vec<Line<'static>> {
    let p = model.performance;
    let mut lines = vec![
        Line::from(Span::styled(
            format!("Since loaded · {}", uptime(p.loaded_ms)),
            theme.muted(),
        )),
        Line::default(),
    ];
    let decode = tps(p.decode_tps());
    let ttft = latency(p.mean_ttft_ms());
    let column = width / 2;
    if column >= 13 && decode.chars().count() < column && ttft.chars().count() <= width - column {
        lines.push(paired(
            Span::styled(decode, theme.accent()),
            Span::styled(ttft, theme.heading()),
            column,
        ));
        lines.push(paired(
            Span::styled("Avg decode", theme.muted()),
            Span::styled("Avg TTFT", theme.muted()),
            column,
        ));
    } else {
        lines.push(Line::from(vec![
            Span::styled("Avg decode  ", theme.muted()),
            Span::styled(decode, theme.accent()),
        ]));
        lines.push(Line::from(vec![
            Span::styled("Avg TTFT    ", theme.muted()),
            Span::styled(ttft, theme.heading()),
        ]));
    }
    if let Some((rates, latest)) = history.trend(model.registration_id) {
        lines.push(Line::default());
        let current = format!("{latest:.1} tok/s");
        let caption = "Recent output";
        if width >= caption.len() + current.len() + 2 {
            lines.push(paired(
                Span::styled(caption, theme.muted()),
                Span::styled(current.clone(), theme.heading()),
                width - current.len(),
            ));
        } else {
            lines.push(Line::from(Span::styled(
                format!("{caption} · {current}"),
                theme.muted(),
            )));
        }
        lines.extend(chart(rates, width, height >= 18, theme));
    }
    lines.push(Line::from(Span::styled(
        format!(
            "{} completed · {} output tokens",
            p.completed_requests, p.output_tokens
        ),
        theme.muted(),
    )));
    if model.admitted_requests != 0 {
        lines.push(Line::from(Span::styled(
            format!("● {} admitted", model.admitted_requests),
            theme.accent(),
        )));
    }
    if let Some(capability) = &model.capabilities {
        lines.push(Line::default());
        let provider = provider(model);
        if width >= 10 + provider.len() {
            lines.push(paired(
                Span::styled("Execution", theme.muted()),
                Span::raw(provider),
                width - provider.len(),
            ));
        } else {
            lines.push(Line::from(format!("Execution · {provider}")));
        }
        lines.push(Line::from(Span::styled(
            format!("Context capacity: {}", capability.max_context),
            theme.muted(),
        )));
        if let Some(placement) = &capability.details.layer_placement {
            lines.push(Line::from(Span::styled(
                format!("Layer execution · {} layers", placement.total_layers),
                theme.muted(),
            )));
            if let Some(strip) = layer_strip(placement, width) {
                let single = strip.devices.len() == 1;
                lines.push(Line::from(
                    strip
                        .cells
                        .iter()
                        .map(|cell| match cell {
                            Some(index) => Span::styled(
                                if single { '▮' } else { device_mark(*index) }.to_string(),
                                layer_style(*index, theme),
                            ),
                            None => Span::styled("╪", theme.muted()),
                        })
                        .collect::<Vec<_>>(),
                ));
                if single {
                    lines.push(Line::from(Span::styled(
                        target(strip.devices[0]),
                        theme.muted(),
                    )));
                } else {
                    for (i, device) in strip.devices.iter().enumerate() {
                        lines.push(Line::from(vec![
                            Span::styled(format!("{} ", device_mark(i)), layer_style(i, theme)),
                            Span::styled(target(device), theme.muted()),
                        ]));
                    }
                    if strip.cells.contains(&None) {
                        lines.push(Line::from(Span::styled(
                            "╪ Mixed range · see details",
                            theme.muted(),
                        )));
                    }
                }
            } else {
                lines.push(Line::from(Span::styled(
                    "Incomplete map · see details",
                    theme.muted(),
                )));
            }
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
        format!(
            "Admitted requests: {} (includes queued, active and retiring readers)",
            model.admitted_requests
        ),
        "Engine acquisition to committed first non-EOS token; includes queue/prefill.".into(),
        "Decode: token intervals / sum of within-request commit intervals; not wire throughput."
            .into(),
        "Recent output: sampled total committed tokens / elapsed wall time.".into(),
    ];
    if let Some(last) = p.last_completed {
        lines.extend([
            String::new(),
            "Last completed request · registration-wide".into(),
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
        ]);
        if let Some(prefill) = cap.prefill {
            lines.push(format!("Prefill: {prefill:?}"));
        }
        if cap.transcription {
            lines.push("Transcription: native".into());
        }
        if let Some(dimensions) = cap.embedding_dimensions {
            lines.push(format!("Embedding dimensions: {dimensions}"));
        }
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
                        "{}–{} · {} layers · {}",
                        span.start,
                        span.end - 1,
                        span.end - span.start,
                        target(&span.execution)
                    ));
                }
            }
            lines.push(
                "Compute placement only; not a weight-residency or utilization measurement.".into(),
            );
            lines.push(
                "Activity is registration-wide; instantaneous layer execution is not observed."
                    .into(),
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
    fn layer_cells_preserve_exact_owner_and_mark_cross_device_bins() {
        use zerogravity::models::LayerSpan;
        let cpu = LayerDevice::HostCpu;
        let gpu = LayerDevice::Device {
            device: zerogravity::resource_arbiter::PhysicalDeviceKey::new("pci", "0000:01:00.0")
                .unwrap(),
        };
        let placement = LayerPlacement {
            total_layers: 6,
            spans: vec![
                LayerSpan {
                    start: 0,
                    end: 2,
                    execution: cpu.clone(),
                },
                LayerSpan {
                    start: 2,
                    end: 5,
                    execution: gpu.clone(),
                },
                LayerSpan {
                    start: 5,
                    end: 6,
                    execution: cpu.clone(),
                },
            ],
        };
        let full = layer_strip(&placement, 6).unwrap();
        assert_eq!(full.devices, vec![&cpu, &gpu]);
        assert_eq!(
            full.cells,
            vec![Some(0), Some(0), Some(1), Some(1), Some(1), Some(0)]
        );
        assert_eq!(layer_strip(&placement, 2).unwrap().cells, vec![None, None]);
        let mut incomplete = placement;
        incomplete.spans[1].start = 3;
        assert!(layer_strip(&incomplete, 6).is_none());
    }

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
