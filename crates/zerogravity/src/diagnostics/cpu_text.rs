//! Explicit synchronous CPU source/audit driver. No process IO, signals,
//! environment lookup or model acquisition; ordinary runtime requests use Router.
use crate::cpu::ops::top_logit_ranks;
use crate::{Error, Result, prepare::PreparedQwen};
use std::io::Write;

/// Low-level caller-owned preparation or the SDK's retained planning envelope.
/// Neither variant claims serving qualification or supplies a new source issuer.
pub enum Input {
    CallerOwned(PreparedQwen),
    Admitted(crate::prepare::selection::PreparedCpu),
}
impl From<PreparedQwen> for Input {
    fn from(prepared: PreparedQwen) -> Self {
        Self::CallerOwned(prepared)
    }
}
impl From<crate::prepare::selection::PreparedCpu> for Input {
    fn from(prepared: crate::prepare::selection::PreparedCpu) -> Self {
        Self::Admitted(prepared)
    }
}

pub struct TextOptions {
    pub prompt: String,
    pub context: usize,
    pub max_tokens: usize,
    pub raw: bool,
    pub prefill_chunk: usize,
    pub teacher_ids: Option<Vec<u32>>,
}
/// Opened only at the original audit boundary, after prompt validation for
/// layers and after successful prefill for logits. Opening belongs to caller.
pub trait AuditSink {
    fn open(&mut self) -> Result<&mut dyn Write>;
}
impl<W: Write> AuditSink for W {
    fn open(&mut self) -> Result<&mut dyn Write> {
        Ok(self)
    }
}
#[derive(Default)]
pub struct Diagnostics<'a> {
    pub logits: Option<&'a mut dyn AuditSink>,
    pub layers: Option<&'a mut dyn AuditSink>,
    pub top_logits: bool,
}
pub enum Observation<'a> {
    Prefill {
        done: usize,
        total: usize,
    },
    Logits {
        step: usize,
        values: &'a [(usize, f32)],
    },
}
/// Owned output retains its original planning credit through final readers.
/// Payload fields precede the charge and cannot be moved out independently.
pub struct TextOutput {
    text: String,
    generated: Vec<u32>,
    consumed: usize,
    _envelope: Option<crate::resource_arbiter::GlobalResourceLease>,
}
impl TextOutput {
    pub fn text(&self) -> &str {
        &self.text
    }
    pub fn generated(&self) -> &[u32] {
        &self.generated
    }
    pub fn consumed(&self) -> usize {
        self.consumed
    }
}
impl TextOptions {
    pub fn validate(&self, has_logits_audit: bool) -> Result<()> {
        if !(1..=32768).contains(&self.context)
            || !(1..=128).contains(&self.max_tokens)
            || !(1..=32768).contains(&self.prefill_chunk)
            || self.prompt.len() > 1024 * 1024
        {
            return Err(Error::Limit(
                "bounded direct CPU context/output/prompt/prefill required".into(),
            ));
        }
        if self
            .teacher_ids
            .as_ref()
            .is_some_and(|ids| ids.len() != self.max_tokens || !has_logits_audit)
        {
            return Err(Error::Malformed(
                "teacher IDs require one per output and a logits audit sink".into(),
            ));
        }
        Ok(())
    }
}
/// Original source math, chunk boundaries, greedy tie order and audit rows.
/// Callers retain source admission and own diagnostic sink storage/lifetimes.
pub fn generate(
    input: impl Into<Input>,
    options: TextOptions,
    diagnostics: Diagnostics<'_>,
    observe: impl FnMut(Observation<'_>) -> Result<()>,
) -> Result<TextOutput> {
    options.validate(diagnostics.logits.is_some())?;
    // Bind credit first so source owners drop before it on early return.
    let (_envelope, prepared) = match input.into() {
        Input::CallerOwned(prepared) => (None, prepared),
        Input::Admitted(prepared) => {
            if options.context > prepared.context {
                return Err(Error::Limit(
                    "CPU execution exceeds its admitted planning context".into(),
                ));
            }
            (Some(prepared.envelope), prepared.prepared)
        }
    };
    let mut output = generate_inner(&prepared, options, diagnostics, observe)?;
    output._envelope = _envelope;
    Ok(output)
}

/// Reuse the same admitted CPU model for independent sequential requests.
/// The mutable borrow excludes overlapping request state. Presentation
/// happens while the original envelope remains live, with no second loader.
pub fn generate_repeated(
    prepared: &mut crate::prepare::selection::PreparedCpu,
    options: TextOptions,
    diagnostics: Diagnostics<'_>,
    observe: impl FnMut(Observation<'_>) -> Result<()>,
    present: impl FnOnce(&TextOutput) -> Result<()>,
) -> Result<()> {
    options.validate(diagnostics.logits.is_some())?;
    if options.context > prepared.context {
        return Err(Error::Limit(
            "CPU execution exceeds its admitted planning context".into(),
        ));
    }
    let output = generate_inner(&prepared.prepared, options, diagnostics, observe)?;
    present(&output)
}

fn generate_inner(
    prepared: &PreparedQwen,
    options: TextOptions,
    mut diagnostics: Diagnostics<'_>,
    mut observe: impl FnMut(Observation<'_>) -> Result<()>,
) -> Result<TextOutput> {
    let TextOptions {
        prompt,
        context,
        max_tokens,
        raw,
        prefill_chunk,
        teacher_ids,
    } = options;
    let model_source = prepared.model();
    let tokenizer = prepared.tokenizer();
    if teacher_ids.as_ref().is_some_and(|ids| {
        ids.iter()
            .any(|&id| u64::from(id) >= model_source.geometry().vocab_size)
    }) {
        return Err(Error::Malformed(
            "teacher token outside source vocabulary".into(),
        ));
    }
    let mut engine = model_source.new_session()?;
    let input = zerogravity_core::task::TextInput::from_text(prompt, !raw);
    let tokens = tokenizer.semantic_prompt_tokens(&input)?;
    drop(input);
    if tokens.is_empty() {
        return Err(crate::Error::Malformed("prompt tokenization empty".into()));
    }
    if tokens
        .len()
        .checked_add(max_tokens - 1)
        .is_none_or(|n| n > context.min(model_source.geometry().max_position_embeddings as usize))
    {
        return Err(crate::Error::Limit(
            "CPU validation maximum is 32768 input tokens; actual throughput is not guaranteed"
                .into(),
        ));
    }
    observe(Observation::Prefill {
        done: 0,
        total: tokens.len(),
    })?;
    let mut layer_writer = diagnostics
        .layers
        .as_mut()
        .map(|sink| sink.open())
        .transpose()?;
    let mut logits = Vec::new();
    let mut done = 0usize;
    for chunk in tokens.chunks(prefill_chunk) {
        let final_chunk = done + chunk.len() == tokens.len();
        if final_chunk && let Some(writer) = layer_writer.as_mut() {
            let (result, observations) = engine.prefill_transaction_with_layer_trace(chunk)?;
            logits = result;
            // One numeric stage vector per JSONL record, at the final prefill
            // token; no 248K-vocabulary vector or entire KV cache is emitted.
            for observation in observations {
                serde_json::to_writer(&mut *writer, &observation)
                    .map_err(|e| crate::Error::Malformed(format!("layer trace encoding: {e}")))?;
                writer.write_all(b"\n")?;
            }
            writer.flush()?;
        } else if final_chunk {
            logits = engine.prefill_transaction(chunk)?;
        } else {
            // Intermediate chunk outputs are never sampled. Preserve the
            // complete atomic StateRoot update, but avoid a redundant full
            // 248K-vocabulary projection and allocation at each boundary.
            engine.prefill_chunk_without_logits(chunk)?;
        }
        done += chunk.len();
        if done.is_multiple_of(128) || done == tokens.len() {
            observe(Observation::Prefill {
                done,
                total: tokens.len(),
            })?;
        }
    }
    let mut generated = Vec::with_capacity(max_tokens);
    let mut audit = diagnostics
        .logits
        .as_mut()
        .map(|sink| sink.open())
        .transpose()?;

    for generation_step in 0..max_tokens {
        if let Some(writer) = audit.as_mut() {
            let ranks = top_logit_ranks(&logits, 20)?;
            let selected = ranks[0].id;
            serde_json::to_writer(
                &mut *writer,
                &serde_json::json!({
                    "stage":"decode",
                    "step": generated.len(),
                    "prefill_tokens":tokens.len(),
                    "position":engine.position(),
                    "root_id":engine.root_id(),
                    "prefill_chunk":prefill_chunk,
                    "selected_token":selected,
                    "committed_token":teacher_ids.as_ref().map(|ids| ids[generated.len()]).unwrap_or(selected),
                    "state_fingerprint":engine.state_fingerprint(),
                    "state_bytes":engine.state_bytes(),
                    "top":ranks,
                }),
            )
            .map_err(|e| {
                crate::Error::Malformed(format!("cannot encode logits audit: {e}"))
            })?;
            writer.write_all(b"\n")?;
        }
        if diagnostics.top_logits {
            let mut ranks: Vec<_> = logits.iter().enumerate().collect();
            ranks.sort_unstable_by(|a, b| b.1.total_cmp(a.1));
            let top = *ranks
                .first()
                .ok_or_else(|| Error::Malformed("model produced no logits".into()))?
                .1;
            let values = ranks
                .iter()
                .take(8)
                .map(|(id, logit)| (*id, **logit - top))
                .collect::<Vec<_>>();
            observe(Observation::Logits {
                step: generated.len(),
                values: &values,
            })?;
        }
        let greedy_id = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .ok_or_else(|| crate::Error::Malformed("model produced no logits".into()))?
            .0 as u32;
        let id = teacher_ids
            .as_ref()
            .map(|ids| ids[generated.len()])
            .unwrap_or(greedy_id);
        if id == tokenizer.eos() {
            break;
        }
        generated.push(id);
        // Last emitted token is visible output; decoding it again is wasted
        // work and can spuriously reject a valid final context-boundary token.
        if generation_step + 1 < max_tokens {
            logits = engine.forward(id)?;
        }
    }
    if let Some(writer) = audit.as_mut() {
        writer.flush()?;
    }
    Ok(TextOutput {
        text: tokenizer.decode(&generated, true)?,
        generated,
        consumed: engine.position(),
        _envelope: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    struct NoSource;
    impl From<NoSource> for Input {
        fn from(_: NoSource) -> Self {
            panic!("invalid controls reached source ownership")
        }
    }
    fn options() -> TextOptions {
        TextOptions {
            prompt: "plain".into(),
            context: 8,
            max_tokens: 1,
            raw: true,
            prefill_chunk: 1,
            teacher_ids: None,
        }
    }
    #[test]
    fn malformed_controls_refuse_before_source_sink_or_observer() {
        let bad = [
            TextOptions {
                context: 0,
                ..options()
            },
            TextOptions {
                context: 32769,
                ..options()
            },
            TextOptions {
                max_tokens: 0,
                ..options()
            },
            TextOptions {
                max_tokens: 129,
                ..options()
            },
            TextOptions {
                prefill_chunk: 0,
                ..options()
            },
            TextOptions {
                prefill_chunk: 32769,
                ..options()
            },
            TextOptions {
                prompt: "x".repeat(1024 * 1024 + 1),
                ..options()
            },
            TextOptions {
                teacher_ids: Some(vec![0]),
                ..options()
            },
        ];
        for options in bad {
            assert!(
                generate(NoSource, options, Diagnostics::default(), |_| panic!(
                    "observer on refused controls"
                ))
                .is_err()
            );
        }
    }
    #[test]
    fn teacher_count_matches_output_and_requires_explicit_audit() {
        let mut sink = Vec::new();
        let diagnostics = Diagnostics {
            logits: Some(&mut sink),
            ..Default::default()
        };
        assert!(
            TextOptions {
                teacher_ids: Some(vec![0]),
                ..options()
            }
            .validate(diagnostics.logits.is_some())
            .is_ok()
        );
        for ids in [vec![], vec![0, 1]] {
            assert!(
                TextOptions {
                    teacher_ids: Some(ids),
                    ..options()
                }
                .validate(diagnostics.logits.is_some())
                .is_err()
            );
        }
        assert!(sink.is_empty());
    }
}
