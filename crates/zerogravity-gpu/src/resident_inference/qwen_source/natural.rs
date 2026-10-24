//! Genuine GPU-selected feedback within the existing two-position source scope.
//! No teacher operand, text/EOS policy, wider temporal or serving permission.
use super::*;

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(super) struct Progress {
    pub calls: usize,
    limit: usize,
}
impl Progress {
    fn seed(self, frontier: u64, maximum: usize, vocabulary: usize, seed: u32) -> Result<Self> {
        if self != Self::default()
            || frontier != 0
            || !(1..=2).contains(&maximum)
            || seed as usize >= vocabulary
        {
            return Err(Error::State(
                "Qwen natural seed/source two-position scope".into(),
            ));
        }
        Ok(Self {
            calls: 0,
            limit: maximum,
        })
    }
    fn next(self, frontier: u64) -> Result<Self> {
        if self.calls == 0 || self.calls >= self.limit || frontier != self.calls as u64 {
            return Err(Error::State(
                "Qwen natural continuation stopped/stale/exhausted".into(),
            ));
        }
        Ok(self)
    }
    fn completed(self, frontier: u64) -> Result<Self> {
        if frontier != self.calls as u64
            || self.limit == 0
            || self.calls >= self.limit
            || self.limit > 2
        {
            return Err(Error::State(
                "Qwen natural proposal progress differs".into(),
            ));
        }
        Ok(Self {
            calls: self.calls + 1,
            ..self
        })
    }
}

pub struct Qwen27bNaturalPending {
    pending: Qwen27bSourceGpuPending,
    progress: Progress,
}
impl Qwen27bNaturalPending {
    /// Scalar observation of the already-Ready original12B packet; no lease escapes.
    pub fn selected(&self) -> Result<(u32, u32)> {
        let token = self
            .pending
            .inner
            .readback
            .device_token
            .as_ref()
            .ok_or_else(|| Error::State("Qwen private Ready packet absent".into()))?;
        Ok((token.id, token.max_bits))
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Qwen27bNaturalCall {
    pub root: RootBinding,
    pub selected_id: u32,
    pub max_bits: u32,
    pub dispatches: usize,
    pub terminal_readback_bytes: usize,
    pub capacity_exhausted: bool,
}
impl Qwen27bSourceGpuSession {
    /// One actual source seed; outputs are a bounded token diagnostic, not text.
    pub async fn prepare_seed(
        &self,
        seed: u32,
        maximum: usize,
        observer: Option<&mut dyn QwenSourceObserver>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Qwen27bNaturalPending> {
        check_cancel(cancelled)?;
        if self.feedback.is_some() {
            return Err(Error::State("Qwen seed has existing feedback".into()));
        }
        let progress = self.natural.seed(
            self.binding().frontier(),
            maximum,
            self.loaded.model.vocab as usize,
            seed,
        )?;
        self.prepare_natural(seed, progress, observer, cancelled)
            .await
    }
    pub async fn prepare_next(
        &self,
        observer: Option<&mut dyn QwenSourceObserver>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Qwen27bNaturalPending> {
        check_cancel(cancelled)?;
        let progress = self.natural.next(self.binding().frontier())?;
        let feedback = self.feedback.as_ref().ok_or_else(|| {
            Error::State("Qwen natural continuation needs committed feedback".into())
        })?;
        if feedback.producer != Some(self.binding())
            || !Arc::ptr_eq(feedback.lease.service(), &self.loaded.model.service)
            || !feedback.lease.is_initialized()
        {
            return Err(Error::State("Qwen original stale/foreign feedback".into()));
        }
        self.prepare_natural(feedback.id, progress, observer, cancelled)
            .await
    }
    async fn prepare_natural(
        &self,
        token: u32,
        progress: Progress,
        observer: Option<&mut dyn QwenSourceObserver>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Qwen27bNaturalPending> {
        if !self.loaded.prepared_natural_available() {
            return Err(Error::Unsupported(
                "Qwen source has no prepared natural readout".into(),
            ));
        }
        let permit = self.loaded.proposals.acquire_arc().await;
        let _call = self.loaded.call_lock.lock().await;
        check_cancel(cancelled)?;
        let host = self
            .loaded
            .source
            .source()
            .reserve_workspace(32 * 1024 * 1024)?;
        let observer = std::cell::RefCell::new(observer);
        let inner = self
            .inner
            .prepare_mode_inner(
                &[token],
                false,
                Readout::Greedy,
                #[cfg(all(feature = "gemma-bf16-qualification", target_os = "linux"))]
                None,
                Some(QwenOptions {
                    bodies: &self.loaded.bodies,
                    observer: &observer,
                    feedback: self.feedback.as_ref(),
                    cancelled,
                }),
            )
            .await?;
        if let Some(observer) = observer.borrow_mut().as_deref_mut() {
            let bytes = inner
                .readback
                .logits
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>();
            observer.completed(
                QwenSourceFrame {
                    position: self.binding().frontier() as usize,
                    kind: QwenSourceFrameKind::Value {
                        layer: 63,
                        stage: "logits",
                    },
                    offset_words: 0,
                    total_words: inner.readback.logits.len(),
                },
                &bytes,
            )?;
        }
        // Full-V observation follows the original known completion; late
        // cancellation discards Ready without fencing or publishing a root.
        check_cancel(cancelled)?;
        Ok(Qwen27bNaturalPending {
            pending: Qwen27bSourceGpuPending {
                inner,
                loaded: self.loaded.clone(),
                _host: host,
                _permit: permit,
            },
            progress,
        })
    }
    pub fn commit_natural(&mut self, pending: Qwen27bNaturalPending) -> Result<Qwen27bNaturalCall> {
        let pending_model = &pending.pending.loaded;
        if !Arc::ptr_eq(&self.loaded, pending_model) {
            return Err(Error::State("Qwen natural foreign model proposal".into()));
        }
        self.inner.validate_prepared(&pending.pending.inner)?;
        let progress = pending.progress.completed(self.binding().frontier())?;
        let mut token = pending
            .pending
            .inner
            .readback
            .device_token
            .as_ref()
            .ok_or_else(|| Error::State("Qwen actual GPU greedy packet absent".into()))?
            .clone();
        if token.producer.is_some()
            || token.id as u64 >= self.loaded.model.vocab
            || !f32::from_bits(token.max_bits).is_finite()
            || !token.lease.is_initialized()
            || !Arc::ptr_eq(token.lease.service(), &self.loaded.model.service)
        {
            return Err(Error::State(
                "Qwen fresh private Ready packet differs".into(),
            ));
        }
        let readback = self.inner.commit(pending.pending.inner)?;
        // All fallible validation precedes the original publication.
        let root = self.binding();
        token.producer = Some(root);
        self.feedback = Some(token);
        self.natural = progress;
        Ok(Qwen27bNaturalCall {
            root,
            selected_id: self.feedback.as_ref().unwrap().id,
            max_bits: self.feedback.as_ref().unwrap().max_bits,
            dispatches: readback.dispatches,
            terminal_readback_bytes: readback.bytes,
            capacity_exhausted: progress.calls == progress.limit,
        })
    }
    pub fn natural_stopped(&self) -> bool {
        self.natural.calls != 0 && self.natural.calls == self.natural.limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn genuine_progress_refuses_teacher_frontier_restart_and_exhaustion() {
        let empty = Progress::default();
        assert!(empty.next(0).is_err());
        assert!(empty.seed(1, 2, 248320, 11).is_err());
        assert!(empty.seed(0, 0, 248320, 11).is_err());
        assert!(empty.seed(0, 3, 248320, 11).is_err());
        assert!(empty.seed(0, 2, 248320, 248320).is_err());
        let first = empty.seed(0, 2, 248320, 17).unwrap().completed(0).unwrap();
        assert!(first.seed(0, 2, 248320, 11).is_err());
        assert!(first.next(0).is_err());
        let second = first.next(1).unwrap().completed(1).unwrap();
        assert!(second.next(2).is_err());
        assert!(second.completed(2).is_err());
        assert!(
            empty
                .seed(0, 1, 248320, 0)
                .unwrap()
                .completed(0)
                .unwrap()
                .next(1)
                .is_err()
        );
    }
}
