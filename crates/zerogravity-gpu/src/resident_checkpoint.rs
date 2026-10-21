//! Opt-in committed numerical checkpoint bridge. No model execution lives here.
use super::*;
use zerogravity_model_qwen35::QwenCheckpoint;

#[path = "resident_checkpoint_qualification.rs"]
#[cfg(feature = "qualification")]
mod qualification;
#[cfg(feature = "qualification")]
pub(crate) use qualification::qualify;

/// Separate diagnostic sibling: the continuing session never imports/reuploads
/// its own state. Exercises genuine context payloads around the padding boundary.
#[cfg(feature = "qualification")]
pub(super) async fn qualify_frontier(
    session: &ResidentSession,
    cpu: &zerogravity_model_qwen35::QwenDenseCpu,
    next: Option<u32>,
) -> Result<()> {
    let binding = session.binding();
    let baseline = session.model.service.resident_bytes()?;
    let uploads = session.model.service.uploaded_value_bytes();
    let exported = session.checkpoint(cpu.loaded_model()).await?;
    compare_domains(&cpu.snapshot().domain_values(), &exported.domain_values())?;
    let mut restored_cpu = cpu.loaded_model().new_session()?;
    restored_cpu.restore_checkpoint(&exported)?;
    if session.model.service.uploaded_value_bytes() != uploads {
        return Err(Error::State("checkpoint export uploaded values".into()));
    }
    let mut sibling = session.model.session().await?;
    if session.model.service.uploaded_value_bytes() != uploads + session.model.state_bytes as u64 {
        return Err(Error::State(
            "checkpoint sibling initialization upload count".into(),
        ));
    }
    sibling.import_checkpoint(&cpu.checkpoint()?).await?;
    let transferred = uploads + 2 * session.model.state_bytes as u64;
    if session.model.service.uploaded_value_bytes() != transferred {
        return Err(Error::State(
            "checkpoint sibling import upload count".into(),
        ));
    }
    compare_domains(
        &exported.domain_values(),
        &sibling.snapshot().read_domains().await?,
    )?;
    if let Some(token) = next {
        let (expected, draft) = restored_cpu.next_logits(token)?;
        let actual = sibling.forward(&[token], true).await?;
        exact(
            &expected,
            &actual.logits,
            "context checkpoint resumed logits",
        )?;
        compare_domains(&draft.domain_values(), &actual.domains)?;
    } else if sibling.prepare(&[11], false).await.is_ok() {
        return Err(Error::State(
            "full-capacity imported checkpoint advanced".into(),
        ));
    }
    drop(sibling);
    if session.binding() != binding
        || session.model.service.resident_bytes()? != baseline
        || session.model.service.pending_events()? != 0
        || session.model.service.uploaded_value_bytes() != transferred
    {
        return Err(Error::State(
            "context checkpoint changed continuing root/admission".into(),
        ));
    }
    println!(
        "H21_CONTEXT_CHECKPOINT_PASS frontier={} all42_physical_domains=bit_exact cpu_gpu_cpu=exact resumed_next_token={} canonical_kv_tail=exact continuing_root_unchanged=true admission_recovered=true",
        binding.frontier(),
        next.is_some()
    );
    Ok(())
}

impl ResidentSession {
    /// Export only a published immutable snapshot, after physical readback.
    pub async fn checkpoint(&self, source: &LoadedQwenModel) -> Result<QwenCheckpoint> {
        if source.source_identity() != self.binding().source() {
            return Err(Error::State(
                "checkpoint codec belongs to another loaded source".into(),
            ));
        }
        let snapshot = self.snapshot();
        let domains = snapshot.read_domains().await?;
        source.checkpoint_from_domains(snapshot.binding.frontier() as usize, &domains)
    }

    pub async fn import_checkpoint(&mut self, checkpoint: &QwenCheckpoint) -> Result<()> {
        self.import_checkpoint_inner(
            checkpoint,
            #[cfg(feature = "hardware-abort-probe")]
            None,
        )
        .await
    }

    async fn import_checkpoint_inner(
        &mut self,
        checkpoint: &QwenCheckpoint,
        #[cfg(feature = "hardware-abort-probe")] pause_uploaded: Option<
            &std::sync::atomic::AtomicBool,
        >,
    ) -> Result<()> {
        self.model.service.ensure_usable()?;
        self.authority.preflight_replacement()?;
        // Payload fields are private to the root codec. Check its local source
        // binding and this physical capacity BEFORE allocation, queue or upload.
        if checkpoint.source_identity() != self.binding().source() {
            return Err(Error::State(
                "checkpoint belongs to another loaded source".into(),
            ));
        }
        if checkpoint.position() > self.model.capacity {
            return Err(Error::Limit("checkpoint exceeds resident capacity".into()));
        }
        let packed = pack(&self.model.states, self.model.capacity, checkpoint)?;
        let attempt = self.authority.begin_import(checkpoint.position() as u64)?;
        let _admission = self.model.allocation.lock().await;
        if self.model.state_bytes > self.model.service.remaining_capacity()? {
            return Err(Error::Limit(
                "resident checkpoint admission exhausted".into(),
            ));
        }
        let ticket = self.model.service.queue().submit(
            self.model.service.budget(),
            ResourceOwner {
                model: self.model.service.owner().model.clone(),
                session: format!("checkpoint-{}", self.binding().session().get()),
            },
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: self.model.state_bytes as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let mut initialization = self
            .model
            .service
            .begin_initialization(attempt.base().root())?;
        let mut domains = Vec::with_capacity(self.model.states.len());
        for (requirement, buffers) in self.model.states.iter().zip(&packed) {
            let mut data = buffers.iter();
            let mut allocate = |elements: usize| {
                let bytes = data
                    .next()
                    .ok_or_else(|| Error::State("checkpoint buffer coverage".into()))?;
                if elements.checked_mul(4) != Some(bytes.len()) {
                    return Err(Error::State("checkpoint physical byte coverage".into()));
                }
                initialization.upload_bytes(bytes, BufferRole::RecurrentState)
            };
            domains.push(make_domain(requirement, &mut allocate)?);
            if data.next().is_some() {
                return Err(Error::State("checkpoint extra buffer".into()));
            }
        }
        #[cfg(feature = "hardware-abort-probe")]
        if let Some(seen) = pause_uploaded {
            seen.store(true, std::sync::atomic::Ordering::Release);
            // Diagnostic only: real H2D copies have been submitted, their
            // original handles remain owned by initialization before this await.
            std::future::pending::<()>().await;
        }
        let ready = initialization.complete_sync().await?;
        slot.finish()?;
        self.model.validate_domains(&domains, true)?;
        let leases = domains
            .iter()
            .flat_map(|domain| &domain.leases)
            .collect::<Vec<_>>();
        if ready.expected_root() != attempt.base().root()
            || !Arc::ptr_eq(ready.service(), &self.model.service)
            || leases.len() != ready.outputs().len()
            || leases
                .iter()
                .zip(ready.outputs())
                .any(|(a, b)| a.id() != b.id())
        {
            return Err(Error::State("checkpoint actual Ready coverage".into()));
        }
        // Only the infallible payload swap and logical commit occur under the
        // physical fence lock. Displaced handles/leases drop AFTER unlocking.
        let mut displaced = domains;
        self.model.service.publish_if_usable(|| {
            let transition = self.authority.prepare_publish(attempt)?;
            std::mem::swap(&mut self.domains, &mut displaced);
            transition.commit();
            Ok(())
        })?;
        drop(displaced);
        Ok(())
    }
}

fn pack(
    requirements: &[StateRequirement],
    capacity: usize,
    checkpoint: &QwenCheckpoint,
) -> Result<Vec<Vec<Vec<u8>>>> {
    let domains = checkpoint
        .domain_values()
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();
    if requirements.len() != domains.len() {
        return Err(Error::State("checkpoint state domain count".into()));
    }
    let bytes = |values: &[f32], elements: usize| -> Result<Vec<u8>> {
        if values.len() > elements || values.iter().any(|v| !v.is_finite()) {
            return Err(Error::State("checkpoint invalid physical values".into()));
        }
        let mut output = vec![
            0u8;
            elements
                .checked_mul(4)
                .ok_or_else(|| Error::Limit("checkpoint buffer size".into()))?
        ];
        for (word, value) in output.chunks_exact_mut(4).zip(values) {
            word.copy_from_slice(&value.to_bits().to_ne_bytes());
        }
        Ok(output)
    };
    requirements
        .iter()
        .map(|r| {
            let values = domains
                .get(&r.slot)
                .ok_or_else(|| Error::State("checkpoint state domain coverage".into()))?;
            match r.layout {
                StateLayout::Vector { elements } if values.len() == elements => {
                    Ok(vec![bytes(values, elements)?])
                }
                StateLayout::Kv { slab, capacity: n } if n == capacity => {
                    let active = slab
                        .checked_mul(checkpoint.position())
                        .ok_or_else(|| Error::Limit("checkpoint active KV size".into()))?;
                    if active.checked_mul(2) != Some(values.len()) {
                        return Err(Error::State("checkpoint active KV geometry".into()));
                    }
                    let full = slab
                        .checked_mul(capacity)
                        .ok_or_else(|| Error::Limit("checkpoint physical KV size".into()))?;
                    Ok(vec![
                        bytes(&values[..active], full)?,
                        bytes(&values[active..], full)?,
                    ])
                }
                _ => Err(Error::State("checkpoint physical layout".into())),
            }
        })
        .collect()
}
