//! Physical import of source-bound numerical state domains; codec stays with the model.
use super::*;

impl ResidentSession {
    pub async fn import_state_domains(
        &mut self,
        source: zerogravity_core::accelerator::SourceIdentity,
        position: usize,
        domains: &[(StateSlot, Vec<f32>)],
    ) -> Result<()> {
        self.import_state_domains_inner(
            source,
            position,
            domains,
            #[cfg(feature = "hardware-abort-probe")]
            None,
        )
        .await
    }

    pub(super) async fn import_state_domains_inner(
        &mut self,
        source: zerogravity_core::accelerator::SourceIdentity,
        position: usize,
        domains: &[(StateSlot, Vec<f32>)],
        #[cfg(feature = "hardware-abort-probe")] pause_uploaded: Option<
            &std::sync::atomic::AtomicBool,
        >,
    ) -> Result<()> {
        self.model.service.ensure_usable()?;
        self.authority.preflight_replacement()?;
        // Validate caller-bound domains before allocation, queue or upload.
        if source != self.binding().source() {
            return Err(Error::State(
                "checkpoint belongs to another loaded source".into(),
            ));
        }
        if position > self.model.capacity {
            return Err(Error::Limit("checkpoint exceeds resident capacity".into()));
        }
        let packed = pack(&self.model.states, self.model.capacity, position, domains)?;
        let attempt = self.authority.begin_import(position as u64)?;
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
    position: usize,
    domain_values: &[(StateSlot, Vec<f32>)],
) -> Result<Vec<Vec<Vec<u8>>>> {
    let domains = domain_values
        .iter()
        .map(|(slot, values)| (*slot, values))
        .collect::<std::collections::BTreeMap<_, _>>();
    if domains.len() != domain_values.len() {
        return Err(Error::State("checkpoint duplicate state domain".into()));
    }
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
                        .checked_mul(position)
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

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::accelerator::program::{ConvDomain, KvDomain};

    fn requirements() -> [StateRequirement; 2] {
        [
            StateRequirement {
                slot: StateSlot::Conv(ConvDomain(0)),
                layout: StateLayout::Vector { elements: 2 },
                bytes: 8,
            },
            StateRequirement {
                slot: StateSlot::Kv(KvDomain(0)),
                layout: StateLayout::Kv {
                    slab: 2,
                    capacity: 4,
                },
                bytes: 64,
            },
        ]
    }
    fn domains() -> Vec<(StateSlot, Vec<f32>)> {
        vec![
            (
                StateSlot::Conv(ConvDomain(0)),
                vec![-0.0, f32::from_bits(1)],
            ),
            (
                StateSlot::Kv(KvDomain(0)),
                vec![-0.0, 1.25, -2.5, f32::from_bits(1)],
            ),
        ]
    }
    fn bits(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|v| u32::from_ne_bytes(v.try_into().unwrap()))
            .collect()
    }
    #[test]
    fn state_import_preserves_active_bits_and_canonical_zero_padding() {
        let packed = pack(&requirements(), 4, 1, &domains()).unwrap();
        assert_eq!(bits(&packed[0][0]), [0x8000_0000, 1]);
        assert_eq!(
            bits(&packed[1][0]),
            [0x8000_0000, 1.25f32.to_bits(), 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            bits(&packed[1][1]),
            [(-2.5f32).to_bits(), 1, 0, 0, 0, 0, 0, 0]
        );
        let mut empty = domains();
        empty[1].1.clear();
        let packed = pack(&requirements(), 4, 0, &empty).unwrap();
        assert!(packed[1].iter().flatten().all(|v| *v == 0));
    }
    #[test]
    fn state_import_rejects_duplicate_missing_foreign_and_invalid_domains() {
        let requirement = requirements();
        let good = domains();
        assert!(pack(&requirement, 4, 1, &[good[0].clone(), good[0].clone()]).is_err());
        assert!(pack(&requirement, 4, 1, &good[..1]).is_err());
        let mut foreign = good.clone();
        foreign[0].0 = StateSlot::Conv(ConvDomain(1));
        assert!(pack(&requirement, 4, 1, &foreign).is_err());
        for (slot, values) in [
            (0, vec![0.0]),
            (0, vec![f32::NAN, 0.0]),
            (0, vec![f32::INFINITY, 0.0]),
            (1, vec![0.0; 3]),
            (1, vec![0.0; 5]),
            (1, vec![0.0, 0.0, f32::NEG_INFINITY, 0.0]),
        ] {
            let mut bad = good.clone();
            bad[slot].1 = values;
            assert!(
                pack(&requirement, 4, 1, &bad).is_err(),
                "invalid domain {slot}"
            );
        }
        assert!(pack(&requirement, 3, 1, &good).is_err());
        assert!(pack(&requirement, 4, 2, &good).is_err());
        let mut overflow = requirement;
        overflow[1].layout = StateLayout::Kv {
            slab: usize::MAX,
            capacity: 4,
        };
        assert!(pack(&overflow, 4, 2, &good).is_err());
    }
}
