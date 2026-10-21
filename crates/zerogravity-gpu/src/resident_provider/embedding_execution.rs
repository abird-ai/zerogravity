//! Original-byte embedding gather through the existing physical source cache.
use super::*;
use zerogravity_core::accelerator::EmbeddingRegion;

impl CubeResidentProvider {
    /// This selection is independent of a vocabulary projection using the
    /// same tensor. Both operations share its original source residency key.
    pub async fn try_embedding_async(
        &self,
        region: EmbeddingRegion<'_>,
    ) -> Result<RegionExecution> {
        if !self.selected.contains("embedding") {
            return Ok(RegionExecution::Unsupported);
        }
        // Host geometry/token/source checks precede queueing or physical
        // submission. A malformed request cannot fence an unsubmitted card.
        region.validate()?;
        if !region.source_identity.is_bound() {
            return Err(Error::Unsupported(
                "source embedding requires bound identity".into(),
            ));
        }
        if region.width > self.max_rows {
            return Err(Error::Limit(
                "source embedding output exceeds provider row bound".into(),
            ));
        }
        let layout =
            crate::embedding::EmbeddingLayout::new(region.kind, region.vocabulary, region.width)?;
        let source_region = MatVecRegion {
            source_identity: region.source_identity,
            tensor_name: region.tensor_name,
            kind: region.kind,
            columns: region.width,
            rows: region.vocabulary,
            packed_weights: region.packed_weights,
            // ResidencyKey and StaticSource are immutable source metadata;
            // neither consumes a live activation. No host Q8 preparation.
            activation: &[],
        };
        let key = ResidencyKey::from_region(source_region)?;
        let source_bytes = StaticSource::gpu_bytes_for(source_region)?;
        if source_bytes != layout.source_bytes {
            return Err(Error::State(
                "source embedding cache byte geometry mismatch".into(),
            ));
        }
        source_accumulation::ensure_capability(&self.device.client(), SourceAccumulation::WideF64)?;
        let value_bytes = region
            .width
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("source embedding output bytes overflow".into()))?;
        // Both values AND GPU-written status belong to the admitted/read-back
        // output cohort. Status is not uncharged or implicitly CPU-validated.
        let output_bytes = value_bytes
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("source embedding readback bytes overflow".into()))?;
        let dynamic_bytes = 4usize;
        if dynamic_bytes.checked_add(output_bytes) != Some(layout.workspace_bytes) {
            return Err(Error::State(
                "source embedding workspace differs from layout".into(),
            ));
        }
        let count = u32::try_from(region.width)
            .map_err(|_| Error::Limit("source embedding launch grid overflow".into()))?;
        let ticket = self.queue.submit(
            &self.service,
            self.owner.clone(),
            &[
                AllocationRequest {
                    kind: AllocationClass::ImmutableWeights,
                    bytes: u64::try_from(source_bytes)
                        .map_err(|_| Error::Limit("source embedding weights overflow".into()))?,
                },
                AllocationRequest {
                    kind: AllocationClass::Activations,
                    bytes: dynamic_bytes as u64,
                },
                AllocationRequest {
                    kind: AllocationClass::Output,
                    bytes: u64::try_from(output_bytes)
                        .map_err(|_| Error::Limit("source embedding readback overflow".into()))?,
                },
            ],
            None,
        )?;
        // Fair waiting owns no source-cache lock and has not submitted GPU
        // commands. Future abandonment here cancels without physical fencing.
        let mut worker_slot = ticket.wait().await?;
        let client = self.device.client();
        crate::physical_quarantine::ensure_service_safe(&client)?;
        let mut resident = self.cache.lock().await;
        crate::physical_quarantine::ensure_service_safe(&client)?;
        let mut staged_source = match resident.cached(&key) {
            Some(existing) => {
                if existing.bytes != source_bytes
                    || existing.elements != source_bytes
                    || existing.host.kind != region.kind
                    || existing.host.rows != region.vocabulary
                    || existing.host.cols != region.width
                    || existing.handle.service != client.service_id()
                {
                    return Err(Error::State(
                        "source embedding resident lease geometry/service changed".into(),
                    ));
                }
                None
            }
            None => {
                // The shared descriptor validates original half scales and
                // preserves original Q8_0/Q6_K bytes, once per source miss.
                // It never computes a token-row embedding or dynamic scale.
                let source = StaticSource::from_region(source_region)?;
                if source.0.gpu_bytes != source_bytes
                    || source.0.gpu_elements != source_bytes
                    || source.1.len() != source_bytes
                {
                    return Err(Error::State(
                        "source embedding prepared byte geometry mismatch".into(),
                    ));
                }
                Some(source)
            }
        };
        resident.ensure_source_and_scratch_room(&key, source_bytes, dynamic_bytes, output_bytes)?;

        worker_slot.begin_submission()?;
        let physical_work = self.service.begin(&client)?;
        let fence_flag = self.service.fence_flag();
        let weight = resident.get_or_try_upload_with_scratch(
            key,
            source_bytes,
            dynamic_bytes,
            output_bytes,
            || {
                let (host, gpu_source) = staged_source.take().ok_or_else(|| {
                    Error::State("missing staged source embedding on cache miss".into())
                })?;
                let handle = client.create_from_slice(&gpu_source);
                Ok(ResidentWeightLease {
                    client: client.clone(),
                    fence_flag,
                    handle,
                    bytes: host.gpu_bytes,
                    elements: host.gpu_elements,
                    host,
                })
            },
        )?;
        let source = weight.handle.clone();
        resident.ensure_weight_and_scratch_budget(dynamic_bytes, output_bytes)?;
        let mut in_flight =
            InFlightRegion::submit(&mut resident, &self.events, dynamic_bytes, output_bytes)?;
        let token = client.create_from_slice(u32::as_bytes(&[region.token]));
        let output = client.empty(value_bytes);
        let status = client.empty(value_bytes);
        in_flight.retain_gpu_buffers(
            &client,
            &[
                source.clone(),
                token.clone(),
                output.clone(),
                status.clone(),
            ],
        )?;
        // SAFETY: complete source, token and output/status dimensions were
        // validated before submission. The kernel additionally bounds-checks
        // the runtime token before every source row access.
        unsafe {
            crate::embedding::gather_quantized::launch_unchecked(
                &client,
                CubeCount::Static(count.div_ceil(32), 1, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(source, source_bytes),
                BufferArg::from_raw_parts(token, 1),
                BufferArg::from_raw_parts(output.clone(), region.width),
                BufferArg::from_raw_parts(status.clone(), region.width),
                region.vocabulary,
                region.width,
                layout.q6_k,
            );
        }
        #[cfg(feature = "hardware-abort-probe")]
        crate::physical_abort_probe::after_physical_launch().await;
        let results = client
            .read_async(vec![output, status])
            .await
            .map_err(|e| Error::State(format!("embedding physical readback failed: {e}")))?;
        if results.len() != 2 || results[0].len() != value_bytes || results[1].len() != value_bytes
        {
            return Err(Error::State(
                "incomplete source embedding physical readback".into(),
            ));
        }
        if results[1]
            .chunks_exact(4)
            .any(|p| u32::from_le_bytes(p.try_into().expect("checked embedding status bytes")) != 0)
        {
            return Err(Error::State(
                "physical source embedding reported invalid token/scale/output".into(),
            ));
        }
        let values: Vec<f32> = results[0]
            .chunks_exact(4)
            .map(|p| f32::from_le_bytes(p.try_into().expect("checked embedding output bytes")))
            .collect();
        region.validate_completed(&values)?;
        in_flight.complete(output_bytes)?;
        physical_work.complete()?;
        worker_slot.finish()?;
        Ok(RegionExecution::Completed(values))
    }
}
