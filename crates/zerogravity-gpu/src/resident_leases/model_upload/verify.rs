//! Independent bounded original-member readback after completed model arithmetic.
use super::coverage::Region;
use super::*;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};

/// Read-only itinerary; never credits upload progress or device completion.
fn visit_completed_windows<F>(
    shape: SourceWeightLayout,
    raw: &[u8],
    cancelled: Option<&AtomicBool>,
    mut verify: F,
) -> Result<usize>
where
    F: FnMut(Region, &[u8]) -> Result<()>,
{
    not_cancelled(cancelled)?;
    if raw.len() != shape.bytes() {
        return Err(Error::State(
            "complete learned W readback extent required".into(),
        ));
    }
    let row_bytes = shape
        .columns()
        .checked_mul(2)
        .filter(|n| *n > 0 && *n <= 1024 * 1024)
        .ok_or_else(|| Error::Limit("completed W source row extent".into()))?;
    let rows_per_window = 1024 * 1024 / row_bytes;
    let members = Coverage::new(shape);
    let mut destination = 0usize;
    let mut count = 0usize;
    for (member, &rows) in members.members().iter().enumerate() {
        for first in (0..rows).step_by(rows_per_window) {
            not_cancelled(cancelled)?;
            let end = first
                .checked_add(rows_per_window)
                .ok_or_else(|| Error::Limit("completed W source row end".into()))?
                .min(rows);
            let bytes = (end - first)
                .checked_mul(row_bytes)
                .ok_or_else(|| Error::Limit("completed W source window bytes".into()))?;
            let next = destination
                .checked_add(bytes)
                .filter(|n| *n <= raw.len())
                .ok_or_else(|| Error::Limit("completed W source destination extent".into()))?;
            if bytes == 0 || bytes > 1024 * 1024 {
                return Err(Error::State(
                    "bounded completed W source window required".into(),
                ));
            }
            let region = Region {
                member,
                first,
                end,
                destination_offset: destination,
                bytes,
            };
            verify(region, &raw[destination..next])?;
            destination = next;
            count += 1;
        }
    }
    not_cancelled(cancelled)?;
    if destination != shape.bytes() {
        return Err(Error::State(
            "complete ordered learned W source coverage required".into(),
        ));
    }
    Ok(count)
}

impl ModelWeight {
    /// Caller supplies the SAME lease's whole D2H only inside its trusted
    /// completion/retirement callback. Source equality grants no Ready proof.
    pub(crate) fn verify_completed_readback(
        &self,
        raw: &[u8],
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        self.require_initialized()?;
        let shape = self.owner.layout;
        let source = &self.owner.source;
        let members = Coverage::new(shape);
        if self.owner.matrices.len() != members.members().len()
            || self
                .owner
                .matrices
                .iter()
                .zip(members.members())
                .any(|(matrix, &rows)| matrix.rows() != rows || matrix.columns() != shape.columns())
        {
            return Err(Error::State(
                "completed W original ordered source members".into(),
            ));
        }
        let count = visit_completed_windows(shape, raw, cancelled, |region, actual| {
            let matrix = &self.owner.matrices[region.member];
            // read_rows retains its own original complete-chunk admission.
            // This one window drops before the next; raw is already admitted.
            let window = match cancelled {
                Some(flag) => {
                    source.read_rows_cancellable(matrix, region.first..region.end, flag)?
                }
                None => source.read_rows(matrix, region.first..region.end)?,
            };
            let offset = region
                .first
                .checked_mul(shape.columns())
                .and_then(|n| n.checked_mul(2))
                .and_then(|n| matrix.tensor().byte_offset.checked_add(n as u64))
                .ok_or_else(|| Error::Limit("completed W original source byte offset".into()))?;
            let original = window.matrix();
            if window.row_range() != (region.first..region.end)
                || window.source_offset() != offset
                || window.as_slice().len() != region.bytes
                || original.source_identity() != source.source_identity()
                || original.tensor().name != matrix.tensor().name
                || original.tensor().kind != matrix.tensor().kind
                || original.tensor().byte_offset != matrix.tensor().byte_offset
                || original.axes() != matrix.axes()
                || original.rows() != matrix.rows()
                || original.columns() != matrix.columns()
                || actual != window.as_slice()
            {
                return Err(Error::State(
                    "completed learned W differs from fresh original member/rows/bytes".into(),
                ));
            }
            not_cancelled(cancelled)
        })?;
        self.require_initialized()?;
        not_cancelled(cancelled)?;
        Ok(count)
    }

    pub(crate) fn copy_counts(&self) -> Result<(usize, usize)> {
        self.require_initialized()?;
        let progress = self.owner.progress()?;
        Ok((progress.enqueued, progress.checkpoints))
    }
    // Both old independent flights and the sealed current-flight adapter use
    // this exact original member/range/source validation. No expected operands.
    pub(super) fn checked_verification_window(
        &self,
        matrix: &VerifiedMatrix,
        rows: std::ops::Range<usize>,
        destination: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<(Arc<TensorWindow>, ResidentReadback, usize)> {
        not_cancelled(cancelled)?;
        let source = &self.owner.source;
        let shape = self.owner.layout;
        let window = Arc::new(match cancelled {
            Some(flag) => source.read_rows_cancellable(matrix, rows.clone(), flag)?,
            None => source.read_rows(matrix, rows.clone())?,
        });
        let bytes = rows
            .end
            .checked_sub(rows.start)
            .and_then(|n| n.checked_mul(matrix.columns()))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| Error::Limit("full24 independent W readback extent".into()))?;
        if bytes == 0
            || bytes > 1024 * 1024
            || window.as_slice().len() != bytes
            || window.row_range() != rows
            || window.matrix().tensor().name != matrix.tensor().name
            || window.matrix().source_identity() != source.source_identity()
            || window.source_offset()
                != matrix
                    .tensor()
                    .byte_offset
                    .checked_add((rows.start * matrix.columns() * 2) as u64)
                    .ok_or_else(|| Error::Limit("independent source byte offset".into()))?
            || window.matrix().axes() != matrix.axes()
            || window.matrix().tensor().kind != matrix.tensor().kind
            || window.matrix().rows() != matrix.rows()
            || window.matrix().columns() != matrix.columns()
            || destination
                .checked_add(bytes)
                .is_none_or(|n| n > shape.bytes())
        {
            return Err(Error::State(
                "full24 independent exact source member/rows/readback coverage".into(),
            ));
        }
        let read = self.lease().readback(destination, bytes)?;
        Ok((window, read, bytes))
    }
    /// Fresh source windows are read again. No complete learned W host pack is
    /// retained; each original read-only flight owns its exact source window.
    pub(crate) async fn verify_source_windows(
        self: &Arc<Self>,
        completion_owner: crate::physical_quarantine::CompletionOwner,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        self.require_initialized()?;
        let source = &self.owner.source;
        let service = self.lease().service();
        let admission = Arc::new(source.reserve_workspace(2 * 1024 * 1024)?);
        let shape = self.owner.layout;
        let mut destination = 0usize;
        let mut count = 0usize;
        for matrix in &self.owner.matrices {
            let rows_per_window = 1024 * 1024 / (matrix.columns() * 2);
            for first in (0..matrix.rows()).step_by(rows_per_window) {
                not_cancelled(cancelled)?;
                let end = (first + rows_per_window).min(matrix.rows());
                let (window, read, bytes) =
                    self.checked_verification_window(matrix, first..end, destination, cancelled)?;
                let ticket = service.queue().submit(
                    service.budget(),
                    service.owner().clone(),
                    &[AllocationRequest {
                        kind: AllocationClass::TransferStaging,
                        bytes: bytes as u64,
                    }],
                    None,
                )?;
                let mut slot = ticket.wait().await?;
                not_cancelled(cancelled)?;
                slot.begin_submission()?;
                let mut flight =
                    service.submit(0, &[(self.lease().clone(), DeviceAccess::Read)])?;
                flight.retain_completion_owner(completion_owner.clone())?;
                flight.retain_completion_owner(self.clone())?;
                flight.retain_completion_owner(window.clone())?;
                flight.retain_completion_owner(admission.clone())?;
                let raw=flight.complete_read_only_readback(&[read],|raw| {
                    if raw.len()!=1 || &raw[0][..]!=window.as_slice() {
                        return Err(Error::State("full24 actual immutable learned W differs from fresh original source".into()));
                    }
                    Ok(())
                }).await?;
                drop(raw);
                slot.finish()?;
                destination += bytes;
                count += 1;
            }
        }
        if destination != shape.bytes() {
            return Err(Error::State(
                "full24 complete source W readback required".into(),
            ));
        }
        Ok(count)
    }
}

#[cfg(test)]
mod completed_tests {
    use super::*;

    #[test]
    fn completed_itinerary_matches_existing_upload_members_and_final_tails() {
        for (shape, expected) in [
            (SourceWeightLayout::GdnQkvz, 16),
            (SourceWeightLayout::GdnBa, 2),
            (SourceWeightLayout::GdnOutput, 4),
            (SourceWeightLayout::AttentionQkv, 10),
            (SourceWeightLayout::AttentionOutput, 4),
            (SourceWeightLayout::MlpGateUp, 14),
            (SourceWeightLayout::MlpDown, 8),
        ] {
            let raw = vec![0x55; shape.bytes()];
            let mut upload = Coverage::new(shape);
            let count = visit_completed_windows(shape, &raw, None, |region, actual| {
                assert_eq!(upload.next()?, Some(region));
                assert_eq!(actual.len(), region.bytes);
                assert_eq!(
                    actual,
                    &raw[region.destination_offset..region.destination_offset + region.bytes]
                );
                upload.begin(region)?;
                upload.checkpoint(region)?;
                Ok(())
            })
            .unwrap();
            upload.require_complete().unwrap();
            assert_eq!(count, expected);
        }
    }

    #[test]
    fn wrong_complete_extent_refuses_before_any_fresh_source_visit() {
        let shape = SourceWeightLayout::GdnBa;
        for bytes in [0, shape.bytes() - 1, shape.bytes() + 1] {
            let mut visits = 0;
            let raw = vec![0; bytes];
            assert!(
                visit_completed_windows(shape, &raw, None, |_, _| {
                    visits += 1;
                    Ok(())
                })
                .is_err()
            );
            assert_eq!(visits, 0);
        }
    }

    #[test]
    fn equal_extent_gate_up_members_remain_ordered_and_mutations_refuse() {
        let shape = SourceWeightLayout::MlpGateUp;
        let half = shape.bytes() / 2;
        let mut raw = vec![0x19; shape.bytes()];
        raw[half..].fill(0xa3);
        let compare = |region: Region, actual: &[u8]| {
            let expected = [0x19, 0xa3][region.member];
            if actual.iter().any(|byte| *byte != expected) {
                return Err(Error::State("synthetic fresh member mismatch".into()));
            }
            Ok(())
        };
        assert_eq!(
            visit_completed_windows(shape, &raw, None, compare).unwrap(),
            14
        );
        raw[half] ^= 1;
        assert!(visit_completed_windows(shape, &raw, None, compare).is_err());
        raw[half] ^= 1;
        raw.rotate_left(half);
        assert!(visit_completed_windows(shape, &raw, None, compare).is_err());
    }

    #[test]
    fn final_down_remainder_is_compared_including_its_last_byte() {
        let shape = SourceWeightLayout::MlpDown;
        let mut raw = vec![0x57; shape.bytes()];
        let mut final_region = None;
        let count = visit_completed_windows(shape, &raw, None, |region, actual| {
            assert!(actual.iter().all(|byte| *byte == 0x57));
            final_region = Some(region);
            Ok(())
        })
        .unwrap();
        assert_eq!(count, 8);
        let tail = final_region.unwrap();
        assert_eq!((tail.first, tail.end, tail.bytes), (1022, 1024, 14336));
        assert_eq!(tail.destination_offset + tail.bytes, raw.len());
        *raw.last_mut().unwrap() ^= 1;
        assert!(
            visit_completed_windows(shape, &raw, None, |_, actual| {
                if actual.iter().any(|byte| *byte != 0x57) {
                    return Err(Error::State("synthetic final byte changed".into()));
                }
                Ok(())
            })
            .is_err()
        );
    }

    #[test]
    fn cancellation_refuses_before_source_and_between_source_windows() {
        let raw = vec![0; SourceWeightLayout::GdnBa.bytes()];
        let cancelled = AtomicBool::new(true);
        let mut visits = 0;
        assert!(
            visit_completed_windows(SourceWeightLayout::GdnBa, &raw, Some(&cancelled), |_, _| {
                visits += 1;
                Ok(())
            })
            .is_err()
        );
        assert_eq!(visits, 0);
        cancelled.store(false, Ordering::Release);
        assert!(
            visit_completed_windows(SourceWeightLayout::GdnBa, &raw, Some(&cancelled), |_, _| {
                visits += 1;
                cancelled.store(true, Ordering::Release);
                Ok(())
            })
            .is_err()
        );
        assert_eq!(visits, 1);
    }

    #[test]
    fn cancellation_after_last_window_does_not_return_a_verified_count() {
        let raw = vec![0; SourceWeightLayout::GdnBa.bytes()];
        let cancelled = AtomicBool::new(false);
        let mut visits = 0;
        assert!(
            visit_completed_windows(SourceWeightLayout::GdnBa, &raw, Some(&cancelled), |_, _| {
                visits += 1;
                if visits == 2 {
                    cancelled.store(true, Ordering::Release);
                }
                Ok(())
            })
            .is_err()
        );
        assert_eq!(visits, 2);
    }

    #[test]
    fn source_failure_is_propagated_without_mutating_raw_or_continuing() {
        let raw = vec![0x6b; SourceWeightLayout::GdnBa.bytes()];
        let mut visits = 0;
        let error = visit_completed_windows(SourceWeightLayout::GdnBa, &raw, None, |_, _| {
            visits += 1;
            Err(Error::State("synthetic source read failure".into()))
        })
        .unwrap_err();
        assert!(
            matches!(error, Error::State(message) if message == "synthetic source read failure")
        );
        assert_eq!(visits, 1);
        assert!(raw.iter().all(|byte| *byte == 0x6b));
    }
    #[test]
    fn qwen2b_completed_windows_preserve_upload_order_and_full_tail() {
        for (shape, count) in [
            (SourceWeightLayout::Qwen2bMlpGateUp, 48),
            (SourceWeightLayout::Qwen2bMlpDown, 25),
        ] {
            // Callback observes ranges only; no 72MiB expected/shadow pack.
            let raw = vec![0; shape.bytes()];
            let mut upload = Coverage::new(shape);
            let mut last = None;
            let visits = visit_completed_windows(shape, &raw, None, |region, _| {
                assert_eq!(upload.next()?, Some(region));
                upload.begin(region)?;
                upload.checkpoint(region)?;
                last = Some(region);
                Ok(())
            })
            .unwrap();
            assert_eq!(visits, count);
            upload.require_complete().unwrap();
            let last = last.unwrap();
            assert_eq!(last.destination_offset + last.bytes, shape.bytes());
            if shape == SourceWeightLayout::Qwen2bMlpDown {
                assert_eq!((last.first, last.end, last.bytes), (2040, 2048, 98304));
            }
        }
    }
}
