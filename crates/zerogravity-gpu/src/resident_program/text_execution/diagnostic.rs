//! Diagnostic access to the original prepared trig kernel; no reference data.
use super::{cold::view, *};
use cubecl::prelude::*;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};

impl Cold {
    pub(in crate::resident_program) async fn diagnostic_trig(
        &self,
        angles: &[f32],
        owner: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<Vec<f32>> {
        if angles.is_empty() || !angles.len().is_multiple_of(32) {
            return Err(Error::Malformed(
                "source trig complete physical rows".into(),
            ));
        }
        let input_bytes = angles
            .len()
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("source trig input extent".into()))?;
        let output_bytes = input_bytes
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("source trig output extent".into()))?;
        let required = layout::sum([input_bytes, output_bytes, 4])?;
        self.service.ensure_usable()?;
        self.native
            .require_live(self.service.client(), self.source.source_identity())?;
        let mut slot = self
            .service
            .queue()
            .submit(
                self.service.budget(),
                self.service.owner().clone(),
                &[AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: required as u64,
                }],
                None,
            )?
            .wait()
            .await?;
        slot.begin_submission()?;
        let mut init = self.service.begin_initialization(0)?;
        init.retain_completion_owner(owner.clone())?;
        let input = init.upload_bytes(f32::as_bytes(angles), BufferRole::Activation)?;
        let output = init.allocate_empty(output_bytes, BufferRole::Output)?;
        let status = init.upload_bytes(&0u32.to_ne_bytes(), BufferRole::Scratch)?;
        drop(init.complete_sync().await?);
        let table = self.trig_table()?;
        crate::resident_program::BufferRange::new(&table)?
            .require_exact_len(std::mem::size_of_val(&crate::source_trig::cpu::TABLES))?;
        let accesses = [
            (input.clone(), DeviceAccess::Read),
            (self.tables[2].clone(), DeviceAccess::Read),
            (output.clone(), DeviceAccess::Write),
            (status.clone(), DeviceAccess::Write),
        ];
        let mut flight = self.service.submit(0, &accesses)?;
        flight.retain_completion_owner(owner)?;
        for row in 0..angles.len() / 32 {
            let x = view(&input, row * 128, 128)?;
            let y = view(&output, row * 256, 256)?;
            // SAFETY: exact admitted row views; the original sealed constructor
            // and flight retain all owners through completion or quarantine.
            unsafe {
                crate::source_trig::cpu::sincos_angles::launch_unchecked(
                    self.service.client(),
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(32),
                    BufferArg::from_raw_parts(x, 32),
                    BufferArg::from_raw_parts(table.clone(), crate::source_trig::cpu::TABLES.len()),
                    BufferArg::from_raw_parts(y, 64),
                    BufferArg::from_raw_parts(status.handle(), 1),
                    32,
                );
            }
        }
        let outputs = [output.clone(), status.clone()];
        let (ready, read) = flight
            .complete_with_readback(
                &outputs,
                &[output.readback(0, output_bytes)?, status.readback(0, 4)?],
                |read| {
                    if read.len() != 2
                        || read[0].len() != output_bytes
                        || read[1].len() != 4
                        || read[1].iter().any(|byte| *byte != 0)
                    {
                        return Err(Error::State("source trig physical extent/status".into()));
                    }
                    Ok(())
                },
            )
            .await?;
        let values = read[0]
            .chunks_exact(4)
            .map(|word| f32::from_ne_bytes(word.try_into().expect("checked float word")))
            .collect();
        slot.finish()?;
        drop((ready, read, outputs, accesses, input, output, status));
        Ok(values)
    }
}
