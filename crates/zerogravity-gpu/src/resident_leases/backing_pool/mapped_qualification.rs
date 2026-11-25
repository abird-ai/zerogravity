//! Actual coherent mapped backing through the existing portable GPU probe.
use super::*;
use cubecl::prelude::*;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, ResourceArbiter};

pub(crate) async fn qualify_host_visible<T: std::any::Any + Send + Sync>(
    service: &Arc<ResidentService>,
    sibling: &Arc<ResidentService>,
    host_key: &PhysicalDeviceKey,
    source_owner: Arc<T>,
) -> Result<()> {
    const BYTES: usize = 4096;
    const OFFSET: usize = 512;
    const VIEW: usize = 1024;
    let initial = service.resident_bytes()?;
    let uploaded = service.uploaded_value_bytes();
    let arbiter = ResourceArbiter::process_shared();
    let host_before = arbiter.snapshot(host_key)?.reserved;
    let metadata_bytes = metadata_bytes(1, 1, 2)?;
    let metadata = require_metadata(
        service,
        arbiter
            .admit(
                host_key,
                ResourceOwner::new("mapped-backing", "metadata")?,
                &[AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: metadata_bytes as u64,
                }],
            )?
            .into_host_workspace(service.source_identity(), source_owner)?,
        metadata_bytes,
    )?;
    let plane = ResidentPlaneLayout {
        bytes: BYTES,
        role: BufferRole::Scratch,
        alignment: 4,
    };
    require_layout(service, &[plane])?;
    let backing = BackingPlane::allocate(service, plane, &metadata)?;
    drop(metadata);
    let full = backing.view(
        ResidentViewLayout {
            plane: 0,
            offset: 0,
            bytes: BYTES,
            role: plane.role,
            alignment: plane.alignment,
        },
        None,
        None,
    )?;
    let sub = backing.view(
        ResidentViewLayout {
            plane: 0,
            offset: OFFSET,
            bytes: VIEW,
            role: plane.role,
            alignment: plane.alignment,
        },
        None,
        None,
    )?;
    assert_eq!(service.resident_bytes()?, initial + BYTES);
    let mut expected = [0x35; BYTES];
    let payload = [0x9a; VIEW];
    // Same physical service is not the same source or logical lease.
    assert!(
        sibling
            .initialize_host_visible(&sub, &payload, 0)
            .await
            .is_err()
    );
    assert!(
        service
            .initialize_host_visible(&sub, &payload[..VIEW - 1], 0)
            .await
            .is_err()
    );
    assert!(!sub.is_initialized());
    assert_eq!(service.pending_events()?, 0);
    service.ensure_usable()?;
    let (ready, full_id) = service.initialize_host_visible(&full, &expected, 0).await?;
    drop(ready);
    let (ready, sub_id) = service.initialize_host_visible(&sub, &payload, 0).await?;
    drop(ready);
    assert!(full_id.same_allocation(sub_id) && full_id.overlaps(sub_id));
    assert_eq!((sub_id.offset(), sub_id.bytes()), (OFFSET, VIEW));
    assert_eq!(service.resident_bytes()?, initial + BYTES);
    assert_eq!(
        service.uploaded_value_bytes(),
        uploaded + (BYTES + VIEW) as u64
    );
    assert!(full.is_initialized() && sub.is_initialized());
    assert!(
        service
            .initialize_host_visible(&sub, &payload, 0)
            .await
            .is_err()
    );
    expected[OFFSET..OFFSET + VIEW].copy_from_slice(&payload);

    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: (BYTES + 4) as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut init = service.begin_initialization(0)?;
    let status = init.upload_zeroed(4, BufferRole::Scratch)?;
    let output = init.allocate_empty(BYTES, BufferRole::Output)?;
    drop(init.complete_sync().await?);
    slot.finish()?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: BYTES as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let flight = service.submit(
        0,
        &[
            (full.clone(), DeviceAccess::Read),
            (output.clone(), DeviceAccess::Write),
            (status.clone(), DeviceAccess::Write),
        ],
    )?;
    // Original GPU operator consumes the complete backing, including the
    // changed subrange and untouched neighbours. No host read of GPU writes.
    unsafe {
        crate::program_kernels::copy_strided_words::launch_unchecked(
            service.client(),
            CubeCount::Static(32, 1, 1),
            CubeDim::new_1d(32),
            BufferArg::from_raw_parts(full.handle(), BYTES / 4),
            BufferArg::from_raw_parts(output.handle(), BYTES / 4),
            BufferArg::from_raw_parts(status.handle(), 1),
            0,
            0,
            (BYTES / 4) as u32,
            (BYTES / 4) as u32,
            1,
            BYTES / 4,
            false,
        );
    }
    let (ready, _) = flight
        .complete_with_readback(
            &[output.clone(), status.clone()],
            &[output.readback(0, BYTES)?, status.readback(0, 4)?],
            |data| {
                if data.len() != 2 || data[0].as_ref() != expected || data[1].as_ref() != [0u8; 4] {
                    return Err(Error::State(
                        "mapped backing GPU consumption differs".into(),
                    ));
                }
                Ok(())
            },
        )
        .await?;
    drop(ready);
    slot.finish()?;
    drop((status, output, sub, full, backing));
    assert_eq!(service.resident_bytes()?, initial);
    assert_eq!(arbiter.snapshot(host_key)?.reserved, host_before);
    assert_eq!(service.pending_events()?, 0);
    service.ensure_usable()?;
    eprintln!(
        "HOST_VISIBLE_BACKING_PASS exact_range=true original_charge=true gpu_consumption=true known_retirement=true"
    );
    Ok(())
}
