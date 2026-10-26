//! Explicit admitted workspace views; original full-allocation APIs stay exact.
use super::*;
use std::ops::Range;

#[derive(Clone, Copy, Debug)]
pub struct CubinWorkspaceView {
    pub backing_bytes: u64,
    pub bytes: u64,
    pub alignment: u64,
}
struct ViewSeal {
    backing: Handle,
    layout: CubinWorkspaceView,
    _run: Arc<dyn Any + Send + Sync>,
}
/// Mint a view at enqueue time, never in a persistent prepared/cold owner:
/// the seal retains the actual exclusive run until actor acknowledgement.
pub struct OwnedCudaBinding {
    handle: Handle,
    view: Option<ViewSeal>,
}
fn range(h: &Handle) -> Result<Range<u64>, ServerError> {
    let start = h.offset_start.unwrap_or(0);
    let end = h
        .size()
        .checked_sub(h.offset_end.unwrap_or(0))
        .filter(|end| start < *end)
        .ok_or_else(|| failure("workspace actual managed range"))?;
    Ok(start..end)
}
fn same_memory(a: &Handle, b: &Handle) -> bool {
    a.service == b.service && a.clone().binding().memory.id() == b.clone().binding().memory.id()
}
fn require_view(
    backing: &Handle,
    view: &Handle,
    service: ServiceId,
    layout: CubinWorkspaceView,
) -> Result<(), ServerError> {
    let parent = range(backing)?;
    let child = range(view)?;
    if backing.service != service
        || !same_memory(backing, view)
        || backing.size() != view.size()
        || layout.backing_bytes == 0
        || parent.end - parent.start != layout.backing_bytes
        || layout.bytes == 0
        || child.end - child.start != layout.bytes
        || !layout.alignment.is_power_of_two()
        || !layout.backing_bytes.is_multiple_of(layout.alignment)
        || !parent.start.is_multiple_of(layout.alignment)
        || !child.start.is_multiple_of(layout.alignment)
        || child.start < parent.start
        || child.end > parent.end
    {
        return Err(failure(
            "workspace actual backing/view service/extent/alignment differs",
        ));
    }
    Ok(())
}
impl OwnedCudaBinding {
    pub fn whole(handle: Handle) -> Self {
        Self { handle, view: None }
    }
    pub(crate) fn handle(&self) -> Handle {
        self.handle.clone()
    }
    pub(crate) fn require(
        &self,
        service: ServiceId,
        bytes: u64,
        alignment: u64,
    ) -> Result<(), ServerError> {
        if let Some(view) = &self.view {
            if view.layout.bytes != bytes || view.layout.alignment != alignment {
                return Err(failure("workspace binding differs from exact compiled ABI"));
            }
            require_view(&view.backing, &self.handle, service, view.layout)
        } else {
            canonical_handle(&self.handle, service, bytes, alignment)
        }
    }
    /// Copies historically require an exact full owner without allocator tail.
    /// An explicit view instead revalidates its original sealed backing.
    pub(crate) fn require_copy(&self, service: ServiceId, bytes: u64) -> Result<(), ServerError> {
        if let Some(view) = &self.view {
            if view.layout.bytes != bytes {
                return Err(failure("workspace exact copy extent differs"));
            }
            require_view(&view.backing, &self.handle, service, view.layout)
        } else if self.handle.service == service
            && self.handle.size() == bytes
            && self.handle.offset_start.unwrap_or(0) == 0
            && self.handle.offset_end.unwrap_or(0) == 0
            && bytes != 0
        {
            Ok(())
        } else {
            Err(failure("copy exact full original owner differs"))
        }
    }
}
impl OwnedCudaQueue {
    /// # Safety
    /// The original engine admitted this real backing and exact view under the
    /// same source/service/generation. `owner` is its currently active exclusive
    /// workspace run, retained by the original armed flight through completion
    /// or quarantine. No cold owner may retain this temporary active-run seal.
    pub unsafe fn admit_workspace_binding(
        &self,
        backing: Handle,
        view: Handle,
        layout: CubinWorkspaceView,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<OwnedCudaBinding, ServerError> {
        require_view(&backing, &view, self.service_id(), layout)?;
        Ok(OwnedCudaBinding {
            handle: view,
            view: Some(ViewSeal {
                backing,
                layout,
                _run: owner,
            }),
        })
    }
}
fn require_aliases(
    bindings: &[OwnedCudaBinding],
    writes: impl Fn(usize) -> bool,
) -> Result<(), ServerError> {
    for (index, a) in bindings.iter().enumerate() {
        let ar = range(&a.handle)?;
        for (j, b) in bindings[..index].iter().enumerate() {
            if !same_memory(&a.handle, &b.handle) {
                continue;
            }
            let br = range(&b.handle)?;
            if ar.start < br.end && br.start < ar.end && (writes(index) || writes(j)) {
                return Err(failure(
                    "workspace native read/write physical ranges overlap",
                ));
            }
        }
    }
    Ok(())
}
impl OwnedCudaKernel {
    pub(crate) fn require_workspace_bindings(
        &self,
        bindings: &[OwnedCudaBinding],
        bytes: Option<&[u64]>,
    ) -> Result<(), ServerError> {
        self.require_active()?;
        if bindings.len() != self.manifest.bindings.len()
            || bytes.is_some_and(|b| b.len() != bindings.len())
        {
            return Err(failure("workspace compiled binding count"));
        }
        for (i, (binding, spec)) in bindings.iter().zip(&self.manifest.bindings).enumerate() {
            let bytes = bytes.map_or(spec.bytes, |b| b[i]);
            binding.require(
                self.set.queue.service_id(),
                bytes,
                self.allocation_alignment,
            )?;
        }
        require_aliases(bindings, |i| self.manifest.bindings[i].io.writes())
    }
    /// # Safety
    /// Same source/cursor/access contract as original enqueue. Every view seal
    /// carries the genuine active run already owned by this original flight.
    pub unsafe fn enqueue_workspace_bindings(
        &self,
        bindings: Vec<OwnedCudaBinding>,
        clear: Option<CubinWorkspaceClear>,
    ) -> Result<(), ServerError> {
        self.require_workspace_bindings(&bindings, None)?;
        if let Some(clear) = clear {
            clear.validate(&self.manifest)?;
        }
        let (set, key, stream) = (self.set.key, self.key, self.set.queue.stream);
        self.set
            .queue
            .actor
            .submit_blocking(move |server| {
                let handles = bindings.iter().map(OwnedCudaBinding::handle).collect();
                // Binding seals remain alive until this original command returns.
                server.enqueue_owned_cubin(stream, set, key, handles, clear, None)
            })
            .map_err(|e| failure(format!("workspace original cubin actor: {e:?}")))?
    }
    /// # Safety
    /// Original authenticated source window and active exact X/Y geometry share
    /// the SAME armed flight. Checkpoint/overwrite retirement remains original.
    pub unsafe fn enqueue_streamed_vocabulary_head_bindings(
        &self,
        bindings: [OwnedCudaBinding; 3],
        layout: crate::SourceWeightLayout,
        input_rows: usize,
        ordinal: usize,
    ) -> Result<(), ServerError> {
        self.require_active()?;
        if super::super::head_policy::streamed::require(&self.image, &self.manifest)? != layout {
            return Err(failure(
                "workspace streamed head original source layout differs",
            ));
        }
        let selection = super::super::head_policy::streamed::Window {
            layout,
            input_rows,
            ordinal,
        };
        let bytes = [
            selection.full_bytes(0)?,
            selection.full_bytes(1)?,
            selection.full_bytes(2)?,
        ];
        self.require_workspace_bindings(&bindings, Some(&bytes))?;
        let (set, key, stream) = (self.set.key, self.key, self.set.queue.stream);
        self.set
            .queue
            .actor
            .submit_blocking(move |server| {
                let handles = bindings.iter().map(OwnedCudaBinding::handle).collect();
                server.enqueue_owned_cubin(
                    stream,
                    set,
                    key,
                    handles,
                    None,
                    Some(super::super::head_policy::streamed::Selection::Streamed(
                        selection,
                    )),
                )
            })
            .map_err(|e| failure(format!("workspace original streamed head actor: {e:?}")))?
    }
    /// # Safety
    /// Exact cached FA executed earlier in this same original armed flight;
    /// its private output and public destination remain retained to completion.
    pub unsafe fn copy_text_attention_output_bindings(
        &self,
        source: OwnedCudaBinding,
        destination: OwnedCudaBinding,
    ) -> Result<(), ServerError> {
        self.require_active()?;
        super::super::text_policy::require_n1_copy(&self.image, &self.manifest)?;
        for binding in [&source, &destination] {
            binding.require(self.set.queue.service_id(), 4096, self.allocation_alignment)?;
        }
        let bindings = [source, destination];
        require_aliases(&bindings, |i| i == 1)?;
        let (set, key, stream) = (self.set.key, self.key, self.set.queue.stream);
        self.set
            .queue
            .actor
            .submit_blocking(move |server| {
                let source = bindings[0].handle();
                let destination = bindings[1].handle();
                server.copy_owned_attention_output(stream, set, key, source, destination)
            })
            .map_err(|e| failure(format!("workspace original FA copy actor: {e:?}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_common::device::DeviceId;
    use cubecl_core::server::{MemoryLayoutDescriptor, MemoryLayoutPolicy};
    use cubecl_environment::stream::StreamId;
    use cubecl_server::allocator::ContiguousMemoryLayoutPolicy;
    fn handles() -> (ServiceId, Handle, Handle) {
        let service = ServiceId::of::<ContiguousMemoryLayoutPolicy>(DeviceId::new(0, 0));
        let (base, mut views) = ContiguousMemoryLayoutPolicy::new(512).apply(
            service,
            StreamId::current(),
            &[MemoryLayoutDescriptor::contiguous([4096].into(), 1)],
        );
        (service, base, views.remove(0).memory)
    }
    #[test]
    fn exact_admitted_view_does_not_change_original_whole_policy() {
        let (service, backing, mut view) = handles();
        view.offset_start = Some(512);
        view.offset_end = Some(2048);
        let layout = CubinWorkspaceView {
            backing_bytes: 4096,
            bytes: 1536,
            alignment: 512,
        };
        require_view(&backing, &view, service, layout).unwrap();
        assert!(canonical_handle(&view, service, 1536, 512).is_err());
        for fault in 0..6 {
            let mut bad = view.clone();
            let mut declaration = layout;
            match fault {
                0 => bad.offset_end = Some(2049),
                1 => bad.offset_start = Some(513),
                2 => bad.offset_end = Some(u64::MAX),
                3 => declaration.backing_bytes = 3584,
                4 => declaration.alignment = 3,
                _ => bad.service = ServiceId::of::<u32>(DeviceId::new(0, 0)),
            }
            assert!(require_view(&backing, &bad, service, declaration).is_err());
        }
        let (_, unrelated, _) = handles();
        assert!(require_view(&unrelated, &view, service, layout).is_err());
    }
}
