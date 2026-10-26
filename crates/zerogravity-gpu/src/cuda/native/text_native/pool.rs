//! Full image+complete ABI pooling; no entry-only or shape-only reuse.
use super::super::{
    cubin_manifests::ProjectionImage,
    projection_cubin::{native_error, not_cancelled, resident::PreparedImages},
};
use super::{
    packets::{Catalogue, Key, Role, Site},
    programs::insert,
};
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentService};
use cubecl::cuda::{CubinIo, CubinKernelManifest, CubinWorkspaceClear, OwnedCudaKernel};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};
use zerogravity_core::{Error, Result, device_events::DeviceAccess};

struct Record {
    image: Option<ProjectionImage>,
    manifest: CubinKernelManifest,
    kernel: Arc<OwnedCudaKernel>,
}
pub(super) struct Pool {
    catalogue: Catalogue,
    records: Vec<Record>,
    sites: BTreeMap<Key, Vec<usize>>,
}
impl Pool {
    pub(super) fn prepare(
        catalogue: Catalogue,
        images: Arc<PreparedImages>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        init.retain_completion_owner(images.clone())?;
        let mut records: Vec<Record> = Vec::new();
        let mut sites = BTreeMap::new();
        for site in catalogue.sites() {
            let mut ids = Vec::with_capacity(site.packets.len());
            for packet in &site.packets {
                not_cancelled(cancelled)?;
                let existing = records.iter().position(|record| {
                    record.image == packet.image && record.manifest == packet.manifest
                });
                let index = if let Some(index) = existing {
                    index
                } else {
                    let image = match packet.image {
                        Some(kind) => images.image(kind)?,
                        None => images.activation()?,
                    };
                    // SAFETY: generated literal source ABI; the same original
                    // set independently seals the complete image+manifest digest.
                    let kernel = unsafe { images.set().prepare(image, packet.manifest.clone()) }
                        .map_err(native_error)?;
                    init.retain_completion_owner(kernel.clone())?;
                    records.push(Record {
                        image: packet.image,
                        manifest: packet.manifest.clone(),
                        kernel,
                    });
                    records.len() - 1
                };
                ids.push(index);
            }
            if site.copy_output.is_some_and(|i| i >= ids.len())
                || sites.insert(site.key, ids).is_some()
            {
                return Err(Error::State(
                    "text pooled exact source-site packet/copy index".into(),
                ));
            }
        }
        Ok(Self {
            catalogue,
            records,
            sites,
        })
    }
    pub(super) fn site(&self, key: Key) -> Result<&Site> {
        self.catalogue.site(key)
    }
    pub(super) fn kernels(&self) -> Vec<Arc<OwnedCudaKernel>> {
        self.records.iter().map(|r| r.kernel.clone()).collect()
    }
    pub(super) fn bind_accesses(
        &self,
        site: &Site,
        roles: &BTreeMap<Role, ResidentLease>,
        service: &Arc<ResidentService>,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let mut all = BTreeMap::new();
        for packet in &site.packets {
            if packet.roles.len() != packet.manifest.bindings.len() {
                return Err(Error::State("text native exact role count".into()));
            }
            for (role, spec) in packet.roles.iter().zip(&packet.manifest.bindings) {
                let lease = roles
                    .get(role)
                    .ok_or_else(|| Error::State(format!("text source role missing: {role:?}")))?;
                if !Arc::ptr_eq(lease.service(), service)
                    || lease.service_generation() != service.generation()
                    || lease.source_identity() != service.source_identity()
                    || lease.bytes() as u64 != spec.bytes
                {
                    return Err(Error::State(format!(
                        "text source role/whole original lease extent differs: {role:?}"
                    )));
                }
                insert(
                    &mut all,
                    lease.clone(),
                    if spec.io == CubinIo::Read {
                        DeviceAccess::Read
                    } else {
                        DeviceAccess::Write
                    },
                );
            }
        }
        if site.copy_output.is_some() {
            for (role, access) in [
                (Role::PrivateO, DeviceAccess::Read),
                (Role::O, DeviceAccess::Write),
            ] {
                let lease = roles
                    .get(&role)
                    .ok_or_else(|| Error::State("text actual cached FA copy role absent".into()))?;
                if lease.bytes() != 4096 || !Arc::ptr_eq(lease.service(), service) {
                    return Err(Error::State(
                        "text exact cached FA whole copy extent/service".into(),
                    ));
                }
                insert(&mut all, lease.clone(), access);
            }
        }
        Ok(all.into_values().collect())
    }
    /// Seal exact roles/handles once for this immutable current-call binding.
    /// No permit or Ready authority is carried by this cold command metadata.
    pub(super) fn bind_site(
        &self,
        key: Key,
        roles: &BTreeMap<Role, ResidentLease>,
        service: &Arc<ResidentService>,
    ) -> Result<BoundSite> {
        let site = self.site(key)?;
        let accesses = self.bind_accesses(site, roles, service)?;
        let records = self
            .sites
            .get(&key)
            .ok_or_else(|| Error::State("text native pooled site absent".into()))?;
        let mut packets = Vec::with_capacity(site.packets.len());
        for (packet, index) in site.packets.iter().zip(records) {
            let leases = packet
                .roles
                .iter()
                .map(|r| {
                    roles
                        .get(r)
                        .cloned()
                        .ok_or_else(|| Error::State("text original role lease absent".into()))
                })
                .collect::<Result<Vec<_>>>()?;
            if packet.clears.len() > 1 {
                return Err(Error::Unsupported(
                    "text multiple workspace prefix clears".into(),
                ));
            }
            let clear = packet
                .clears
                .first()
                .map(|(role, offset, bytes)| {
                    let binding = packet
                        .roles
                        .iter()
                        .position(|r| r == role)
                        .ok_or_else(|| Error::State("text fixed clear role absent".into()))?;
                    match (*offset, *bytes) {
                        (0, 32) => CubinWorkspaceClear::zero32(binding).map_err(native_error),
                        (0, 64) => CubinWorkspaceClear::zero64(binding).map_err(native_error),
                        _ => Err(Error::Unsupported(
                            "text source clear must use original bounded prefix API".into(),
                        )),
                    }
                })
                .transpose()?;
            packets.push(BoundPacket {
                kernel: self.records[*index].kernel.clone(),
                leases,
                clear,
            });
        }
        let copy = site
            .copy_output
            .map(|index| {
                let from = roles
                    .get(&Role::PrivateO)
                    .ok_or_else(|| Error::State("text private FA O absent".into()))?;
                let to = roles
                    .get(&Role::O)
                    .ok_or_else(|| Error::State("text public FA O absent".into()))?;
                Ok::<_, Error>((
                    self.records[records[index]].kernel.clone(),
                    from.clone(),
                    to.clone(),
                ))
            })
            .transpose()?;
        Ok(BoundSite {
            service: service.clone(),
            generation: service.generation(),
            packets,
            copy,
            accesses,
        })
    }
    pub(super) unsafe fn retire_after_physical_completion(
        &self,
    ) -> std::result::Result<(), cubecl::server::ServerError> {
        for record in &self.records {
            // SAFETY: Storage caller owns original completed retirement flight.
            unsafe { record.kernel.retire_after_physical_completion() }?;
        }
        Ok(())
    }
}

struct BoundPacket {
    kernel: Arc<OwnedCudaKernel>,
    leases: Vec<ResidentLease>,
    clear: Option<CubinWorkspaceClear>,
}
/// Private, immutable product of the exact Pool + role/lease validator.
/// Original current cursor/attempt/flight checks remain in TextPrograms.
pub(super) struct BoundSite {
    service: Arc<ResidentService>,
    generation: u64,
    packets: Vec<BoundPacket>,
    copy: Option<(Arc<OwnedCudaKernel>, ResidentLease, ResidentLease)>,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
}
impl BoundSite {
    pub(super) fn accesses(&self) -> &[(ResidentLease, DeviceAccess)] {
        &self.accesses
    }
    pub(super) fn enqueue(
        &self,
        service: &Arc<ResidentService>,
        cancelled: Option<&AtomicBool>,
        counts: &[AtomicUsize; 3],
    ) -> Result<()> {
        if !Arc::ptr_eq(service, &self.service) || service.generation() != self.generation {
            return Err(Error::State(
                "text sealed packet service/generation differs".into(),
            ));
        }
        service.ensure_usable()?;
        for packet in &self.packets {
            not_cancelled(cancelled)?;
            // SAFETY: immutable exact image/ABI/roles were sealed by Pool;
            // the original registered flight retains every actual owner.
            let bindings = packet
                .leases
                .iter()
                .map(ResidentLease::cubin_binding)
                .collect::<Result<Vec<_>>>()?;
            unsafe {
                packet
                    .kernel
                    .enqueue_workspace_bindings(bindings, packet.clear)
            }
            .map_err(native_error)?;
            if packet.clear.is_some() {
                counts[1].fetch_add(1, Ordering::Release);
            }
            counts[0].fetch_add(1, Ordering::Release);
        }
        if let Some((kernel, from, to)) = &self.copy {
            not_cancelled(cancelled)?;
            // SAFETY: same selected copy ABI and original armed access cohort.
            unsafe {
                kernel
                    .copy_text_attention_output_bindings(from.cubin_binding()?, to.cubin_binding()?)
            }
            .map_err(native_error)?;
            counts[2].fetch_add(1, Ordering::Release);
        }
        Ok(())
    }
}
