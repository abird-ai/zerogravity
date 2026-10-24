//! Driver device metadata only. Never retains/sets a primary context, creates
//! a Client/stream, compiles code, allocates memory or launches a command.
#[derive(Clone, Debug)]
pub struct CudaDeviceMetadata {
    pub ordinal: usize,
    pub pci_bus_id: String,
    pub total_bytes: u64,
    pub compute_capability: (i32, i32),
    pub uuid: [u8; 16],
}
pub fn observe_cuda_devices() -> Result<Vec<CudaDeviceMetadata>, String> {
    use cudarc::driver::{result, sys};
    // SAFETY: existing loader presence check; no context is created.
    if !unsafe { sys::is_culib_present() } {
        return Ok(Vec::new());
    }
    result::init().map_err(|e| format!("CUDA inventory init: {e}"))?;
    let count = result::device::get_count().map_err(|e| e.to_string())?;
    if !(0..=64).contains(&count) {
        return Err("CUDA inventory device count exceeds64".into());
    }
    let mut observations = Vec::new();
    for ordinal in 0..count {
        let device = result::device::get(ordinal).map_err(|e| e.to_string())?;
        let mut pci = [0i8; 64];
        let uuid = result::device::get_uuid(device)
            .map_err(|e| e.to_string())?
            .bytes
            .map(|b| b as u8);
        // SAFETY: driver-issued device and bounded writable output; these
        // cuDevice calls query metadata and have no context-retain side effect.
        let (total, major, minor) = unsafe {
            sys::cuDeviceGetPCIBusId(pci.as_mut_ptr(), pci.len() as i32, device)
                .result()
                .map_err(|e| e.to_string())?;
            (
                result::device::total_mem(device).map_err(|e| e.to_string())?,
                result::device::get_attribute(
                    device,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                )
                .map_err(|e| e.to_string())?,
                result::device::get_attribute(
                    device,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                )
                .map_err(|e| e.to_string())?,
            )
        };
        let bytes = pci.iter().map(|&b| b as u8).collect::<Vec<_>>();
        let pci_bus_id = std::ffi::CStr::from_bytes_until_nul(&bytes)
            .map_err(|_| "CUDA inventory PCI terminator")?
            .to_str()
            .map_err(|_| "CUDA inventory PCI UTF8")?
            .to_string();
        observations.push(CudaDeviceMetadata {
            ordinal: ordinal as usize,
            pci_bus_id,
            total_bytes: total as u64,
            compute_capability: (major, minor),
            uuid,
        });
    }
    Ok(observations)
}
