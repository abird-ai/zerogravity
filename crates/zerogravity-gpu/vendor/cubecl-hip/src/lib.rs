#[allow(unused_imports)]
#[macro_use]
extern crate derive_new;
extern crate alloc;

pub mod compiler;
pub mod compute;
pub mod device;
mod prepared_native;
pub mod runtime;
pub use compiler::{HipCompilationOptions, HipCompiler, HipRepresentation};
pub use compute::{HipSourcePageActor, OwnedHipTransferRetirement};
#[cfg(feature = "source-page-event-probe")]
pub use compute::{
    SourcePageEventProbeMode, SourcePageEventProbeSnapshot, SourcePageEventProbeState,
};
pub use device::*;
pub use prepared_native::{
    HipNativeAdmission, HipNativeEnvironment, HipNativeInventory, HipScopedObservation,
    HipScopedSetObservation, NativeRetention, PreparedHipNative, RegisteredHipNative,
    SourcePageAuthority, scoped_metadata_bytes,
};
pub use runtime::{HipRuntime, preflight_physical};

/// Whether both dynamic HIP runtime and RTC libraries can be loaded. Does not
/// enumerate devices, create a client, compile or launch a kernel. Set SDK
/// environment paths before the first lookup; the loader caches its result.
pub fn is_runtime_available() -> bool {
    cubecl_hip_sys::is_available()
}

// HIP has no runtime on Apple platforms. Keep the compiler-only unit tests above
// available there, but don't generate hundreds of tests that unconditionally
// open device 0.
#[cfg(all(test, any(target_os = "linux", target_os = "windows")))]
mod tests {
    use half::{bf16, f16};
    pub type TestRuntime = crate::HipRuntime;

    cubecl_std::testgen!();
    cubecl_core::testgen_all!(f32: [f16, bf16, f32], i32: [i16, i32], u32: [u16, u32]);
    cubecl_core::testgen_complex_validation!();
    cubecl_core::testgen_launch_dynamic_count!();
    cubecl_core::testgen_profiling!();
    cubecl_core::testgen_profiling_nested!();
}
