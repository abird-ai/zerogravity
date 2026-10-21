//! Selecting between the HIP C++ backend and the LLVM backend.

#[cfg(not(any(feature = "cpp", feature = "llvm")))]
compile_error!("enable the cpp or llvm compiler feature");

use cubecl_core::ir::amd::GfxArch;
use cubecl_core::prelude::KernelDefinition;
use cubecl_cpp::shared::CompilationOptions;
use cubecl_cpp::{ComputeKernel, shared::CppCompiler, target::Hip};
use cubecl_server::compiler::{CompilationError, Compiler};
use cubecl_server::kernel::BufferIOAttr;

/// Which backend turns a `KernelDefinition` into something the HIP driver can load.
///
/// LLVM is optional. With it enabled, the `cpp` feature selects which one a default [`HipCompiler`]
/// runs. CPP remains the default when both compiler features are selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HipBackend {
    /// Transpile to HIP C++ and hand the source to HIP RTC.
    Cpp,
    /// Lower through pliron and LLVM to a linked AMD code object.
    #[cfg(feature = "llvm")]
    Llvm,
}

impl Default for HipBackend {
    fn default() -> Self {
        #[cfg(feature = "cpp")]
        {
            HipBackend::Cpp
        }
        #[cfg(all(not(feature = "cpp"), feature = "llvm"))]
        {
            HipBackend::Llvm
        }
        #[cfg(not(any(feature = "cpp", feature = "llvm")))]
        unreachable!("a compiler feature is required")
    }
}

#[derive(Clone, Debug)]
pub enum HipCompiler {
    Cpp(CppCompiler<Hip>),
    #[cfg(feature = "llvm")]
    Llvm(cubecl_llvm::PlironCompiler),
}

impl HipCompiler {
    pub fn new(backend: HipBackend) -> Self {
        match backend {
            HipBackend::Cpp => HipCompiler::Cpp(CppCompiler::default()),
            #[cfg(feature = "llvm")]
            HipBackend::Llvm => HipCompiler::Llvm(cubecl_llvm::PlironCompiler {
                target: cubecl_llvm::LlvmTarget::AmdGpu,
            }),
        }
    }
}

impl Default for HipCompiler {
    fn default() -> Self {
        Self::new(HipBackend::default())
    }
}

/// Options for whichever backend is selected. One struct rather than an enum:
/// `arch` is a device property, filled in regardless of which backend reads it.
#[derive(Debug, Default, Clone)]
pub struct HipCompilationOptions {
    pub cpp: CompilationOptions,
    /// The device, as the runtime parsed it out of `gcnArchName`.
    pub arch: Option<GfxArch>,
}

pub enum HipRepresentation {
    Cpp(ComputeKernel),
    #[cfg(feature = "llvm")]
    Llvm(cubecl_llvm::AmdGpuModule),
}

// `ComputeKernel` does not implement `Debug` (only `Display`, for its rendered
// source), so this is written by hand rather than derived.
impl core::fmt::Debug for HipRepresentation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            HipRepresentation::Cpp(_) => f.write_str("HipRepresentation::Cpp"),
            #[cfg(feature = "llvm")]
            HipRepresentation::Llvm(module) => f
                .debug_tuple("HipRepresentation::Llvm")
                .field(module)
                .finish(),
        }
    }
}

impl HipRepresentation {
    /// Shared memory the launch must reserve. Both backends give their kernels one block of
    /// dynamic shared memory, so this is what the launch passes as `sharedMemBytes`.
    pub fn shared_memory_size(&self) -> usize {
        match self {
            HipRepresentation::Cpp(kernel) => kernel.shared_memory_size,
            #[cfg(feature = "llvm")]
            HipRepresentation::Llvm(module) => module.shared_memory_size,
        }
    }
}

impl core::fmt::Display for HipRepresentation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            HipRepresentation::Cpp(kernel) => write!(f, "{kernel}"),
            #[cfg(feature = "llvm")]
            HipRepresentation::Llvm(module) => write!(f, "{}", module.ir),
        }
    }
}

impl Compiler for HipCompiler {
    type Representation = HipRepresentation;
    type CompilationOptions = HipCompilationOptions;

    /// Forwarded from the wrapped backend. The trait's default answers `None`,
    /// which reads as every buffer both read and written -- and that would make
    /// a pure output look like an input, so a relaunch repairing a tainted
    /// buffer would be skipped for reading what it only writes.
    fn buffer_io(repr: &Self::Representation) -> Option<Vec<BufferIOAttr>> {
        match repr {
            HipRepresentation::Cpp(kernel) => <CppCompiler<Hip> as Compiler>::buffer_io(kernel),
            #[cfg(feature = "llvm")]
            HipRepresentation::Llvm(module) => Some(module.io.clone()),
        }
    }

    fn compile(
        &mut self,
        kernel: KernelDefinition,
        options: &Self::CompilationOptions,
    ) -> Result<Self::Representation, CompilationError> {
        match self {
            HipCompiler::Cpp(compiler) => Ok(HipRepresentation::Cpp(
                compiler.compile(kernel, &options.cpp)?,
            )),
            #[cfg(feature = "llvm")]
            HipCompiler::Llvm(compiler) => {
                let pliron_options = cubecl_llvm::PlironOptions {
                    // The HIP entry ABI has no parameter block to put them in.
                    grid_constants: false,
                    arch: options.arch.clone(),
                    ..Default::default()
                };
                match compiler.compile(kernel, &pliron_options)? {
                    cubecl_llvm::PlironArtifact::AmdGpuCode(module) => {
                        Ok(HipRepresentation::Llvm(module))
                    }
                    _ => unreachable!("the HIP runtime always configures LlvmTarget::AmdGpu"),
                }
            }
        }
    }

    fn extension(&self) -> &'static str {
        match self {
            HipCompiler::Cpp(compiler) => compiler.extension(),
            #[cfg(feature = "llvm")]
            HipCompiler::Llvm(_) => "ll",
        }
    }

    fn lang_tag(&self) -> &'static str {
        match self {
            HipCompiler::Cpp(compiler) => compiler.lang_tag(),
            #[cfg(feature = "llvm")]
            HipCompiler::Llvm(compiler) => compiler.lang_tag(),
        }
    }
}

#[cfg(test)]
mod compiler_selection_tests {
    #[test]
    fn default_compiler_respects_compiled_feature() {
        #[cfg(feature = "cpp")]
        assert!(matches!(
            super::HipCompiler::default(),
            super::HipCompiler::Cpp(_)
        ));
        #[cfg(all(not(feature = "cpp"), feature = "llvm"))]
        assert!(matches!(
            super::HipCompiler::default(),
            super::HipCompiler::Llvm(_)
        ));
    }
}
