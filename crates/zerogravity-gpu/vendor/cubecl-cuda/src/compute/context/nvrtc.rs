//! A raw NVRTC program borrows its source through checked destruction.
use super::compiler_context::CompilerContextState;
use super::compiler_limits::CudaCompilerContextPolicy;
use super::program_lifetime::{ProgramGuard, ProgramResource};
use super::*;
use crate::install::{cccl_include_path, include_path};

struct NvrtcProgram {
    handle: cudarc::nvrtc::sys::nvrtcProgram,
    _source: CString,
}
impl ProgramResource for NvrtcProgram {
    type Error = cudarc::nvrtc::result::NvrtcError;
    fn destroy(&mut self) -> Result<(), Self::Error> {
        // SAFETY: this sole owner attempts destruction once. On an error the
        // complete object is retained without retry, including borrowed source.
        unsafe { cudarc::nvrtc::result::destroy_program(self.handle) }
    }
}

fn error(reason: impl Into<String>) -> CompilationError {
    CompilationError::Generic {
        reason: reason.into(),
        backtrace: BackTrace::capture(),
    }
}

impl CudaContext {
    /// Sticky uncertainty uses the original poisoned-device propagation. A
    /// later empty sync must not settle the submitting initialization owner.
    pub(crate) fn check_compiler(&self) -> Result<(), cubecl_server::driver::DevicePoison> {
        if self.compiler_unknown
            || self
                .compiler_context
                .as_ref()
                .is_some_and(|c| c.check_environment().is_err())
        {
            return Err(cubecl_server::driver::DevicePoison::new(
                "CUDA compiler/module lifetime is uncertain; process restart required",
            ));
        }
        Ok(())
    }

    pub(super) fn compiler_failure(&mut self, reason: String) -> CompilationError {
        self.compiler_unknown = true;
        cubecl_server::driver::DevicePoison::new(reason).into()
    }

    pub(super) fn compile_to_ptx(&mut self, source: &str) -> Result<Vec<c_char>, CompilationError> {
        self.check_compiler()?;
        let arch = if self.arch.version >= 90 {
            format!("--gpu-architecture=sm_{}a", self.arch)
        } else {
            format!("--gpu-architecture=sm_{}", self.arch)
        };
        // Adopted policy consumes its captured actual options. No CUDA_PATH
        // reread can change inputs between the actor's check and NVRTC call.
        let (include_path, cccl_include_path) = if let Some(context) = &self.compiler_context {
            context.include_paths()
        } else {
            let cccl = cccl_include_path();
            (include_path(), cccl.exists().then_some(cccl))
        };
        let include_option = format!("--include-path={}", include_path.to_str().unwrap());
        let cccl_include_option = cccl_include_path
            .as_ref()
            .map(|path| format!("--include-path={}", path.to_str().unwrap()));
        let mut options = vec![arch.as_str(), include_option.as_str(), "-lineinfo"];
        if let Some(option) = &cccl_include_option {
            options.push(option);
        }

        let c_source = CString::new(source).map_err(|err| error(format!("CUDA source: {err}")))?;
        let handle = cudarc::nvrtc::result::create_program(c_source.as_c_str(), None)
            .map_err(|err| error(format!("nvrtcCreateProgram: {err}")))?;
        let program = ProgramGuard::new(
            NvrtcProgram {
                handle,
                _source: c_source,
            },
            &mut self.compiler_unknown,
        );
        // Keep body errors until checked destruction has also settled. All
        // fallible compile/log/PTX paths and unwinding use the same owner.
        let compiled = compile(
            program.resource().handle,
            &options,
            source,
            self.compiler_context.as_mut(),
        );
        match program.finish() {
            Ok(()) => compiled,
            Err(cleanup) => {
                let earlier = match compiled {
                    Ok(_) => "PTX extraction succeeded".to_owned(),
                    Err(err) => format!("earlier compiler failure: {err}"),
                };
                Err(self.compiler_failure(format!(
                    "nvrtcDestroyProgram uncertain: {cleanup:?}; {earlier}; original program/source retained until process exit"
                )))
            }
        }
    }
}

fn compile(
    program: cudarc::nvrtc::sys::nvrtcProgram,
    options: &[&str],
    source: &str,
    context: Option<&mut CompilerContextState>,
) -> Result<Vec<c_char>, CompilationError> {
    // SAFETY: ProgramGuard retains the sole live program and its CString
    // through this call; the guard performs destruction after the result.
    unsafe {
        if let Err(compile_error) = cudarc::nvrtc::result::compile_program(program, options) {
            let log = program_log(program, context.is_some()).map_err(|err| {
                error(format!(
                    "nvrtcCompileProgram: {compile_error}; nvrtcGetProgramLog: {err}"
                ))
            })?;
            // NVRTC logs are diagnostic bytes; invalid UTF-8 or a missing NUL
            // must not panic or bypass checked program retirement.
            let bytes = log.iter().map(|&byte| byte as u8).collect::<Vec<_>>();
            let log = String::from_utf8_lossy(bytes.strip_suffix(&[0]).unwrap_or(&bytes));
            let mut message = format!("[Compilation Error] nvrtcCompileProgram: {compile_error}");
            for line in log.lines().filter(|line| !line.is_empty()) {
                message.push_str("\n    ");
                message.push_str(line);
            }
            return Err(error(format!("{message}\n[Source]  \n{source}")));
        }
        if let Some(context) = context {
            let mut size = 0usize;
            cudarc::nvrtc::sys::nvrtcGetPTXSize(program, &mut size)
                .result()
                .map_err(|err| error(format!("nvrtcGetPTXSize: {err}")))?;
            context.ptx(size).map_err(error)?;
            let mut ptx = vec![0; size];
            cudarc::nvrtc::sys::nvrtcGetPTX(program, ptx.as_mut_ptr())
                .result()
                .map_err(|err| error(format!("nvrtcGetPTX: {err}")))?;
            Ok(ptx)
        } else {
            cudarc::nvrtc::result::get_ptx(program)
                .map_err(|err| error(format!("nvrtcGetPTX: {err}")))
        }
    }
}

unsafe fn program_log(
    program: cudarc::nvrtc::sys::nvrtcProgram,
    bounded: bool,
) -> Result<Vec<c_char>, CompilationError> {
    if !bounded {
        return unsafe { cudarc::nvrtc::result::get_program_log(program) }
            .map_err(|e| error(format!("nvrtcGetProgramLog: {e}")));
    }
    let mut size = 0usize;
    unsafe { cudarc::nvrtc::sys::nvrtcGetProgramLogSize(program, &mut size) }
        .result()
        .map_err(|e| error(format!("nvrtcGetProgramLogSize: {e}")))?;
    if size == 0 || size > CudaCompilerContextPolicy::MAX_LOG_BYTES {
        return Err(error("admitted CUDA compiler log extent exceeded"));
    }
    let mut log = vec![0; size];
    unsafe { cudarc::nvrtc::sys::nvrtcGetProgramLog(program, log.as_mut_ptr()) }
        .result()
        .map_err(|e| error(format!("nvrtcGetProgramLog: {e}")))?;
    Ok(log)
}
