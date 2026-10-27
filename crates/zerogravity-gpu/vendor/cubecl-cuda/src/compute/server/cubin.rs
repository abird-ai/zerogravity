//! Native modules/functions and pointer fixups stay inside the original actor.
use super::*;
use crate::owned_command::cubin::{CubinImageKey, CubinSetPolicy, failure};
use crate::owned_command::{CubinKernelManifest, CubinWorkspaceClear, VerifiedCubin};
use cudarc::driver::sys::{CUfunction, CUfunction_attribute, CUmodule, CUresult};
use std::ffi::CString;
mod attention_copy;
mod cache;
pub(crate) use cache::CubinModuleSet;
use cache::{CubinFunction, CubinKernel, CubinModule};
#[cfg(test)]
mod tests;
fn status(code: CUresult, operation: &str) -> Result<(), ServerError> {
    if code == CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(failure(format!("original cubin {operation}: {code:?}")))
    }
}
fn launch_scope<'server, 'read, S: WriteScoped>(
    server: &'server mut S,
    stream: StreamId,
    key: u64,
    reads: impl Iterator<Item = &'read BufferBinding>,
    written: Vec<BufferBinding>,
) -> ExecuteScope<'server, S> {
    // Inspect inputs before provisional output taint, including ReadWrite.
    // Native key is immutable/unique for this module's full actor lifetime.
    ExecuteScope::launching(
        server,
        cubecl_server::id::KernelId::new::<CubinKernel>().info(key),
        stream,
        reads,
        written,
    )
}
fn binding_region(
    binding: &BufferBinding,
    relative: std::ops::Range<u64>,
) -> Result<BufferBinding, ServerError> {
    let start = binding.offset_start.unwrap_or(0);
    let end = binding
        .size
        .checked_sub(binding.offset_end.unwrap_or(0))
        .ok_or_else(|| failure("workspace claim original end range"))?;
    let begin = start
        .checked_add(relative.start)
        .ok_or_else(|| failure("workspace claim start overflow"))?;
    let stop = start
        .checked_add(relative.end)
        .filter(|stop| *stop <= end && begin < *stop)
        .ok_or_else(|| failure("workspace claim exceeds original binding"))?;
    let mut region = binding.clone();
    region.offset_start = Some(begin);
    region.offset_end = Some(binding.size - stop);
    Ok(region)
}
impl CudaServer {
    pub(crate) fn admit_owned_cubin_set(
        &mut self,
        key: u64,
        owner: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<(), ServerError> {
        if self.cubin_set.is_some() {
            return Err(failure("one active cubin module set per original actor"));
        }
        // Host ownership is installed before any module load/driver allocation.
        self.cubin_set = Some(CubinModuleSet::new(key, owner));
        Ok(())
    }
    pub(crate) fn admit_owned_connected_qwen08_cubin_set(
        &mut self,
        key: u64,
        owner: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<(), ServerError> {
        if self.cubin_set.is_some() {
            return Err(failure("one active cubin module set per original actor"));
        }
        // Immutable scope is installed with the SAME owner before native work.
        self.cubin_set = Some(CubinModuleSet::new_with_policy(
            key,
            owner,
            CubinSetPolicy::ConnectedQwen08,
        ));
        Ok(())
    }
    pub(crate) fn admit_owned_fixed_prompt_mlp_cubin_set(
        &mut self,
        key: u64,
        owner: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<(), ServerError> {
        if self.cubin_set.is_some() {
            return Err(failure("one active cubin module set per original actor"));
        }
        self.cubin_set = Some(CubinModuleSet::new_with_policy(
            key,
            owner,
            CubinSetPolicy::FixedPromptMlp,
        ));
        Ok(())
    }
    pub(crate) fn admit_owned_qwen2b_mlp_cubin_set(
        &mut self,
        key: u64,
        owner: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<(), ServerError> {
        if self.cubin_set.is_some() {
            return Err(failure("one active cubin module set per original actor"));
        }
        self.cubin_set = Some(CubinModuleSet::new_with_policy(
            key,
            owner,
            CubinSetPolicy::Qwen2bMlp,
        ));
        Ok(())
    }
    pub(crate) fn admit_owned_fixed_prompt_qkv_cubin_set(
        &mut self,
        key: u64,
        owner: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<(), ServerError> {
        if self.cubin_set.is_some() {
            return Err(failure("one active cubin module set per original actor"));
        }
        self.cubin_set = Some(CubinModuleSet::new_with_policy(
            key,
            owner,
            CubinSetPolicy::FixedPromptQkv,
        ));
        Ok(())
    }
    pub(crate) fn admit_owned_fixed_prompt_qkkv_cubin_set(
        &mut self,
        key: u64,
        owner: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<(), ServerError> {
        if self.cubin_set.is_some() {
            return Err(failure("one active cubin module set per original actor"));
        }
        self.cubin_set = Some(CubinModuleSet::new_with_policy(
            key,
            owner,
            CubinSetPolicy::FixedPromptQkKv,
        ));
        Ok(())
    }
    pub(crate) fn admit_owned_fixed_prompt_fa_cubin_set(
        &mut self,
        key: u64,
        owner: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<(), ServerError> {
        if self.cubin_set.is_some() {
            return Err(failure("one active cubin module set per original actor"));
        }
        self.cubin_set = Some(CubinModuleSet::new_with_policy(
            key,
            owner,
            CubinSetPolicy::FixedPromptFa,
        ));
        Ok(())
    }
    pub(crate) fn admit_owned_fixed_prompt_layer3_cubin_set(
        &mut self,
        key: u64,
        owner: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<(), ServerError> {
        if self.cubin_set.is_some() {
            return Err(failure("one active cubin module set per original actor"));
        }
        self.cubin_set = Some(CubinModuleSet::new_with_policy(
            key,
            owner,
            CubinSetPolicy::FixedPromptLayer3,
        ));
        Ok(())
    }
    pub(crate) fn admit_owned_fixed_prompt_full_cubin_set(
        &mut self,
        key: u64,
        owner: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<(), ServerError> {
        if self.cubin_set.is_some() {
            return Err(failure("one active cubin module set per original actor"));
        }
        self.cubin_set = Some(CubinModuleSet::new_with_policy(
            key,
            owner,
            CubinSetPolicy::FixedPromptFull,
        ));
        Ok(())
    }
    pub(crate) fn admit_owned_fixed_prompt_natural_cubin_set(
        &mut self,
        key: u64,
        owner: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<(), ServerError> {
        if self.cubin_set.is_some() {
            return Err(failure("one active cubin module set per original actor"));
        }
        self.cubin_set = Some(CubinModuleSet::new_with_policy(
            key,
            owner,
            CubinSetPolicy::FixedPromptNatural,
        ));
        Ok(())
    }
    pub(crate) fn admit_owned_text_fresh_prefill_cubin_set(
        &mut self,
        key: u64,
        owner: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<(), ServerError> {
        if self.cubin_set.is_some() {
            return Err(failure("one active cubin module set per original actor"));
        }
        let policy = CubinSetPolicy::Text08FreshPrefill;
        crate::owned_command::cubin::text_policy::require_complete(policy)?;
        self.cubin_set = Some(CubinModuleSet::new_with_policy(key, owner, policy));
        Ok(())
    }
    pub(crate) fn admit_owned_text_cubin_set(
        &mut self,
        key: u64,
        owner: Arc<dyn std::any::Any + Send + Sync>,
        two_b: bool,
        cohort4: bool,
    ) -> Result<(), ServerError> {
        if self.cubin_set.is_some() {
            return Err(failure("one active cubin module set per original actor"));
        }
        let policy = if cohort4 && two_b {
            CubinSetPolicy::Text2BCachedCohort4
        } else if cohort4 {
            CubinSetPolicy::Text08Cohort4
        } else if two_b {
            CubinSetPolicy::Text2B
        } else {
            CubinSetPolicy::Text08
        };
        crate::owned_command::cubin::text_policy::require_complete(policy)?;
        self.cubin_set = Some(CubinModuleSet::new_with_policy(key, owner, policy));
        Ok(())
    }
    pub(crate) fn admit_owned_first_attention_cubin_set(
        &mut self,
        key: u64,
        owner: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<(), ServerError> {
        if self.cubin_set.is_some() {
            return Err(failure("one active cubin module set per original actor"));
        }
        self.cubin_set = Some(CubinModuleSet::new_with_policy(
            key,
            owner,
            CubinSetPolicy::FirstAttentionCaptured,
        ));
        Ok(())
    }
    pub(crate) fn admit_owned_first_attention_gate_cubin_set(
        &mut self,
        key: u64,
        owner: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<(), ServerError> {
        if self.cubin_set.is_some() {
            return Err(failure("one active cubin module set per original actor"));
        }
        self.cubin_set = Some(CubinModuleSet::new_with_policy(
            key,
            owner,
            CubinSetPolicy::FirstAttentionComputedGate,
        ));
        Ok(())
    }
    pub(crate) fn admit_owned_source_rotary_cubin_set(
        &mut self,
        key: u64,
        owner: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<(), ServerError> {
        if self.cubin_set.is_some() {
            return Err(failure("one active cubin module set per original actor"));
        }
        self.cubin_set = Some(CubinModuleSet::new_with_policy(
            key,
            owner,
            CubinSetPolicy::SourceRotary,
        ));
        Ok(())
    }
    pub(crate) fn admit_owned_vocabulary_head_cubin_set(
        &mut self,
        key: u64,
        owner: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<(), ServerError> {
        if self.cubin_set.is_some() {
            return Err(failure("one active cubin module set per original actor"));
        }
        self.cubin_set = Some(CubinModuleSet::new_with_policy(
            key,
            owner,
            CubinSetPolicy::VocabularyHead,
        ));
        Ok(())
    }
    pub(crate) fn admit_owned_full_qwen08_cubin_set(
        &mut self,
        key: u64,
        owner: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<(), ServerError> {
        if self.cubin_set.is_some() {
            return Err(failure("one active cubin module set per original actor"));
        }
        self.cubin_set = Some(CubinModuleSet::new_with_policy(
            key,
            owner,
            CubinSetPolicy::FullQwen08,
        ));
        Ok(())
    }
    pub(crate) fn prepare_owned_cubin(
        &mut self,
        stream: StreamId,
        set_key: u64,
        key: u64,
        image: Arc<VerifiedCubin>,
        manifest: Arc<CubinKernelManifest>,
    ) -> Result<u64, ServerError> {
        manifest.validate()?;
        let allocation_alignment = self.ctx.properties.memory.alignment;
        if !allocation_alignment.is_power_of_two() {
            return Err(failure("cubin original allocation alignment invalid"));
        }
        let h = &self.ctx.properties.hardware;
        let grid = [h.max_cube_count.0, h.max_cube_count.1, h.max_cube_count.2];
        let block = [h.max_cube_dim.0, h.max_cube_dim.1, h.max_cube_dim.2];
        if self.ctx.arch.version != 86
            || image.binary_sm() != manifest.binary_sm
            || manifest.bindings.len() as u32 > h.max_bindings
            || manifest.dynamic_shared_bytes as usize > h.max_shared_memory_size
            || manifest.grid.into_iter().zip(grid).any(|(n, max)| n > max)
            || manifest
                .block
                .into_iter()
                .zip(block)
                .any(|(n, max)| n > max)
            || manifest.block.into_iter().product::<u32>() > h.max_units_per_cube
        {
            return Err(failure("closed cubin original SM86 hardware limits"));
        }
        let image_key = image.image_key();
        {
            let set = self
                .cubin_set
                .as_mut()
                .ok_or_else(|| failure("original cubin set absent"))?;
            set.require_active(set_key)?;
            image.require_admission_owner(set.owner.as_ref().unwrap())?;
            set.policy.require_manifest(&image, &manifest)?;
            if set
                .issuer
                .is_some_and(|issuer| !issuer.same_issuer(image_key))
                || set.records.contains_key(&key)
            {
                return Err(failure(
                    "cubin module set original issuer/record identity differs",
                ));
            }
            let module = set.modules.get(&image_key);
            let function = module.and_then(|m| m.functions.get(&manifest.entry));
            cache::policy_limits(
                set.policy,
                set.modules.len(),
                set.modules.values().map(|m| m.functions.len()).sum(),
                set.records.len(),
                module.is_none(),
                function.is_none(),
            )?;
            if let Some(function) = function {
                if !function.ready {
                    return Err(failure("partial cached cubin function cannot be reused"));
                }
                cache::require_layout(&function.layout, function.dynamic_shared, &manifest)?;
            }
            set.issuer = Some(image_key);
            set.modules.entry(image_key).or_insert_with(|| CubinModule {
                image,
                module: 0,
                ready: false,
                functions: HashMap::new(),
            });
            // Retain the exact record before resolving/creating native state.
            set.records.insert(
                key,
                CubinKernel {
                    image: image_key,
                    manifest: manifest.clone(),
                    ready: false,
                },
            );
        }
        let result = self.prepare_cubin_native(stream, image_key, manifest);
        let set = self.cubin_set.as_mut().unwrap();
        match result {
            Ok(()) => {
                set.records.get_mut(&key).unwrap().ready = true;
                Ok(allocation_alignment)
            }
            Err(error) => {
                set.poisoned = true;
                Err(error)
            }
        }
    }
    fn prepare_cubin_native(
        &mut self,
        stream: StreamId,
        image_key: CubinImageKey,
        manifest: Arc<CubinKernelManifest>,
    ) -> Result<(), ServerError> {
        self.profiled_stream(stream, "pinned cubin prepare")?;
        let module = self
            .cubin_set
            .as_mut()
            .unwrap()
            .modules
            .get_mut(&image_key)
            .unwrap();
        if !module.ready {
            let mut native = std::ptr::null_mut();
            let loaded = unsafe {
                cudarc::driver::sys::cuModuleLoadData(
                    &mut native,
                    module.image.bytes.as_ptr().cast(),
                )
            };
            module.module = native as usize;
            status(loaded, "cuModuleLoadData")?;
            if native.is_null() {
                return Err(failure("cubin successful null module"));
            }
            module.ready = true;
        }
        if module.functions.contains_key(&manifest.entry) {
            return Ok(());
        }
        module.functions.insert(
            manifest.entry.clone(),
            CubinFunction {
                function: 0,
                layout: Vec::new(),
                dynamic_shared: manifest.dynamic_shared_bytes,
                ready: false,
            },
        );
        let function = module.functions.get_mut(&manifest.entry).unwrap();
        let name =
            CString::new(manifest.entry.as_bytes()).map_err(|_| failure("cubin entry NUL"))?;
        let mut native = std::ptr::null_mut();
        let found = unsafe {
            cudarc::driver::sys::cuModuleGetFunction(
                &mut native,
                module.module as CUmodule,
                name.as_ptr(),
            )
        };
        function.function = native as usize;
        status(found, "cuModuleGetFunction")?;
        if native.is_null() {
            return Err(failure("cubin successful null function"));
        }
        for (index, param) in manifest.parameters.iter().enumerate() {
            let mut offset = usize::MAX;
            let mut bytes = usize::MAX;
            status(
                unsafe {
                    cudarc::driver::sys::cuFuncGetParamInfo(native, index, &mut offset, &mut bytes)
                },
                "cuFuncGetParamInfo",
            )?;
            if offset != param.offset || bytes != param.template.len() {
                return Err(failure(
                    "cubin actual parameter offset/size differs from manifest",
                ));
            }
            function.layout.push((offset, bytes));
        }
        let mut offset = 0;
        let mut bytes = 0;
        if unsafe {
            cudarc::driver::sys::cuFuncGetParamInfo(
                native,
                manifest.parameters.len(),
                &mut offset,
                &mut bytes,
            )
        } != CUresult::CUDA_ERROR_INVALID_VALUE
        {
            return Err(failure(
                "cubin actual parameter count differs from manifest",
            ));
        }
        // Shared allowance is immutable for this entry; cached manifests must agree.
        status(
            unsafe {
                cudarc::driver::sys::cuFuncSetAttribute(
                    native,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    manifest.dynamic_shared_bytes as i32,
                )
            },
            "cuFuncSetAttribute",
        )?;
        function.ready = true;
        Ok(())
    }
    pub(crate) fn enqueue_owned_cubin(
        &mut self,
        stream: StreamId,
        set_key: u64,
        key: u64,
        bindings: Vec<Handle>,
        clear: Option<CubinWorkspaceClear>,
        head_selection: Option<crate::owned_command::cubin::head_policy::streamed::Selection>,
    ) -> Result<(), ServerError> {
        let set = self
            .cubin_set
            .as_ref()
            .ok_or_else(|| failure("original cubin set absent"))?;
        set.require_active(set_key)?;
        let record = set
            .records
            .get(&key)
            .filter(|r| r.ready)
            .ok_or_else(|| failure("original cubin manifest absent/not ready"))?;
        let manifest = record.manifest.clone();
        let function = set
            .modules
            .get(&record.image)
            .filter(|m| m.ready)
            .and_then(|m| m.functions.get(&manifest.entry))
            .filter(|f| f.ready)
            .ok_or_else(|| failure("original cached cubin function absent/not ready"))?
            .function;
        if bindings.len() != manifest.bindings.len() {
            return Err(failure("cubin binding count changed"));
        }
        if let Some(selection) = head_selection {
            if clear.is_some() {
                return Err(failure("selected head cannot clear workspace"));
            }
            let module = set
                .modules
                .get(&record.image)
                .ok_or_else(|| failure("selected head image absent"))?;
            selection.require(set.policy, &module.image, &manifest)?;
        }
        // Only validated, completely overwritten reads can lose their prior
        // dependency. No existing failure claim changes during validation.
        if let Some(clear) = clear {
            clear.validate(&manifest)?;
        }
        let original_service = bindings.iter().all(|h| h.service == self.utilities.service);
        let native: Vec<_> = bindings.into_iter().map(Handle::binding).collect();
        let mut written = self.write_set();
        for (index, (binding, spec)) in native.iter().zip(&manifest.bindings).enumerate() {
            if !spec.io.writes() {
                continue;
            }
            if let Some(clear) = clear.filter(|c| c.binding() == index) {
                for range in clear.write_regions(&manifest)? {
                    written.push(binding_region(binding, range)?);
                }
            } else if index == 2 {
                if let Some(
                    crate::owned_command::cubin::head_policy::streamed::Selection::Streamed(window),
                ) = head_selection
                {
                    written.push(binding_region(binding, window.output_range()?)?);
                } else {
                    written.push(binding.clone());
                }
            } else {
                written.push(binding.clone());
            }
        }
        launch_scope(
            self,
            stream,
            key,
            native
                .iter()
                .zip(&manifest.bindings)
                .enumerate()
                .filter(|(i, (_, b))| b.io.reads() && clear.is_none_or(|c| c.binding() != *i))
                .map(|(_, (h, _))| h),
            written,
        )
        .execute(move |server| {
            if !original_service {
                return Err(failure("cubin actor original handle service differs"));
            }
            let mut command = server.command(stream, native.iter())?;
            if command.stream().capturing.is_recording() {
                return Err(failure("pinned cubin graph capture unsupported"));
            }
            let mut addresses = Vec::with_capacity(native.len());
            let mut original_ranges =
                Vec::with_capacity(if head_selection.is_some() { 3 } else { 0 });
            let mut clear_pointer = None;
            for (index, (binding, spec)) in native.into_iter().zip(&manifest.bindings).enumerate() {
                let resource = command.resource(binding)?;
                let expected_bytes = match head_selection {
                    Some(selection) => selection.full_bytes(index)?,
                    None => spec.bytes,
                };
                if resource.size != expected_bytes {
                    return Err(failure("cubin actual original buffer extent differs"));
                }
                if let Some(clear) = clear.filter(|c| c.binding() == index) {
                    clear_pointer = Some(clear.checked_native_range(
                        spec.bytes,
                        resource.size,
                        resource.ptr,
                    )?);
                }
                if head_selection.is_some() {
                    original_ranges.push((resource.ptr, resource.size));
                }
                addresses.push(resource.ptr);
            }
            if let Some(selection) = head_selection {
                // Native complete-resource range/nonalias checks precede the
                // selected pointer substitution; no cropped Handle enters.
                addresses = selection.checked_addresses(&original_ranges)?.to_vec();
            }
            let mut parameters = manifest.arguments(&addresses)?;
            let mut pointers: Vec<*mut c_void> = parameters
                .iter_mut()
                .map(|p| p.as_mut_ptr().cast())
                .collect();
            if let Some(clear) = clear {
                let pointer = clear_pointer
                    .ok_or_else(|| failure("cubin validated workspace clear pointer absent"))?;
                // Full resources and arguments are valid before either enqueue.
                // The original command selects the context and physical stream.
                status(
                    unsafe {
                        cudarc::driver::sys::cuMemsetD8Async(
                            pointer,
                            0,
                            clear.bytes(),
                            command.stream().sys,
                        )
                    },
                    "cuMemsetD8Async workspace prefix",
                )?;
            }
            unsafe {
                cudarc::driver::result::launch_kernel(
                    function as CUfunction,
                    (manifest.grid[0], manifest.grid[1], manifest.grid[2]),
                    (manifest.block[0], manifest.block[1], manifest.block[2]),
                    manifest.dynamic_shared_bytes,
                    command.stream().sys,
                    &mut pointers,
                )
            }
            .map_err(|e| failure(format!("original cubin cuLaunchKernel: {e:?}")))
        })
        .into_result()
    }
    pub(crate) fn retire_owned_cubin(&mut self, set_key: u64, key: u64) -> Result<(), ServerError> {
        let set = self
            .cubin_set
            .as_mut()
            .ok_or_else(|| failure("original cubin set absent"))?;
        set.require_active(set_key)?;
        if set.records.remove(&key).is_none() {
            return Err(failure("original cubin record retirement absent"));
        }
        // No unload: every function in this image shares the same module lifetime.
        Ok(())
    }
    pub(crate) fn retire_owned_cubin_set(&mut self, set_key: u64) -> Result<(), ServerError> {
        let set = self
            .cubin_set
            .as_ref()
            .ok_or_else(|| failure("original cubin set retirement absent"))?;
        set.require_active(set_key)?;
        set.ready_to_retire()?;
        if let Err(error) = self.ctx.unsafe_set_current() {
            self.cubin_set.as_mut().unwrap().poisoned = true;
            return Err(failure(format!(
                "cubin set retire original context: {error:?}"
            )));
        }
        let set = self.cubin_set.as_mut().unwrap();
        for module in set.modules.values_mut() {
            let result = status(
                unsafe { cudarc::driver::sys::cuModuleUnload(module.module as CUmodule) },
                "cuModuleUnload after original completion",
            );
            if let Err(error) = result {
                set.poisoned = true;
                return Err(error);
            }
            // Record real success before another module can fail unloading.
            module.module = 0;
            module.ready = false;
            for function in module.functions.values_mut() {
                function.function = 0;
                function.ready = false;
            }
        }
        if let Err(error) = set.finish_retirement() {
            set.poisoned = true;
            return Err(error);
        }
        drop(self.cubin_set.take());
        Ok(())
    }
}
