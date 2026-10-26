//! Original actor's bounded module/function/manifest ownership.
use super::*;
use std::any::Any;

pub(crate) struct CubinFunction {
    pub function: usize,
    pub layout: Vec<(usize, usize)>,
    pub dynamic_shared: u32,
    pub ready: bool,
}
pub(crate) struct CubinModule {
    pub image: Arc<VerifiedCubin>,
    pub module: usize,
    pub ready: bool,
    pub functions: HashMap<String, CubinFunction>,
}
pub(crate) struct CubinKernel {
    pub image: CubinImageKey,
    pub manifest: Arc<CubinKernelManifest>,
    pub ready: bool,
}
pub(crate) struct CubinModuleSet {
    pub key: u64,
    pub(super) policy: CubinSetPolicy,
    pub owner: Option<Arc<dyn Any + Send + Sync>>,
    pub issuer: Option<CubinImageKey>,
    pub modules: HashMap<CubinImageKey, CubinModule>,
    pub records: HashMap<u64, CubinKernel>,
    pub poisoned: bool,
    pub retired: bool,
}
impl std::fmt::Debug for CubinModuleSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CubinModuleSet")
            .field("key", &self.key)
            .field("policy", &self.policy)
            .field("modules", &self.modules.len())
            .field("records", &self.records.len())
            .field("poisoned", &self.poisoned)
            .field("retired", &self.retired)
            .finish()
    }
}
impl CubinModuleSet {
    pub fn new(key: u64, owner: Arc<dyn Any + Send + Sync>) -> Self {
        Self::new_with_policy(key, owner, CubinSetPolicy::ProjectionSix)
    }
    pub(super) fn new_with_policy(
        key: u64,
        owner: Arc<dyn Any + Send + Sync>,
        policy: CubinSetPolicy,
    ) -> Self {
        Self {
            key,
            policy,
            owner: Some(owner),
            issuer: None,
            modules: HashMap::new(),
            records: HashMap::new(),
            poisoned: false,
            retired: false,
        }
    }
    pub fn require_active(&self, key: u64) -> Result<(), ServerError> {
        if self.key != key || self.poisoned || self.retired || self.owner.is_none() {
            return Err(failure("original cubin set absent/poisoned/retired"));
        }
        Ok(())
    }
    pub fn ready_to_retire(&self) -> Result<(), ServerError> {
        self.require_active(self.key)?;
        if !self.records.is_empty() {
            return Err(failure("cubin set still has live manifest records"));
        }
        Ok(())
    }
    pub fn finish_retirement(&mut self) -> Result<(), ServerError> {
        self.ready_to_retire()?;
        if self
            .modules
            .values()
            .any(|m| m.module != 0 || m.functions.values().any(|f| f.function != 0))
        {
            return Err(failure("cubin set native handles not retired"));
        }
        self.retired = true;
        drop(self.owner.take());
        Ok(())
    }
}
impl Drop for CubinModuleSet {
    fn drop(&mut self) {
        if !self.retired {
            // No allocation/driver call during abandonment, including actor loss.
            // Actual partial handles, source images and original charge stay held.
            std::mem::forget(self.owner.take());
            std::mem::forget(std::mem::take(&mut self.modules));
            std::mem::forget(std::mem::take(&mut self.records));
        }
    }
}

pub(super) fn policy_limits(
    policy: CubinSetPolicy,
    modules: usize,
    functions: usize,
    records: usize,
    new_module: bool,
    new_function: bool,
) -> Result<(), ServerError> {
    match policy {
        CubinSetPolicy::ProjectionSix => {
            limits(modules, functions, records, new_module, new_function)
        }
        CubinSetPolicy::Text08
        | CubinSetPolicy::Text08Cohort4
        | CubinSetPolicy::Text2B
        | CubinSetPolicy::Text2BCachedCohort4
        | CubinSetPolicy::FullQwen08
        | CubinSetPolicy::SourceRotary
        | CubinSetPolicy::ConnectedQwen08
        | CubinSetPolicy::FixedPromptMlp
        | CubinSetPolicy::Qwen2bMlp
        | CubinSetPolicy::FixedPromptQkv
        | CubinSetPolicy::FixedPromptQkKv
        | CubinSetPolicy::FixedPromptFa
        | CubinSetPolicy::FixedPromptLayer3
        | CubinSetPolicy::FixedPromptFull
        | CubinSetPolicy::FixedPromptNatural
        | CubinSetPolicy::FirstAttentionCaptured
        | CubinSetPolicy::FirstAttentionComputedGate
        | CubinSetPolicy::VocabularyHead => {
            policy.limits(modules, functions, records, new_module, new_function)
        }
    }
}
pub(super) fn limits(
    modules: usize,
    functions: usize,
    records: usize,
    new_module: bool,
    new_function: bool,
) -> Result<(), ServerError> {
    if modules > 6
        || functions > 8
        || records >= 19
        || (new_module && modules == 6)
        || (new_function && functions == 8)
    {
        return Err(failure(
            "closed cubin six-module/eight-function/nineteen-record capacity",
        ));
    }
    Ok(())
}
pub(super) fn require_layout(
    layout: &[(usize, usize)],
    shared: u32,
    manifest: &CubinKernelManifest,
) -> Result<(), ServerError> {
    if shared != manifest.dynamic_shared_bytes
        || layout.len() != manifest.parameters.len()
        || layout
            .iter()
            .zip(&manifest.parameters)
            .any(|(&(offset, size), p)| offset != p.offset || size != p.template.len())
    {
        return Err(failure(
            "cached cubin function ABI/shared allowance differs",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod ownership_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Owner(Arc<AtomicUsize>);
    impl Drop for Owner {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::AcqRel);
        }
    }
    #[test]
    fn original_charge_releases_only_at_explicit_clean_set_retirement() {
        let drops = Arc::new(AtomicUsize::new(0));
        let owner = Arc::new(Owner(drops.clone()));
        let weak = Arc::downgrade(&owner);
        let mut set = CubinModuleSet::new(1, owner.clone());
        drop(owner);
        assert!(weak.upgrade().is_some());
        assert!(set.require_active(2).is_err());
        set.finish_retirement().unwrap();
        drop(set);
        assert!(weak.upgrade().is_none());
        assert_eq!(drops.load(Ordering::Acquire), 1);
    }
    #[test]
    fn unknown_or_poisoned_actor_drop_retains_original_charge_without_driver_call() {
        for poison in [false, true] {
            let drops = Arc::new(AtomicUsize::new(0));
            let owner = Arc::new(Owner(drops.clone()));
            let weak = Arc::downgrade(&owner);
            let mut set = CubinModuleSet::new(1, owner.clone());
            drop(owner);
            set.poisoned = poison;
            if poison {
                assert!(set.finish_retirement().is_err());
            }
            drop(set);
            assert!(weak.upgrade().is_some());
            assert_eq!(drops.load(Ordering::Acquire), 0);
        }
    }
    #[test]
    fn closed_counts_admit_reuse_but_refuse_an_extra_module_function_or_record() {
        assert!(limits(6, 8, 18, false, false).is_ok());
        for values in [
            (6, 8, 18, true, false),
            (6, 8, 18, false, true),
            (6, 8, 19, false, false),
        ] {
            assert!(limits(values.0, values.1, values.2, values.3, values.4).is_err());
        }
    }
    #[test]
    fn shared_entry_reuse_requires_identical_reflected_layout_and_allowance() {
        use crate::owned_command::{
            CubinBinding, CubinIo, CubinLaunchAttributes, CubinParameter, CubinPointerFixup,
        };
        let mut manifest = CubinKernelManifest {
            entry: "observed".into(),
            binary_sm: 80,
            grid: [1, 1, 1],
            block: [32, 1, 1],
            dynamic_shared_bytes: 33792,
            attributes: CubinLaunchAttributes::None,
            bindings: vec![CubinBinding {
                bytes: 16,
                alignment: 16,
                io: CubinIo::Read,
            }],
            parameters: vec![CubinParameter {
                offset: 0,
                template: vec![0; 8],
            }],
            fixups: vec![CubinPointerFixup {
                parameter: 0,
                byte_offset: 0,
                binding: 0,
                binding_offset: 0,
                extent: 16,
                alignment: 16,
                io: CubinIo::Read,
            }],
        };
        manifest.validate().unwrap();
        require_layout(&[(0, 8)], 33792, &manifest).unwrap();
        assert!(require_layout(&[(0, 8)], 16896, &manifest).is_err());
        assert!(require_layout(&[(0, 16)], 33792, &manifest).is_err());
        assert!(require_layout(&[(0, 8), (8, 8)], 33792, &manifest).is_err());
        manifest.parameters[0].offset = 8;
        assert!(require_layout(&[(0, 8)], 33792, &manifest).is_err());
    }
}
