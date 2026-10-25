//! Tokenizer bootstrap/constructor admission shared by every source owner.
//!
//! `tokenizers 0.22.2` compiles the ByteLevel `RE` program, `LazyBYTES_CHAR`
//! and `CHAR_BYTES` once per process into statics it never drops. That storage
//! belongs to the process, not to a router, a source or a route alias, so this
//! module owns the single process-global lease and every source owner (CPU,
//! GGUF, canonical47, short-HF) admits through it before parsing.
//!
//! The permanent lease and the compiler scratch are both charged *before* the
//! statics are initialised. The first transition is serialised; a partial
//! initialisation keeps its permanent owner, and only the scratch is released.
//!
//! The cold constructor is separate: its peak (`serde_json`/`serde_core`
//! Content and flattened Value trees, typed merge vector, the cloned BPE model,
//! the compiled split program) exists only while the model is built, so it is
//! admitted as `Scratch` and released as soon as the tokenizer exists. Callers
//! that hold the exact immutable source also run the reviewed borrowing JSON
//! census first and charge `3*C + census_scratch_peak(F)` in addition to the
//! static constructor rows; `C` is the censused tree upper bound.

use std::sync::{Arc, Mutex, MutexGuard};

use zerogravity_artifacts::json_allocation::{ValueConstruction, census_json, census_scratch_peak};
pub use zerogravity_core::resource_arbiter::TransferAdmission;
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, GlobalResourceLease, ResourceOwner,
    lease_actual_inventory_bytes,
};

use crate::tokenize::{self, QwenTokenizer, TokenizerPlan};

struct Registry {
    bootstrap: Option<Arc<Bootstrap>>,
    /// Preserved even when priming fails after a global static may exist.
    permanent: Option<Arc<GlobalResourceLease>>,
}

static REGISTRY: Mutex<Registry> = Mutex::new(Registry {
    bootstrap: None,
    permanent: None,
});

fn registry() -> MutexGuard<'static, Registry> {
    REGISTRY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Process-lifetime tokenizer bootstrap plus the approved profile plan.
pub struct Bootstrap {
    plan: TokenizerPlan,
    /// Typed lease in the original host arbiter/key domain. No erasure.
    lease: Arc<GlobalResourceLease>,
}

/// Admit the process-global ByteLevel owner exactly once.
///
/// The permanent credit and compiler scratch are acquired before the first
/// static is touched. The permanent owner is published before priming so a
/// later failure preserves the charge for the statics that may already exist.
/// Every reuse validates the original arbiter/key domain.
pub fn approve_arbiter(host: &TransferAdmission, session: &str) -> crate::Result<Arc<Bootstrap>> {
    let mut registry = registry();
    if let Some(published) = registry.bootstrap.as_ref() {
        published.validate_domain(host)?;
        return Ok(Arc::clone(published));
    }
    let plan = tokenize::approved_plan()?;
    let permanent = match registry.permanent.as_ref() {
        Some(permanent) => {
            permanent.bytes_in_domain(&host.arbiter, &host.resource)?;
            Arc::clone(permanent)
        }
        None => Arc::new(arbiter_lease(
            host,
            session,
            AllocationClass::ImmutableWeights,
            plan.bootstrap_bytes,
        )?),
    };
    // Failure here leaves the statics untouched. A newly admitted permanent
    // lease is still local and is released; an earlier preserved owner stays.
    let scratch = arbiter_lease(
        host,
        session,
        AllocationClass::Scratch,
        plan.bootstrap_peak_bytes,
    )?;
    // Preserve the permanent owner before any global can be initialised.
    registry.permanent = Some(Arc::clone(&permanent));
    match tokenize::prime_byte_level() {
        Ok(()) => {
            drop(scratch);
            let candidate = Arc::new(Bootstrap {
                plan,
                lease: permanent,
            });
            registry.bootstrap = Some(Arc::clone(&candidate));
            Ok(candidate)
        }
        Err(error) => {
            drop(scratch);
            // The permanent owner remains in the registry: a partially
            // initialised static is never left uncharged.
            Err(error)
        }
    }
}

impl Bootstrap {
    pub fn plan(&self) -> TokenizerPlan {
        self.plan
    }

    /// Validate that this process-lifetime owner still belongs to the caller's
    /// original arbiter/key domain.
    pub fn validate_domain(&self, host: &TransferAdmission) -> crate::Result<()> {
        let bytes = self.lease.bytes_in_domain(&host.arbiter, &host.resource)?;
        if bytes < self.plan.bootstrap_bytes {
            return Err(crate::Error::State(
                "tokenizer bootstrap lease is smaller than its priced plan".into(),
            ));
        }
        Ok(())
    }

    /// Cold-constructor scratch for one source whose exact bytes are not held
    /// here. The Value/Content tree is charged by the source-derived envelope.
    pub fn constructor_arbiter(
        &self,
        host: &TransferAdmission,
        session: &str,
    ) -> crate::Result<Constructor> {
        self.validate_domain(host)?;
        let bytes = self
            .plan
            .bpe_constructor_static_bytes
            .checked_add(self.plan.bpe_constructor_value_bytes)
            .ok_or_else(|| crate::Error::Limit("tokenizer constructor peak".into()))?;
        Ok(Constructor {
            _lease: admit_direct(host, session, AllocationClass::Scratch, bytes)?,
        })
    }

    /// Cold-constructor scratch for one source plus the reviewed borrowing
    /// census of its exact immutable bytes. The census traversal scratch is
    /// released after the census; `3*C + census_scratch_peak(F)` is then held
    /// while the real parse builds the Content/Value trees, in addition to the
    /// static non-Value constructor rows. `C` is the censused tree upper bound;
    /// a refusal never reaches the constructor admission.
    pub fn constructor_of_arbiter(
        &self,
        host: &TransferAdmission,
        session: &str,
        source: &[u8],
    ) -> crate::Result<Constructor> {
        self.validate_domain(host)?;
        let source_len = u64::try_from(source.len())
            .map_err(|_| crate::Error::Limit("json census source length".into()))?;
        let scratch_bytes = u64::try_from(census_scratch_peak(source.len())?)
            .map_err(|_| crate::Error::Limit("json census scratch".into()))?;
        let scratch = admit_direct(host, session, AllocationClass::Scratch, scratch_bytes)?;
        let census = census_json(source, ValueConstruction::SerdeValue);
        drop(scratch);
        // A refusal is not a zero-cost admission: the parse itself would
        // refuse the bytes, so never continue with an unpriced constructor.
        let census = census?;
        let census_tree = u64::try_from(census.tree_heap_upper())
            .map_err(|_| crate::Error::Limit("json census tree peak".into()))?;
        let tree = tokenize::constructor_tree_bytes(census_tree, source_len)?;
        let bytes = self
            .plan
            .bpe_constructor_static_bytes
            .checked_add(tree)
            .ok_or_else(|| crate::Error::Limit("tokenizer constructor peak".into()))?;
        Ok(Constructor {
            _lease: admit_direct(host, session, AllocationClass::Scratch, bytes)?,
        })
    }

    /// Retained `R` Split/BPE inventory plus reserved `G + V` request scratch
    /// for one actual tokenizer owner; aliases of the same tokenizer share the
    /// owner, distinct tokenizer instances own distinct credits. The lease
    /// lives as long as the model that owns the tokenizer. The ByteLevel
    /// program is the process-global owner, charged by [`approve_arbiter`], not
    /// here. `G` is reserved application capacity for the tokenizer's existing
    /// private mutex; `V` is its serialized fancy-regex VM request scratch.
    /// Both are prepaid once per actual tokenizer, not per request.
    pub fn retain_arbiter(
        &self,
        host: &TransferAdmission,
        session: &str,
    ) -> crate::Result<Retained> {
        self.validate_domain(host)?;
        let retained = self.plan.retained_bytes;
        let growth = self
            .plan
            .request_cache_growth_bytes
            .checked_add(self.plan.serialized_vm_bytes)
            .ok_or_else(|| crate::Error::Limit("tokenizer cache growth plus VM".into()))?;
        if retained == 0 || growth == 0 {
            return Err(crate::Error::Limit("zero retained tokenizer credit".into()));
        }
        let owner = ResourceOwner::new(&host.owner.model, session)?;
        let inventory = lease_actual_inventory_bytes(&host.resource, &owner, 3)?;
        Ok(Retained {
            _lease: host.arbiter.admit(
                &host.resource,
                owner,
                &[
                    AllocationRequest {
                        kind: AllocationClass::ImmutableWeights,
                        bytes: retained,
                    },
                    AllocationRequest {
                        kind: AllocationClass::Scratch,
                        bytes: growth,
                    },
                    AllocationRequest {
                        kind: AllocationClass::Other,
                        bytes: inventory,
                    },
                ],
            )?,
        })
    }

    /// The parsed source must agree with the plan it was admitted under.
    pub fn verify(&self, tokenizer: &QwenTokenizer) -> crate::Result<()> {
        if tokenizer.serving_inventory()? != self.plan {
            return Err(crate::Error::State(
                "tokenizer inventory disagrees with the admitted plan".into(),
            ));
        }
        Ok(())
    }
}

/// Constructor scratch lease; dropping it releases the ledger entry.
pub struct Constructor {
    _lease: GlobalResourceLease,
}

/// Retained tokenizer `R + G + V` lease; held by the model that owns it. `R`
/// is immutable Split/BPE inventory; `G + V` is application scratch capacity
/// for the tokenizer's existing private mutex and serialized fancy-regex VM,
/// shared by aliases of this instance.
pub struct Retained {
    _lease: GlobalResourceLease,
}

fn arbiter_lease(
    host: &TransferAdmission,
    session: &str,
    kind: AllocationClass,
    bytes: u64,
) -> crate::Result<GlobalResourceLease> {
    admit_direct(host, session, kind, bytes)
}

/// One direct admission with C's checked lease-inventory overhead folded into
/// the same reservation. This is the only path A uses; no erased owner, no
/// recursive admission and no caller closure authority.
fn admit_direct(
    host: &TransferAdmission,
    session: &str,
    kind: AllocationClass,
    bytes: u64,
) -> crate::Result<GlobalResourceLease> {
    if bytes == 0 {
        return Err(crate::Error::Limit(
            "zero tokenizer allocation request".into(),
        ));
    }
    let owner = ResourceOwner::new(&host.owner.model, session)?;
    let inventory = lease_actual_inventory_bytes(&host.resource, &owner, 2)?;
    host.arbiter.admit(
        &host.resource,
        owner,
        &[
            AllocationRequest { kind, bytes },
            AllocationRequest {
                kind: AllocationClass::Other,
                bytes: inventory,
            },
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter};

    const HOST_MIB: u64 = 4096;
    /// One shared positive domain: the process-global bootstrap is published
    /// once, so every accounting test reuses the same original arbiter/key.
    const SHARED: &str = "tokenizer-admission-shared";

    /// The bootstrap lease is process-global, so these accounting tests must
    /// not interleave on the shared arbiter domain.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn serial() -> std::sync::MutexGuard<'static, ()> {
        SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn host(name: &str, mib: u64) -> crate::Result<TransferAdmission> {
        let arbiter = ResourceArbiter::process_shared();
        let resource = PhysicalDeviceKey::new("host", name)?;
        arbiter.register_matching(resource.clone(), mib * 1024 * 1024)?;
        Ok(TransferAdmission {
            arbiter,
            resource,
            owner: ResourceOwner::new("tokenizer-admission", "test")?,
        })
    }

    fn reserved(host: &TransferAdmission) -> crate::Result<u64> {
        Ok(host.arbiter.snapshot(&host.resource)?.reserved)
    }

    fn total(host: &TransferAdmission, session: &str, bytes: u64) -> u64 {
        let owner = ResourceOwner::new(&host.owner.model, session).unwrap();
        bytes + lease_actual_inventory_bytes(&host.resource, &owner, 2).unwrap()
    }

    fn source_credit_total(host: &TransferAdmission, session: &str, plan: &TokenizerPlan) -> u64 {
        let owner = ResourceOwner::new(&host.owner.model, session).unwrap();
        plan.retained_bytes
            .checked_add(plan.request_cache_growth_bytes)
            .and_then(|bytes| bytes.checked_add(plan.serialized_vm_bytes))
            .unwrap()
            .checked_add(lease_actual_inventory_bytes(&host.resource, &owner, 3).unwrap())
            .unwrap()
    }

    #[test]
    fn bootstrap_is_charged_once_and_constructor_scratch_is_released() -> crate::Result<()> {
        let _serial = serial();
        let host = host(SHARED, HOST_MIB)?;
        let plan = tokenize::approved_plan()?;
        let before = reserved(&host)?;
        let bootstrap = approve_arbiter(&host, "bootstrap")?;
        bootstrap.validate_domain(&host)?;
        let charged = reserved(&host)?;
        // A previous test in this process may already have published the
        // process-lifetime owner only in its own domain; this request either
        // charged this domain or was refused, never silently inherited.
        assert!(charged >= before);
        assert_eq!(bootstrap.plan(), plan);
        // Later owners in the same original domain reuse the published lease
        // without adding a second permanent charge.
        let second = approve_arbiter(&host, "bootstrap-again")?;
        second.validate_domain(&host)?;
        assert_eq!(reserved(&host)?, charged);
        assert_eq!(second.plan(), plan);
        // Cold constructor scratch exists only while the source is built.
        let constructor = second.constructor_arbiter(&host, "constructor")?;
        let peak = plan
            .bpe_constructor_static_bytes
            .saturating_add(plan.bpe_constructor_value_bytes);
        assert_eq!(
            reserved(&host)?,
            charged + total(&host, "constructor", peak)
        );
        // Retained inventory lives as long as the owner that holds it.
        let retained = second.retain_arbiter(&host, "retained")?;
        let credit = source_credit_total(&host, "retained", &plan);
        assert_eq!(
            reserved(&host)?,
            charged + total(&host, "constructor", peak) + credit
        );
        drop(constructor);
        assert_eq!(reserved(&host)?, charged + credit);
        drop(retained);
        assert_eq!(reserved(&host)?, charged);
        // The source subtotal is one `bootstrap + R + G + V` charge; only the
        // admitted prompt (without `V`) and decode are per request. The
        // constructor peak is released before any request arrives. Compute the
        // same checked one-request subtotal here from the public plan/peak
        // functions; the test-only `fits` helper lives in the model crate's
        // private module.
        let source = plan
            .bootstrap_bytes
            .checked_add(plan.retained_bytes)
            .and_then(|bytes| bytes.checked_add(plan.request_cache_growth_bytes))
            .and_then(|bytes| bytes.checked_add(plan.serialized_vm_bytes))
            .ok_or_else(|| crate::Error::Limit("tokenizer source subtotal".into()))?;
        let request = tokenize::admitted_prompt_peak(512)
            .and_then(|prompt| {
                tokenize::decode_peak(512).and_then(|decode| prompt.checked_add(decode))
            })
            .ok_or_else(|| crate::Error::Limit("tokenizer request subtotal".into()))?;
        assert!(
            source
                .checked_add(request)
                .is_some_and(|subtotal| subtotal <= HOST_MIB * 1024 * 1024),
            "source subtotal plus one request must fit the 4 GiB host"
        );
        Ok(())
    }

    #[test]
    fn distinct_retained_owners_and_arc_alias_lifetime_are_independent() -> crate::Result<()> {
        let _serial = serial();
        let host = host(SHARED, HOST_MIB)?;
        let plan = tokenize::approved_plan()?;
        let bootstrap = approve_arbiter(&host, "owner-bootstrap")?;
        let charged = reserved(&host)?;
        // Two distinct `Retained` owners each carry their own `R + G + V + L`
        // credit; no parser or model is instantiated by this ledger test.
        let source_a = Arc::new(bootstrap.retain_arbiter(&host, "source-a")?);
        let a = source_credit_total(&host, "source-a", &plan);
        assert_eq!(reserved(&host)?, charged + a);
        let source_b = bootstrap.retain_arbiter(&host, "source-b")?;
        let b = source_credit_total(&host, "source-b", &plan);
        assert_eq!(reserved(&host)?, charged + a + b);
        // `Arc` clones alias one retained owner; dropping one handle keeps its
        // credit alive.
        let alias = Arc::clone(&source_a);
        drop(source_a);
        assert_eq!(reserved(&host)?, charged + a + b);
        drop(alias);
        assert_eq!(reserved(&host)?, charged + b);
        drop(source_b);
        assert_eq!(reserved(&host)?, charged);
        Ok(())
    }

    #[test]
    fn census_constructor_charges_static_rows_plus_three_tree_rows_and_scratch() -> crate::Result<()>
    {
        let _serial = serial();
        let host = host(SHARED, HOST_MIB)?;
        let plan = tokenize::approved_plan()?;
        let bootstrap = approve_arbiter(&host, "census-bootstrap")?;
        let charged = reserved(&host)?;
        // A small well-formed JSON document: the static rows dominate.
        let document = br#"{"version":"1.0","model":{"type":"BPE"}}"#;
        let census = census_json(document, ValueConstruction::SerdeValue)?;
        let census_peak = u64::try_from(census.tree_heap_upper())
            .map_err(|_| crate::Error::Limit("json census tree peak".into()))?;
        let tree = tokenize::constructor_tree_bytes(census_peak, document.len() as u64)?;
        let constructor = bootstrap.constructor_of_arbiter(&host, "census", document)?;
        let peak = plan
            .bpe_constructor_static_bytes
            .checked_add(tree)
            .ok_or_else(|| crate::Error::Limit("tokenizer constructor peak".into()))?;
        assert_eq!(reserved(&host)?, charged + total(&host, "census", peak));
        drop(constructor);
        assert_eq!(reserved(&host)?, charged);
        Ok(())
    }

    #[test]
    fn census_refusal_releases_scratch_and_never_charges_the_constructor() -> crate::Result<()> {
        let _serial = serial();
        let host = host(SHARED, HOST_MIB)?;
        let bootstrap = approve_arbiter(&host, "refusal-bootstrap")?;
        let charged = reserved(&host)?;
        let refused = br#"{"model":{"type":"BPE"}"#;
        let error = bootstrap
            .constructor_of_arbiter(&host, "refused-census", refused)
            .err()
            .unwrap();
        assert!(matches!(
            error,
            crate::Error::Limit(_) | crate::Error::Malformed(_)
        ));
        assert_eq!(reserved(&host)?, charged);
        Ok(())
    }

    #[test]
    fn bootstrap_reuse_in_another_domain_is_refused() -> crate::Result<()> {
        let _serial = serial();
        let first = host(SHARED, HOST_MIB)?;
        let published = approve_arbiter(&first, "first")?;
        published.validate_domain(&first)?;
        let other = host("tokenizer-admission-other-domain", HOST_MIB)?;
        let error = approve_arbiter(&other, "other").err().unwrap();
        assert!(matches!(error, crate::Error::State(_)));
        // A cached bootstrap cannot lend its permanent owner to a different
        // original key: constructor and retained admissions revalidate first.
        assert!(published.validate_domain(&other).is_err());
        assert!(
            published
                .constructor_arbiter(&other, "other-constructor")
                .is_err()
        );
        assert!(published.retain_arbiter(&other, "other-retained").is_err());
        Ok(())
    }

    #[test]
    fn failed_pre_initialization_admission_leaves_no_statics_charge() -> crate::Result<()> {
        let _serial = serial();
        // A distinct tiny domain: the permanent or scratch admission must fail
        // before any global is touched, unless the already-published global
        // refuses this domain instead. Both outcomes are refusals.
        let tiny = host("tokenizer-admission-tiny", 1)?;
        assert!(approve_arbiter(&tiny, "tiny").is_err());
        Ok(())
    }

    #[test]
    fn zero_byte_requests_are_refused() -> crate::Result<()> {
        let _serial = serial();
        let host = host(SHARED, HOST_MIB)?;
        let error = arbiter_lease(&host, "zero", AllocationClass::Scratch, 0)
            .err()
            .unwrap();
        assert!(matches!(error, crate::Error::Limit(_)));
        Ok(())
    }
}
