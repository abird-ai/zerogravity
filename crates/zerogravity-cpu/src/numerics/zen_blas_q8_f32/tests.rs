use super::*;
#[cfg(target_arch = "x86_64")]
use std::sync::Arc;
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};

struct Budget {
    arbiter: ResourceArbiter,
    resource: PhysicalDeviceKey,
    owner: ResourceOwner,
}
impl Budget {
    fn new(bytes: u64) -> Self {
        let arbiter = ResourceArbiter::new();
        let resource = PhysicalDeviceKey::new("host", "closed-zen-unit").unwrap();
        arbiter.register(resource.clone(), bytes).unwrap();
        Self {
            arbiter,
            resource,
            owner: ResourceOwner::new("source-free", "zen-unit").unwrap(),
        }
    }
    fn admission(&self) -> BulkF32Admission<'_> {
        BulkF32Admission {
            arbiter: &self.arbiter,
            resource: &self.resource,
            owner: &self.owner,
        }
    }
    fn reserved(&self) -> u64 {
        self.arbiter.snapshot(&self.resource).unwrap().reserved
    }
}

#[test]
fn even_odd_fma_and_panel_reset_have_independent_exact_counterexamples() {
    let mut weights = [1.0f32; WIDTH];
    let mut input = [0.0f32; WIDTH];
    input[..4].copy_from_slice(&[16_777_216.0, 1.0, -16_777_216.0, 1.0]);
    // Two interleaved source chains give2; a single ordered K chain gives1.
    assert_eq!(cell(&weights, &input).to_bits(), 2.0f32.to_bits());

    input.fill(0.0);
    input[0] = 16_777_216.0;
    input[320] = 1.0;
    input[322] = -16_777_216.0;
    input[324] = 1.0;
    // Panel1 starts fresh: its sum−16777214 added to panel0 gives2.
    // Carrying panel0 into its even chain would lose the first1 and give1.
    assert_eq!(cell(&weights, &input).to_bits(), 2.0f32.to_bits());

    // A source-representable Q8_0 pair in separate32-word blocks. The product
    // atK32 has a2^-33 residual beyond an f32 multiply's rounding boundary.
    weights.fill(0.0);
    input.fill(0.0);
    weights[0] = 1.0;
    weights[32] = f32::from_bits(0x3f80_2000);
    input[0] = -f32::from_bits(0x3f7f_fff2);
    input[32] = f32::from_bits(0x3f7f_c002);
    assert_eq!(cell(&weights, &input).to_bits(), 0x2f00_0000);
    assert_eq!(
        ((weights[32] * input[32]) + input[0]).to_bits(),
        0.0f32.to_bits()
    );
}

#[test]
fn panel_write_retains_source_signed_zero_and_final224_panel() {
    let mut weights = [0.0f32; WIDTH];
    let mut input = [0.0f32; WIDTH];
    // +0 initial C and a final alpha1 FMA preserve the source's zero sign.
    weights.fill(-0.0);
    input.fill(1.0);
    assert_eq!(cell(&weights, &input).to_bits(), 0.0f32.to_bits());
    weights[2047] = -2.0;
    input[2047] = 3.0;
    assert_eq!(cell(&weights, &input).to_bits(), (-6.0f32).to_bits());
}

#[cfg(target_arch = "x86_64")]
fn request<'a>(weights: &'a [u8], input: &'a [f32]) -> ZenQ8F32Request<'a> {
    ZenQ8F32Request {
        source: SourceIdentity::for_new_allocation().unwrap(),
        weights,
        input,
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn exact_source_lengths_and_finite_domain_refuse_without_admission() {
    let mut weights = vec![0u8; WEIGHT_BYTES];
    let mut input = vec![0.0f32; OUTPUT_WORDS];
    let budget = Budget::new(SCRATCH_BYTES + OUTPUT_BYTES);
    let mut unbound = request(&weights, &input);
    unbound.source = SourceIdentity::unbound();
    assert!(matches!(
        execute(unbound, budget.admission(), None),
        Err(Error::Malformed(_))
    ));
    for invalid in [
        request(&weights[..WEIGHT_BYTES - 1], &input),
        request(&weights, &input[..OUTPUT_WORDS - 1]),
    ] {
        assert!(matches!(
            execute(invalid, budget.admission(), None),
            Err(Error::Malformed(_))
        ));
    }
    input[OUTPUT_WORDS - 1] = f32::NAN;
    assert!(matches!(
        execute(request(&weights, &input), budget.admission(), None),
        Err(Error::Malformed(_))
    ));
    input[OUTPUT_WORDS - 1] = 0.0;
    weights[WEIGHT_BYTES - 34..WEIGHT_BYTES - 32].copy_from_slice(&0x7c00u16.to_le_bytes());
    assert!(matches!(
        execute(request(&weights, &input), budget.admission(), None),
        Err(Error::Malformed(_))
    ));
    assert_eq!(budget.reserved(), 0);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn one_byte_pressure_and_pre_cancel_leave_no_private_reservation() {
    let weights = vec![0u8; WEIGHT_BYTES];
    let input = vec![0.0f32; OUTPUT_WORDS];
    let budget = Budget::new(SCRATCH_BYTES + OUTPUT_BYTES - 1);
    assert!(matches!(
        execute(request(&weights, &input), budget.admission(), None),
        Err(Error::Limit(_))
    ));
    assert_eq!(budget.reserved(), 0);
    let budget = Budget::new(SCRATCH_BYTES + OUTPUT_BYTES);
    let cancelled = AtomicBool::new(true);
    assert!(matches!(
        execute(
            request(&weights, &input),
            budget.admission(),
            Some(&cancelled)
        ),
        Err(Error::State(_))
    ));
    assert_eq!(budget.reserved(), 0);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn cancellation_after_real_output_holds_both_tickets_then_releases_them() {
    let weights = vec![0u8; WEIGHT_BYTES];
    let input = vec![0.0f32; OUTPUT_WORDS];
    let budget = Budget::new(SCRATCH_BYTES + OUTPUT_BYTES);
    let cancelled = Arc::new(AtomicBool::new(false));
    let observed_cancelled = cancelled.clone();
    let observed_arbiter = budget.arbiter.clone();
    let observed_resource = budget.resource.clone();
    AFTER_FIRST_OUTPUT.with(|hook| {
        *hook.borrow_mut() = Some(Box::new(move || {
            let snapshot = observed_arbiter.snapshot(&observed_resource).unwrap();
            assert_eq!(snapshot.reservations, 2);
            assert_eq!(snapshot.per_class[&AllocationClass::Scratch], SCRATCH_BYTES);
            assert_eq!(snapshot.per_class[&AllocationClass::Output], OUTPUT_BYTES);
            observed_cancelled.store(true, Ordering::Release);
        }));
    });
    assert!(matches!(
        execute(
            request(&weights, &input),
            budget.admission(),
            Some(&cancelled)
        ),
        Err(Error::State(_))
    ));
    assert!(cancelled.load(Ordering::Acquire));
    assert_eq!(budget.reserved(), 0);
    assert!(!budget.arbiter.snapshot(&budget.resource).unwrap().fenced);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn finite_input_overflow_retires_full_dequant_and_partial_output() {
    let mut weights = vec![0u8; WEIGHT_BYTES];
    weights[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
    weights[2] = 2;
    let input = vec![f32::MAX; OUTPUT_WORDS];
    let budget = Budget::new(SCRATCH_BYTES + OUTPUT_BYTES);
    assert!(matches!(
        execute(request(&weights, &input), budget.admission(), None),
        Err(Error::Limit(_))
    ));
    assert_eq!(budget.reserved(), 0);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn floating_point_controls_refuse_before_and_after_private_math_without_repair() {
    fn read() -> u32 {
        let mut value = 0u32;
        // SAFETY: the read-only control instruction writes one valid word.
        unsafe {
            std::arch::asm!("stmxcsr [{at}]", at = in(reg) &mut value,
                options(nostack, preserves_flags));
        }
        value
    }
    struct Restore(u32);
    impl Drop for Restore {
        fn drop(&mut self) {
            // SAFETY: restore this thread's own saved valid controls.
            unsafe {
                std::arch::asm!("ldmxcsr [{at}]", at = in(reg) &self.0,
                    options(nostack, preserves_flags));
            }
        }
    }
    let original = Restore(read());
    let weights = vec![0u8; WEIGHT_BYTES];
    let input = vec![0.0f32; OUTPUT_WORDS];
    let budget = Budget::new(SCRATCH_BYTES + OUTPUT_BYTES);
    let changed = (original.0 & !0xffc0) | 0x1f80 | (1 << 15);
    // SAFETY: documented FTZ only, scoped to this test thread and Restore.
    unsafe {
        std::arch::asm!("ldmxcsr [{at}]", at = in(reg) &changed,
            options(nostack, preserves_flags));
    }
    assert!(matches!(
        execute(request(&weights, &input), budget.admission(), None),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(read(), changed);
    assert_eq!(budget.reserved(), 0);
    // SAFETY: restore before testing a mutation after actual output exists.
    unsafe {
        std::arch::asm!("ldmxcsr [{at}]", at = in(reg) &original.0,
            options(nostack, preserves_flags));
    }
    AFTER_FIRST_OUTPUT.with(|hook| {
        *hook.borrow_mut() = Some(Box::new(|| {
            let changed = read() | (1 << 6);
            // SAFETY: documented DAZ only; the outer Restore remains live.
            unsafe {
                std::arch::asm!("ldmxcsr [{at}]", at = in(reg) &changed,
                    options(nostack, preserves_flags));
            }
        }));
    });
    assert!(matches!(
        execute(request(&weights, &input), budget.admission(), None),
        Err(Error::Unsupported(_))
    ));
    assert_ne!(read() & (1 << 6), 0);
    assert_eq!(budget.reserved(), 0);
}

#[test]
#[cfg(not(target_arch = "x86_64"))]
fn unqualified_platform_refuses_without_admission() {
    let budget = Budget::new(SCRATCH_BYTES + OUTPUT_BYTES);
    assert!(matches!(
        execute(
            ZenQ8F32Request {
                source: SourceIdentity::for_new_allocation().unwrap(),
                weights: &[],
                input: &[]
            },
            budget.admission(),
            None
        ),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(budget.reserved(), 0);
}
