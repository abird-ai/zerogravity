use super::*;
#[cfg(target_arch = "x86_64")]
use std::sync::Arc;

fn budget(limit: u64) -> (ResourceArbiter, PhysicalDeviceKey, ResourceOwner) {
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("host", "bulk-f32-unit").unwrap();
    arbiter.register(key.clone(), limit).unwrap();
    (
        arbiter,
        key,
        ResourceOwner::new("source-free", "unit").unwrap(),
    )
}

#[test]
#[cfg(target_arch = "x86_64")]
fn non_source_float_controls_refuse_before_admission_without_mutation() {
    struct Restore(u32);
    impl Drop for Restore {
        fn drop(&mut self) {
            // SAFETY: restore the same thread's saved valid MXCSR value.
            unsafe {
                std::arch::asm!("ldmxcsr [{at}]", at = in(reg) &self.0,
                    options(nostack, preserves_flags));
            }
        }
    }
    fn read() -> u32 {
        let mut value = 0u32;
        // SAFETY: writes one valid stack word without changing controls.
        unsafe {
            std::arch::asm!("stmxcsr [{at}]", at = in(reg) &mut value,
                options(nostack, preserves_flags));
        }
        value
    }
    let original = Restore(read());
    let source = SourceIdentity::for_new_allocation().unwrap();
    let a = [1.0; 32];
    let b = [1.0; 16];
    let (arbiter, key, owner) = budget(4096);
    for bits in [1 << 15, 1 << 6, 1 << 13, 1 << 14, 3 << 13] {
        let changed = (original.0 & !0xffc0) | 0x1f80 | bits;
        // SAFETY: only documented rounding/denormal control bits are changed
        // on this test thread; Restore also runs if an assertion panics.
        unsafe {
            std::arch::asm!("ldmxcsr [{at}]", at = in(reg) &changed,
                options(nostack, preserves_flags));
        }
        assert!(matches!(
            execute(
                request(source, &a, &b),
                BulkF32Admission {
                    arbiter: &arbiter,
                    resource: &key,
                    owner: &owner,
                },
                None
            ),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(read(), changed);
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    }
    // Restore source controls, then change them after actual output exists.
    // The public post-check must refuse and retire the private result lease.
    unsafe {
        std::arch::asm!("ldmxcsr [{at}]", at = in(reg) &original.0,
            options(nostack, preserves_flags));
    }
    AFTER_FIRST_OUTPUT.with(|hook| {
        *hook.borrow_mut() = Some(Box::new(|| {
            let changed = read() | (1 << 15);
            // SAFETY: set documented FTZ on this test thread; Restore is live.
            unsafe {
                std::arch::asm!("ldmxcsr [{at}]", at = in(reg) &changed,
                    options(nostack, preserves_flags));
            }
        }));
    });
    assert!(matches!(
        execute(
            request(source, &a, &b),
            BulkF32Admission {
                arbiter: &arbiter,
                resource: &key,
                owner: &owner
            },
            None,
        ),
        Err(Error::Unsupported(_))
    ));
    assert_ne!(read() & (1 << 15), 0);
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
}
fn request<'a>(source: SourceIdentity, a: &'a [f32], b: &'a [f32]) -> BulkF32Request<'a> {
    BulkF32Request {
        source,
        profile: BulkF32Profile::GgmlD812350HaswellTinyBlas,
        left: F32View {
            source,
            name: "left",
            values: a,
            offset: 0,
            shape: [8, 4, 1, 1],
            strides: [1, 8, 32, 32],
        },
        right: F32View {
            source,
            name: "right",
            values: b,
            offset: 0,
            shape: [8, 2, 1, 1],
            strides: [1, 8, 16, 16],
        },
    }
}

#[test]
fn source_one_chain_and_horizontal_order_have_independent_counterexamples() {
    let a = [1.0; 32];
    let mut b = [0.0; 32];
    b[0] = 16_777_216.0;
    b[8] = 1.0;
    b[16] = -16_777_216.0;
    b[24] = 1.0;
    // Four independent Vec8 partials yield2; the original one-chain yields1.
    assert_eq!(dot(&a, &b).to_bits(), 1.0f32.to_bits());
    let h = [67_108_864.0, 1.0, -67_108_864.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    // Even/odd horizontal pairing yields2; adjacent-pair horizontal gives0.
    assert_eq!(dot(&[1.0; 8], &h).to_bits(), 2.0f32.to_bits());
}

#[test]
fn real_zero_filled_tail_fmas_preserve_source_signed_zero_semantics() {
    let tiny = f32::from_bits(1);
    assert_eq!(dot(&[-tiny; 8], &[tiny; 8]).to_bits(), (-0.0f32).to_bits());
    // Original load_partial fills BOTH unused operands with +0 before FMA.
    assert_eq!(dot(&[-tiny; 9], &[tiny; 9]).to_bits(), 0.0f32.to_bits());
    assert_eq!(dot(&[2.0; 3], &[3.0; 3]).to_bits(), 18.0f32.to_bits());
}

#[test]
#[cfg(target_arch = "x86_64")]
fn strided_offset_broadcast_canaries_and_output_lease_are_real() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let mut a = vec![f32::NAN; 128];
    for head in 0..2 {
        for row in 0..4 {
            a[3 + head * 64 + row * 12..3 + head * 64 + row * 12 + 8]
                .fill((head * 4 + row + 1) as f32);
        }
    }
    let mut b = vec![f32::NAN; 66];
    b[1..65].fill(1.0);
    let mut r = request(source, &a, &b);
    r.left.offset = 3;
    r.left.strides = [1, 12, 64, 128];
    r.left.shape = [8, 4, 2, 1];
    r.right.offset = 1;
    r.right.shape = [8, 2, 4, 1];
    r.right.strides = [1, 8, 16, 64];
    let (arbiter, key, owner) = budget(4096);
    let result = execute(
        r,
        BulkF32Admission {
            arbiter: &arbiter,
            resource: &key,
            owner: &owner,
        },
        None,
    )
    .unwrap();
    assert_eq!(result.shape(), [4, 2, 4, 1]);
    for head in 0..4 {
        for column in 0..2 {
            for row in 0..4 {
                assert_eq!(
                    result.values()[row + 4 * (column + 2 * head)],
                    (head / 2 * 4 + row + 1) as f32 * 8.0
                );
            }
        }
    }
    let live = arbiter.snapshot(&key).unwrap();
    assert_eq!(live.reservations, 1);
    assert_eq!(live.per_class[&AllocationClass::Output], 128);
    assert_eq!(live.per_class[&AllocationClass::Scratch], SCRATCH_BYTES);
    assert!(a[0].is_nan() && a[11].is_nan() && b[0].is_nan() && b[65].is_nan());
    assert_eq!(result.source(), source);
    drop(result);
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn foreign_source_stride_span_overflow_and_ar1_are_closed() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let a = [1.0; 32];
    let b = [1.0; 16];
    let (arbiter, key, owner) = budget(4096);
    let valid = request(source, &a, &b);
    let mut cases = Vec::new();
    let mut r = valid;
    r.right.source = SourceIdentity::for_new_allocation().unwrap();
    cases.push(r);
    let mut r = valid;
    r.left.strides[1] = 7;
    cases.push(r);
    let mut r = valid;
    r.left.offset = 1;
    cases.push(r);
    let mut r = valid;
    r.left.offset = usize::MAX;
    cases.push(r);
    let mut r = valid;
    r.left.strides[3] = usize::MAX;
    r.left.shape[3] = 2;
    cases.push(r);
    let mut r = valid;
    r.right.shape[1] = 1;
    r.right.strides = [1, 8, 8, 8];
    cases.push(r);
    let mut r = valid;
    r.right.strides[3] = 17;
    cases.push(r);
    for r in cases {
        assert!(
            execute(
                r,
                BulkF32Admission {
                    arbiter: &arbiter,
                    resource: &key,
                    owner: &owner
                },
                None
            )
            .is_err()
        );
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn actual_capacity_pre_cancel_and_mid_output_cancel_release_only_private_storage() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let a = [1.0; 32];
    let b = [1.0; 16];
    let (arbiter, key, owner) = budget(32 + SCRATCH_BYTES - 1);
    assert!(matches!(
        execute(
            request(source, &a, &b),
            BulkF32Admission {
                arbiter: &arbiter,
                resource: &key,
                owner: &owner
            },
            None
        ),
        Err(Error::Limit(_))
    ));
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    let (arbiter, key, owner) = budget(4096);
    let flag = Arc::new(AtomicBool::new(true));
    assert!(
        execute(
            request(source, &a, &b),
            BulkF32Admission {
                arbiter: &arbiter,
                resource: &key,
                owner: &owner
            },
            Some(&flag)
        )
        .is_err()
    );
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    flag.store(false, Ordering::Release);
    let observed_arbiter = arbiter.clone();
    let observed_key = key.clone();
    let observed_flag = flag.clone();
    AFTER_FIRST_OUTPUT.with(|hook| {
        *hook.borrow_mut() = Some(Box::new(move || {
            assert_eq!(
                observed_arbiter.snapshot(&observed_key).unwrap().reserved,
                32 + SCRATCH_BYTES
            );
            observed_flag.store(true, Ordering::Release);
        }))
    });
    assert!(
        execute(
            request(source, &a, &b),
            BulkF32Admission {
                arbiter: &arbiter,
                resource: &key,
                owner: &owner
            },
            Some(&flag)
        )
        .is_err()
    );
    let final_state = arbiter.snapshot(&key).unwrap();
    assert_eq!(final_state.reserved, 0);
    assert!(!final_state.fenced);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn nonfinite_used_words_and_finite_input_overflow_refuse_without_outputs() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let (arbiter, key, owner) = budget(4096);
    let mut a = [1.0; 32];
    let b = [2.0; 16];
    a[19] = f32::INFINITY;
    assert!(
        execute(
            request(source, &a, &b),
            BulkF32Admission {
                arbiter: &arbiter,
                resource: &key,
                owner: &owner
            },
            None
        )
        .is_err()
    );
    a.fill(f32::MAX);
    assert!(
        execute(
            request(source, &a, &b),
            BulkF32Admission {
                arbiter: &arbiter,
                resource: &key,
                owner: &owner
            },
            None
        )
        .is_err()
    );
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
}

#[test]
#[cfg(not(target_arch = "x86_64"))]
fn unsupported_float_environment_refuses_without_admission() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let (arbiter, key, owner) = budget(4096);
    assert!(matches!(
        execute(
            request(source, &[1.0; 32], &[1.0; 16]),
            BulkF32Admission {
                arbiter: &arbiter,
                resource: &key,
                owner: &owner
            },
            None,
        ),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
}
