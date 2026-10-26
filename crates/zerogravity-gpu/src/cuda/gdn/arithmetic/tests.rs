use super::*;
use crate::cuda::gdn::{GdnBuffer, kernel, kernel_with_typed_outputs};
use cubecl::prelude::KernelMetadata;

#[test]
fn compiler_source_identity_covers_each_actual_options_load_and_lifetime_unit() {
    let units = [
        (
            "context.rs",
            include_str!("../../../../vendor/cubecl-cuda/src/compute/context.rs"),
            "module::load_data(ptx.as_ptr() as *const _)",
        ),
        (
            "context/nvrtc.rs",
            include_str!("../../../../vendor/cubecl-cuda/src/compute/context/nvrtc.rs"),
            "\"-lineinfo\"",
        ),
        (
            "context/compiler_context.rs",
            include_str!("../../../../vendor/cubecl-cuda/src/compute/context/compiler_context.rs"),
            "adopted.store(true",
        ),
        (
            "context/compiler_limits.rs",
            include_str!("../../../../vendor/cubecl-cuda/src/compute/context/compiler_limits.rs"),
            "admitted CUDA cumulative PTX extent exceeded",
        ),
        (
            "owned_command/compiler.rs",
            include_str!("../../../../vendor/cubecl-cuda/src/owned_command/compiler.rs"),
            "server.prepare_owned_kernel(kernel)",
        ),
        (
            "lib.rs",
            include_str!("../../../../vendor/cubecl-cuda/src/lib.rs"),
            "pub fn include_path()",
        ),
        (
            "context/program_lifetime.rs",
            include_str!("../../../../vendor/cubecl-cuda/src/compute/context/program_lifetime.rs"),
            "resource.destroy()",
        ),
    ];
    let source = |changed: Option<usize>| {
        let mut text = String::new();
        for (index, (name, body, control)) in units.iter().enumerate() {
            // Delimiters cannot be confused with bytes inside a source unit.
            assert!(!name.contains('\0') && !body.contains('\0'));
            assert!(body.contains(control));
            text.push('\0');
            text.push_str(name);
            text.push('\0');
            if changed == Some(index) {
                text.push_str(&body.replacen(control, "different actual compiler behavior", 1));
            } else {
                text.push_str(body);
            }
        }
        text
    };
    assert_eq!(COMPILATION_SOURCE.context_source, source(None));
    let actual = GdnFamilyIdentity::original08();
    for index in 0..units.len() {
        let mut changed = actual;
        changed.compile_source.context_source = Box::leak(source(Some(index)).into_boxed_str());
        assert!(
            actual.require_same(&changed).is_err(),
            "source unit {index}"
        );
        assert!(!reference_compile_eq(
            &actual.compile_source,
            &changed.compile_source
        ));
    }
}

#[test]
fn static_info_transport_and_locked_metadata_source_are_code_identity() {
    let source = COMPILATION_SOURCE;
    assert!(
        source
            .cuda_server_source
            .contains("resources.extend(info_const);")
    );
    assert!(
        source
            .cuda_server_source
            .contains("Some(info.data.as_ptr() as *mut c_void)")
    );
    assert!(
        source
            .cuda_server_source
            .contains("info.dynamic_metadata_offset < info.data.len()")
    );
    assert!(
        source
            .metadata_binding_info_source
            .contains("pub dynamic_metadata_offset: usize")
    );
    assert!(
        source
            .metadata_binding_info_source
            .contains("Self::new(data, 0)")
    );
    assert_eq!(
        source.metadata_binding_info_version,
        "cubecl-runtime 0.11.0\0registry+https://github.com/rust-lang/crates.io-index\0sha256:a1c2bdd3bfe667906cd55cc8f0a956a8476124263b6838cc5d7d68d9edacfea0"
    );
    let lock = include_str!("../../../../../../Cargo.lock");
    assert!(lock.contains(
        "name = \"cubecl-runtime\"\nversion = \"0.11.0\"\nsource = \"registry+https://github.com/rust-lang/crates.io-index\"\nchecksum = \"a1c2bdd3bfe667906cd55cc8f0a956a8476124263b6838cc5d7d68d9edacfea0\""
    ));

    let actual = GdnFamilyIdentity::original08();
    for change in 0..3 {
        let mut changed = actual;
        match change {
            0 => changed.compile_source.cuda_server_source = "changed CUDA packet transport",
            1 => changed.compile_source.metadata_binding_info_source = "changed metadata source",
            _ => changed.compile_source.metadata_binding_info_version = "changed metadata pin",
        }
        assert!(actual.require_same(&changed).is_err());
    }
}

#[test]
fn exact_code_compile_and_launch_metadata_are_distinct_from_qualification() {
    let spec = kernel(GdnPhase::Initial64, GdnStage::Solve).unwrap();
    let actual = spec.arithmetic;
    let mut variants = Vec::new();
    let mut other = actual.clone();
    other.original_ptx_sha256 = "different original PTX";
    variants.push(other);
    let mut other = actual.clone();
    other.owned_source = "different owned instructions";
    variants.push(other);
    let mut other = actual.clone();
    other.grid[0] += 1;
    variants.push(other);
    let mut other = actual.clone();
    other.shared += 4;
    variants.push(other);
    let mut other = actual.clone();
    other.threads *= 2;
    variants.push(other);
    let mut other = actual.clone();
    other.family.compile_source.context_source = "different NVRTC options";
    variants.push(other);
    let mut other = actual.clone();
    other.family.compile_source.shared_header_source = "different header interpretation";
    variants.push(other);
    let mut other = actual.clone();
    other.family.compile_source.unchecked = false;
    variants.push(other);
    let mut other = actual.clone();
    other.family.schedule_source = "different zero-Ai/launch order";
    variants.push(other);
    for other in variants {
        assert_ne!(actual, other);
    }
    assert_eq!(actual, actual.clone());
}

#[test]
fn role_offset_extent_access_and_phase_all_participate() {
    let actual = kernel(GdnPhase::Initial64, GdnStage::Solve)
        .unwrap()
        .arithmetic;
    for change in 0..4 {
        let mut other = actual.clone();
        let binding = &mut Arc::make_mut(&mut other.bindings)[0];
        match change {
            0 => binding.role = GdnBuffer::Inverse,
            1 => binding.offset += 4,
            2 => binding.physical_bytes -= 4,
            _ => binding.io = cubecl_runtime::kernel::BufferIOAttr::ReadWrite,
        }
        assert_ne!(actual, other);
    }
    assert!(
        actual
            .require_stage(GdnPhase::ContinuedTwo, GdnStage::Solve)
            .is_err()
    );
    assert!(
        actual
            .require_stage(GdnPhase::Initial64, GdnStage::Recompute)
            .is_err()
    );
}

#[test]
fn arbitrary_source_or_foreign_original_identity_cannot_enter_closed_factory() {
    let phase = GdnPhase::Initial64;
    let stage = GdnStage::Solve;
    let spec = kernel(phase, stage).unwrap();
    let mut asset = assets::select(phase, stage).unwrap();
    asset.ptx_sha256 = "8201e612 not an admitted default solve";
    assert!(StageArithmetic::original08(phase, stage, &asset, &spec.bindings).is_err());
    let mut asset = assets::select(phase, stage).unwrap();
    asset.source = "same dimensions but different arithmetic";
    assert!(StageArithmetic::original08(phase, stage, &asset, &spec.bindings).is_err());
    assert!(kernel(GdnPhase::CachedDecode, GdnStage::Solve).is_err());
    assert!(kernel(GdnPhase::Initial64, GdnStage::Decode).is_err());
}

#[test]
fn old_factories_keep_exact_source_abi_grid_and_cache_identity() {
    for phase in [
        GdnPhase::InitialOne,
        GdnPhase::CachedDecode,
        GdnPhase::Initial64,
        GdnPhase::ContinuedTwo,
    ] {
        for &stage in phase.stages() {
            for spec in [
                kernel(phase, stage).unwrap(),
                kernel_with_typed_outputs(phase, stage).unwrap(),
            ] {
                let asset = assets::select(phase, stage).unwrap();
                let old = crate::cuda::kernel::PinnedKernel::new(
                    asset.entry,
                    Arc::from(asset.source),
                    asset.threads,
                    spec.bindings
                        .iter()
                        .map(|b| (b.physical_bytes, b.io))
                        .collect(),
                )
                .unwrap();
                assert_eq!(spec.kernel.id(), old.id());
                assert_eq!(spec.grid, asset.grid);
                assert_eq!(spec.arithmetic.owned_source, asset.source);
                assert_eq!(&*spec.arithmetic.bindings, &spec.bindings);
                assert_eq!(spec.arithmetic.original_ptx_sha256, asset.ptx_sha256);
            }
        }
    }
}

#[test]
fn same_ptx_or_geometry_does_not_relabel_source_family_or_typed_operands() {
    let phase = GdnPhase::InitialOne;
    let old = kernel(phase, GdnStage::Solve).unwrap();
    let other = crate::cuda::gdn::kernel_qwen2b(phase, GdnStage::Solve, false).unwrap();
    assert_eq!(old.kernel.id(), other.kernel.id());
    assert_ne!(old.arithmetic, other.arithmetic);
    assert!(
        old.arithmetic
            .family
            .require_same(other.arithmetic.family())
            .is_err()
    );
    let old = kernel(GdnPhase::CachedDecode, GdnStage::Conv).unwrap();
    let typed = kernel_with_typed_outputs(GdnPhase::CachedDecode, GdnStage::Conv).unwrap();
    assert_eq!(
        old.arithmetic.original_ptx_sha256,
        typed.arithmetic.original_ptx_sha256
    );
    assert_ne!(old.arithmetic.bindings, typed.arithmetic.bindings);
    assert_ne!(old.arithmetic, typed.arithmetic);
}

#[test]
fn family_is_fixed_code_and_has_no_runtime_call_or_root_issuer() {
    let a = kernel(GdnPhase::InitialOne, GdnStage::Solve)
        .unwrap()
        .arithmetic;
    let b = kernel(GdnPhase::Initial64, GdnStage::Solve)
        .unwrap()
        .arithmetic;
    assert_eq!(a.family, b.family);
    assert_ne!(a, b);
    assert!(
        COMPILATION_SOURCE
            .context_source
            .contains("--gpu-architecture=sm_")
    );
    assert!(COMPILATION_SOURCE.context_source.contains("\"-lineinfo\""));
    assert!(
        COMPILATION_SOURCE
            .kernel_source
            .contains("AddressType::U64")
    );
    assert!(
        COMPILATION_SOURCE
            .kernel_source
            .contains("ExecutionMode::Unchecked")
    );
    assert!(a.owned_source.contains("asm volatile("));
    assert!(format!("{a:?}").len() < 1024);
}

#[cfg(target_os = "linux")]
#[test]
fn explicit_two_b_factory_keeps_actual_code_while_foreign_family_refuses() {
    for phase in [GdnPhase::InitialOne, GdnPhase::Initial64] {
        for &stage in phase.stages() {
            let actual = crate::cuda::gdn::kernel_qwen2b_component(phase, stage).unwrap();
            let asset = assets::select_qwen2b(phase, stage).unwrap();
            let old = crate::cuda::kernel::PinnedKernel::new(
                asset.entry,
                Arc::from(asset.source),
                asset.threads,
                actual
                    .bindings
                    .iter()
                    .map(|b| (b.physical_bytes, b.io))
                    .collect(),
            )
            .unwrap();
            assert_eq!(actual.kernel.id(), old.id());
            assert_eq!(actual.grid, asset.grid);
            assert_eq!(actual.arithmetic.family, GdnFamilyIdentity::original2b());
            assert_eq!(actual.arithmetic.owned_source, asset.source);
        }
    }
    let phase = GdnPhase::Initial64;
    let actual = crate::cuda::gdn::kernel_qwen2b_component(phase, GdnStage::Conv).unwrap();
    let two = assets::select_qwen2b(phase, GdnStage::Conv).unwrap();
    assert!(
        StageArithmetic::for_family(
            GdnArithmeticFamily::Original08,
            phase,
            GdnStage::Conv,
            &two,
            &actual.bindings
        )
        .is_err()
    );
    assert!(
        GdnFamilyIdentity::original08()
            .require_same(&GdnFamilyIdentity::original2b())
            .is_err()
    );
    assert!(
        crate::cuda::gdn::kernel_qwen2b_component(GdnPhase::FixedPrompt, GdnStage::Conv).is_err()
    );
}

// Original derived equality, independent of the immutable-storage shortcut.
fn reference_compile_eq(a: &CompilationSourceContract, b: &CompilationSourceContract) -> bool {
    (
        a.kernel_source,
        a.context_source,
        a.shared_header_source,
        a.cuda_compiler_source,
        a.cuda_server_source,
        a.metadata_binding_info_source,
        a.metadata_binding_info_version,
        a.language,
        a.hardware,
        a.address_bits,
        a.unchecked,
        a.float_semantics,
    ) == (
        b.kernel_source,
        b.context_source,
        b.shared_header_source,
        b.cuda_compiler_source,
        b.cuda_server_source,
        b.metadata_binding_info_source,
        b.metadata_binding_info_version,
        b.language,
        b.hardware,
        b.address_bits,
        b.unchecked,
        b.float_semantics,
    )
}

fn reference_family_eq(a: &GdnFamilyIdentity, b: &GdnFamilyIdentity) -> bool {
    reference_compile_eq(&a.compile_source, &b.compile_source)
        && (
            a.capture_catalogue,
            a.selectors,
            a.schedule_source,
            a.bindings_source,
            a.catalogue,
        ) == (
            b.capture_catalogue,
            b.selectors,
            b.schedule_source,
            b.bindings_source,
            b.catalogue,
        )
}

fn reference_stage_eq(a: &StageArithmetic, b: &StageArithmetic) -> bool {
    reference_family_eq(&a.family, &b.family)
        && (
            a.phase,
            a.stage,
            a.entry,
            a.owned_source,
            a.original_ptx_sha256,
            a.threads,
            a.shared,
            a.grid,
            &a.bindings,
        ) == (
            b.phase,
            b.stage,
            b.entry,
            b.owned_source,
            b.original_ptx_sha256,
            b.threads,
            b.shared,
            b.grid,
            &b.bindings,
        )
}

fn separate_text(text: &str) -> &'static str {
    Box::leak(text.to_owned().into_boxed_str())
}

#[test]
fn immutable_source_shortcuts_preserve_content_and_length_equality() {
    let a = String::from("abcλ");
    let b = a.clone();
    assert!(!std::ptr::eq(a.as_ptr(), b.as_ptr()));
    for left in [a.as_str(), &a[..3], "", "abcμ"] {
        for right in [a.as_str(), b.as_str(), &a[..3], "", "abcμ"] {
            assert_eq!(source_text_eq(left, right), left == right);
        }
    }
    let left = [a.as_str(), "second"];
    let equal = [b.as_str(), "second"];
    let reordered = ["second", b.as_str()];
    let changed = [b.as_str(), "seconD"];
    for right in [
        &left[..],
        &equal[..],
        &reordered[..],
        &changed[..],
        &left[..1],
        &[],
    ] {
        assert_eq!(source_slice_eq(&left, right), left.as_slice() == right);
    }
}

#[test]
fn exact_family_shortcut_matches_every_original_source_and_compiler_field() {
    let mut actual = GdnFamilyIdentity::original08();
    // Small immutable fixtures also exercise distinct slice/string allocations.
    actual.selectors = &["selector A", "selector B"];
    actual.catalogue = &["code A", "code B"];
    let mut equal = actual;
    equal.compile_source.kernel_source = separate_text(actual.compile_source.kernel_source);
    equal.compile_source.context_source = separate_text(actual.compile_source.context_source);
    equal.compile_source.shared_header_source =
        separate_text(actual.compile_source.shared_header_source);
    equal.compile_source.cuda_compiler_source =
        separate_text(actual.compile_source.cuda_compiler_source);
    equal.compile_source.cuda_server_source =
        separate_text(actual.compile_source.cuda_server_source);
    equal.compile_source.metadata_binding_info_source =
        separate_text(actual.compile_source.metadata_binding_info_source);
    equal.compile_source.metadata_binding_info_version =
        separate_text(actual.compile_source.metadata_binding_info_version);
    equal.compile_source.language = separate_text(actual.compile_source.language);
    equal.compile_source.hardware = separate_text(actual.compile_source.hardware);
    equal.schedule_source = separate_text(actual.schedule_source);
    equal.bindings_source = separate_text(actual.bindings_source);
    equal.selectors = Box::leak(
        actual
            .selectors
            .iter()
            .map(|x| separate_text(x))
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );
    equal.catalogue = Box::leak(
        actual
            .catalogue
            .iter()
            .map(|x| separate_text(x))
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );
    assert!(!std::ptr::eq(
        actual.catalogue.as_ptr(),
        equal.catalogue.as_ptr()
    ));
    assert!(reference_family_eq(&actual, &equal));
    actual.require_same(&equal).unwrap();
    for change in 0..18 {
        let mut changed = equal;
        match change {
            0 => changed.compile_source.kernel_source = "different kernel",
            1 => changed.compile_source.context_source = "different options",
            2 => changed.compile_source.shared_header_source = "different header",
            3 => changed.compile_source.cuda_compiler_source = "different compiler",
            4 => changed.compile_source.cuda_server_source = "different static-info transport",
            5 => changed.compile_source.metadata_binding_info_source = "different metadata packet",
            6 => changed.compile_source.metadata_binding_info_version = "different runtime pin",
            7 => changed.compile_source.language = "different language",
            8 => changed.compile_source.hardware = "different hardware",
            9 => changed.compile_source.address_bits = 32,
            10 => changed.compile_source.unchecked = false,
            11 => changed.capture_catalogue = Some("different exact source catalogue"),
            12 => changed.selectors = &["selector B", "selector A"],
            13 => changed.schedule_source = "different order",
            14 => changed.bindings_source = "different bindings",
            15 => changed.catalogue = &["code B", "code A"],
            16 => changed.catalogue = &["code A", "code C"],
            _ => changed.catalogue = &actual.catalogue[..1],
        }
        assert!(!reference_family_eq(&actual, &changed));
        assert!(actual.require_same(&changed).is_err(), "field {change}");
        assert_eq!(actual == changed, reference_family_eq(&actual, &changed));
    }
    for family in [
        GdnFamilyIdentity::original08(),
        GdnFamilyIdentity::original2b(),
    ] {
        assert_eq!(actual == family, reference_family_eq(&actual, &family));
        family.require_same(&family).unwrap();
    }
    #[cfg(target_os = "linux")]
    {
        let a = GdnFamilyIdentity::text(
            crate::cuda::gdn::GdnKernelRecipe::fixture(0, 47, false).unwrap(),
        );
        let b = GdnFamilyIdentity::text(
            crate::cuda::gdn::GdnKernelRecipe::fixture(1, 47, false).unwrap(),
        );
        assert_eq!(a == b, reference_family_eq(&a, &b));
        assert!(a.require_same(&b).is_err());
        a.require_same(&a).unwrap();
    }
}

#[test]
fn exact_stage_shortcut_keeps_source_bytes_abi_order_and_launch_coordinates() {
    let phase = GdnPhase::Initial64;
    let stage = GdnStage::Solve;
    let actual = kernel(phase, stage).unwrap().arithmetic;
    let mut asset = assets::select(phase, stage).unwrap();
    asset.entry = separate_text(asset.entry);
    asset.source = separate_text(asset.source);
    asset.ptx_sha256 = separate_text(asset.ptx_sha256);
    let equal = StageArithmetic::original08(phase, stage, &asset, &actual.bindings).unwrap();
    assert!(!std::ptr::eq(
        actual.owned_source.as_ptr(),
        equal.owned_source.as_ptr()
    ));
    assert!(reference_stage_eq(&actual, &equal));
    assert_eq!(actual, equal);
    let mut bytes = asset.source.as_bytes().to_vec();
    bytes[0] ^= 1;
    asset.source = separate_text(&String::from_utf8(bytes).unwrap());
    assert!(StageArithmetic::original08(phase, stage, &asset, &actual.bindings).is_err());
    assert!(actual.bindings.len() > 1);
    for change in 0..11 {
        let mut changed = equal.clone();
        match change {
            0 => changed.family = GdnFamilyIdentity::original2b(),
            1 => changed.phase = GdnPhase::InitialOne,
            2 => changed.stage = GdnStage::Recompute,
            3 => changed.entry = "different entry",
            4 => changed.owned_source = asset.source,
            5 => changed.original_ptx_sha256 = "different PTX digest",
            6 => changed.threads += 32,
            7 => changed.shared += 4,
            8 => changed.grid[0] += 1,
            9 => Arc::make_mut(&mut changed.bindings).reverse(),
            _ => Arc::make_mut(&mut changed.bindings)[0].offset += 4,
        }
        assert!(!reference_stage_eq(&actual, &changed));
        assert_eq!(actual == changed, reference_stage_eq(&actual, &changed));
        assert_ne!(actual, changed, "field {change}");
    }
}
