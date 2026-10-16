//! Real-source ownership and CPU-free preparation; never launches a device.
use std::path::Path;
use zerogravity_core::source::TensorSource;
use zerogravity_model_qwen35::{LoadedQwenModel, StateSpec};

#[test]
#[ignore = "requires existing pinned local Qwen sources; no CPU backend required"]
fn source_owner_and_prepared_plan_do_not_allocate_cpu_session_state() {
    for name in ["qwen35-0.8b", "qwen35-2b"] {
        let folder =
            Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/models")).join(name);
        let model =
            LoadedQwenModel::open(&folder.join("model.gguf"), &folder.join("config.json")).unwrap();
        assert_eq!(model.initialized_cpu_state_bytes(), 0);
        let source: &dyn TensorSource = model.as_ref();
        let tensor = source.tensor("token_embd.weight").unwrap();
        tensor.validate().unwrap();
        assert_eq!(source.source_identity(), model.source_identity());
        assert!(std::ptr::eq(
            tensor.packed_bytes,
            model.tensor_bytes("token_embd.weight").unwrap()
        ));
        let specs = model.state_specs();
        assert_eq!(specs.len(), model.geometry().layers.len());
        assert_eq!(
            specs
                .iter()
                .filter(|s| matches!(s, StateSpec::Delta { .. }))
                .count(),
            18
        );
        assert_eq!(
            specs
                .iter()
                .filter(|s| matches!(s, StateSpec::Full { .. }))
                .count(),
            6
        );
        // Building the shared schedule is metadata work, not CPU session creation.
        let plan = model.forward_plan(512, true, true).unwrap();
        assert_eq!(plan.source(), model.source_identity());
        assert_eq!(model.initialized_cpu_state_bytes(), 0);
        #[cfg(feature = "cpu")]
        {
            let first = model.new_session().unwrap();
            let bytes = model.initialized_cpu_state_bytes();
            assert!(bytes > 0);
            let second = model.new_session().unwrap();
            assert_eq!(model.initialized_cpu_state_bytes(), bytes);
            assert_eq!(first.state_fingerprint(), second.state_fingerprint());
        }
    }
}

#[cfg(feature = "diagnostic-cpu-checkpoint")]
#[test]
#[ignore = "requires pinned H24 native checkpoints; no CPU backend or GPU required"]
fn portable_native_checkpoint_roundtrip_is_cpu_backend_independent() {
    let workspace = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."));
    fn digest(hex: &str) -> [u8; 32] {
        std::array::from_fn(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap())
    }
    let tokens = digest("ce320a561786ac5ffd2ae4c868fead41784ab8a07dfdb6aca21ff4a69d3b8343");
    for (name, source) in [
        (
            "0.8b",
            "57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf",
        ),
        (
            "2b",
            "91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6",
        ),
    ] {
        let folder = workspace.join(format!("target/models/qwen35-{name}"));
        let model =
            LoadedQwenModel::open(&folder.join("model.gguf"), &folder.join("config.json")).unwrap();
        let source = digest(source);
        let input = workspace.join(format!("target/h24-checkpoints/cuda-qwen35-{name}.bin.gpu"));
        let checkpoint = model.load_checkpoint(&input, source, tokens).unwrap();
        assert_eq!(checkpoint.source_identity(), model.source_identity());
        assert_eq!(checkpoint.position(), 3);
        assert_eq!(checkpoint.domain_values().len(), 42);
        let rebuilt = model
            .checkpoint_from_domains(checkpoint.position(), &checkpoint.domain_values())
            .unwrap();
        let output =
            std::env::temp_dir().join(format!("zg-layout-{}-{name}.state", std::process::id()));
        rebuilt.save(&output, source, tokens).unwrap();
        assert_eq!(
            std::fs::read(&output).unwrap(),
            std::fs::read(&input).unwrap()
        );
        assert_eq!(model.initialized_cpu_state_bytes(), 0);
        std::fs::remove_file(output).unwrap();
    }
}
