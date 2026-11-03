use super::*;
use crate::numerics::torch_cpu::{
    projection::project_bf16_rows,
    rows_reference::{Archive, number},
};

#[test]
#[ignore = "requires independently captured unchanged original GDN layer and cache implementation"]
fn complete_original_layers_use_actual_prior_outputs() {
    let archive = Archive::gdn();
    let mut prior: Option<(Vec<f32>, Vec<f32>)> = None;
    let mut group = None;
    let mut source = SourceIdentity::unbound();
    let mut words = 0;
    let mut calls = 0;
    for call in archive.doc["calls"].as_array().unwrap() {
        let case = number(call, "case");
        let rows = number(call, "rows");
        let actual_group = number(call, "group");
        if group != Some(actual_group) {
            prior = None;
            source = SourceIdentity::for_new_allocation().unwrap();
            group = Some(actual_group);
        }
        assert_eq!(prior.is_none(), number(call, "base") == 0);
        let hidden = archive.values(case, "hidden");
        let project = |name: &str, input: &[f32]| {
            let shape = archive.shape(case, name);
            project_bf16_rows(
                source,
                SourceTensorRef {
                    kind: archive.kind(case, name),
                    shape: &shape,
                    packed_bytes: archive.bytes(case, name),
                },
                input,
                rows,
            )
            .unwrap()
        };
        let qkv = project("parameter.in_proj_qkv.weight", &hidden);
        let gate = project("parameter.in_proj_z.weight", &hidden);
        let alpha = project("parameter.in_proj_a.weight", &hidden);
        let beta = project("parameter.in_proj_b.weight", &hidden);
        words += archive.compare(case, "gate", &gate);
        let conv_shape = archive.shape(case, "parameter.conv1d.weight");
        let a_shape = archive.shape(case, "parameter.A_log");
        let dt_shape = archive.shape(case, "parameter.dt_bias");
        let norm_shape = archive.shape(case, "parameter.norm.weight");
        let parameter = |name, shape| SourceTensorRef {
            kind: archive.kind(case, name),
            shape,
            packed_bytes: archive.bytes(case, name),
        };
        let done = forward(
            Inputs {
                source,
                mode: if call["mode"] == "decode" {
                    SourceCallMode::Decode
                } else {
                    SourceCallMode::Prefill
                },
                rows,
                key_heads: number(call, "key_heads"),
                value_heads: number(call, "value_heads"),
                qkv: &qkv,
                gate: &gate,
                alpha: &alpha,
                beta: &beta,
                previous_conv: prior.as_ref().map(|p| p.0.as_slice()),
                previous_state: prior.as_ref().map(|p| p.1.as_slice()),
            },
            GdnParameters {
                convolution_name: "parameter.conv1d.weight",
                convolution: parameter("parameter.conv1d.weight", &conv_shape),
                a_log: parameter("parameter.A_log", &a_shape),
                dt_bias: parameter("parameter.dt_bias", &dt_shape),
                norm: parameter("parameter.norm.weight", &norm_shape),
            },
            transcendentals::StandaloneExpProfile::MklExHaSse2HaFtzOffNormalV1,
        )
        .unwrap();
        for (name, data) in [
            ("conv_state", &done.conv_state),
            ("state", &done.recurrent_state),
            ("core", &done.core),
            ("gated", &done.gated),
        ] {
            words += archive.compare(case, name, data);
        }
        let output = project("parameter.out_proj.weight", &done.gated);
        words += archive.compare(case, "output", &output);
        prior = Some((done.conv_state, done.recurrent_state));
        calls += 1;
        eprintln!(
            "SOURCE_GDN_LAYER_ROWS_CASE case={case} rows={rows} mode={} exact=true prior=actual",
            call["mode"]
        );
    }
    assert_eq!(calls, 6);
    eprintln!(
        "SOURCE_GDN_LAYER_ROWS_EXACT calls={calls} words={words} trained_weights=false gpu=false"
    );
}
