//! Actual factory metadata only; ROOT compares it independently with captures.
use super::*;
#[test]
fn partial_down1_fault_is_cold_restricted_and_hits_original_cancel_gate() {
    assert!(require_down1_arm(1, 0, false).is_ok());
    for (rows, launches, initialized) in
        [(64, 0, false), (2, 0, false), (1, 1, false), (1, 0, true)]
    {
        assert!(require_down1_arm(rows, launches, initialized).is_err());
    }
    let cancel = AtomicBool::new(false);
    after_down1_main(0, &cancel).unwrap();
    assert!(!cancel.load(Ordering::Acquire));
    let unknown = after_down1_main(1, &cancel).unwrap_err();
    assert!(
        unknown
            .to_string()
            .contains("unknown after actual Down main")
    );
    assert!(!cancel.load(Ordering::Acquire));
    after_down1_main(2, &cancel).unwrap();
    assert!(not_cancelled(Some(&cancel)).is_err());
    assert!(after_down1_main(3, &cancel).is_err());
}
fn packets(rows: usize) -> Vec<(Option<ProjectionImage>, CubinKernelManifest)> {
    let mut out = super::super::qwen2b_projection::mlp_diagnostic_manifests(rows)
        .unwrap()
        .into_iter()
        .map(|(image, manifest)| (Some(image), manifest))
        .collect::<Vec<_>>();
    let (image, manifest) =
        super::super::post_attention_projection::qwen2b_mlp_diagnostic_manifest(rows).unwrap();
    out.push((Some(image), manifest));
    out.extend(
        manifests::launches(rows, 8)
            .unwrap()
            .into_iter()
            .map(|p| (Some(p.image), p.manifest)),
    );
    out.push((None, manifests::activation(rows).unwrap()));
    out.extend(
        manifests::launches(rows, 10)
            .unwrap()
            .into_iter()
            .map(|p| (Some(p.image), p.manifest)),
    );
    out
}
#[test]
fn actual_finite_factory_union_has_one_shared_set_capacity() {
    for (rows, functions) in [(1, 7), (64, 6)] {
        let ps = packets(rows);
        assert_eq!(ps.len(), 7);
        let images = ps.iter().map(|(i, _)| *i).collect::<BTreeSet<_>>();
        assert_eq!(images.len(), 5);
        let names = ps
            .iter()
            .map(|(i, m)| (*i, m.entry.as_str()))
            .collect::<BTreeSet<_>>();
        assert_eq!(names.len(), functions);
        assert_eq!(
            images.into_iter().flatten().collect::<BTreeSet<_>>(),
            required_images(rows).unwrap()
        );
        assert!(
            ps.iter()
                .all(|(_, m)| m.attributes == CubinLaunchAttributes::None)
        );
    }
}
// Opt-in ROOT metadata read: --exact ... --nocapture. No runtime or operands.
#[test]
#[ignore = "ROOT metadata collector; no native or numerical acceptance"]
fn actual_finite_factory_packet_metadata_for_root_review() {
    for rows in [1, 64] {
        for (index, (image, manifest)) in packets(rows).into_iter().enumerate() {
            println!(
                "2B_MLP_PACKET rows={rows} index={index} image={image:?} manifest={manifest:?}"
            );
        }
    }
}
