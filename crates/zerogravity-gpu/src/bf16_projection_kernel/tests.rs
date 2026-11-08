use super::{output_index, rounded_index};
use std::{process::Command, time::SystemTime};

#[test]
fn weight_local_groups_cover_each_output_once() {
    for requests in 1..=64 {
        for rows in [1, 3, 31, 64, 257, 6144] {
            let mut seen = vec![false; requests * rows];
            for group in 0..seen.len() {
                let output = output_index(group, requests, rows);
                assert_eq!(output / rows, group % requests);
                assert_eq!(output % rows, group / requests);
                assert!(!seen[output]);
                seen[output] = true;
            }
            assert!(seen.into_iter().all(|visited| visited));
        }
    }
}

#[test]
fn strided_rounding_maps_unique_words_and_preserves_gaps() {
    let bits = [0u32, 0x80000000, 0x3f800000, 0x80010000, 0x7f7f0000];
    for requests in [1, 2, 3, 17, 47, 62, 64] {
        for stride in [1usize, 3, 31, 64, 257] {
            for rows in [1, stride.div_ceil(2), stride] {
                let span = (requests - 1) * stride + rows;
                let mut actual = vec![0x7fc0beef; span];
                let mut seen = vec![false; span];
                for group in 0..requests * rows {
                    let output = output_index(group, requests, rows);
                    let at = rounded_index(output, rows, stride);
                    assert_eq!(at, (group % requests) * stride + group / requests);
                    assert!(!seen[at]);
                    seen[at] = true;
                    actual[at] = bits[output % bits.len()];
                    if rows == stride {
                        assert_eq!(at, output);
                    }
                }
                for at in 0..span {
                    if at % stride < rows {
                        let output = (at / stride) * rows + at % stride;
                        assert!(seen[at]);
                        assert_eq!(actual[at], bits[output % bits.len()]);
                    } else {
                        assert!(!seen[at]);
                        assert_eq!(actual[at], 0x7fc0beef);
                    }
                }
            }
        }
    }
}

#[test]
fn mapping_preserves_single_rows_and_native_index_bounds() {
    for rows in [1, 127, 248320, u32::MAX as usize / 64] {
        for requests in [1, 2, 3, 4, 5, 17, 47, 62, 63, 64] {
            let count = rows * requests;
            assert!(count <= u32::MAX as usize);
            for group in [0, count / 2, count - 1] {
                let output = output_index(group, requests, rows);
                assert!(output < count);
                if requests == 1 {
                    assert_eq!(output, group);
                }
            }
        }
    }
}

#[test]
fn actual_status_tree_preserves_every_lane_and_flag_bit() {
    // Execute the actual stages between barriers, not a duplicate flag reducer.
    let source = include_str!("../bf16_projection_kernel.rs");
    let stages = source.split_once("flags[lane] = invalid;").unwrap().1;
    let stages = stages.split_once("sync_cube();").unwrap().1;
    let (stages, final_lane) = stages.split_once("if lane == 0usize {").unwrap();
    let stages: Vec<_> = stages
        .split("sync_cube();")
        .filter(|stage| !stage.trim().is_empty())
        .collect();
    assert_eq!(stages.len(), 3);
    let finish = final_lane.split_once("let mut combined = 0u32;").unwrap().1;
    let finish = finish.split_once("if (bits &").unwrap().0;
    let mut program =
        String::from("fn actual(mut flags:[u32;64])->u32 { let mut sums=[0.0f32;64];\n");
    for stage in stages {
        program.push_str(&format!("for lane in 0..64usize {{ {stage} }}\n"));
    }
    program.push_str(&format!("let mut combined=0u32; {finish} combined }}\n"));
    program.push_str(
        r#"
fn check(flags:[u32;64]) {
    assert_eq!(actual(flags),flags.into_iter().fold(0,|a,b|a|b));
}
fn main() {
    check([0;64]);check([u32::MAX;64]);
    for lane in 0..64 {
        for bit in 0..32 {
            let mut flags=[0;64];flags[lane]=1<<bit;check(flags);
        }
        for peer in 0..64 {
            for (a,b) in [(1,2),(2,4),(1<<31,16)] {
                let mut flags=[0;64];flags[lane]|=a;flags[peer]|=b;check(flags);
            }
        }
    }
    let mut seed=0x973521afu32;
    for _ in 0..4096 {
        let mut flags=[0;64];
        for flag in &mut flags {
            seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;*flag=seed;
        }
        check(flags);
    }
}
"#,
    );
    let nonce = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "zg-projection-status-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir(&root).unwrap();
    let src = root.join("actual.rs");
    let exe = root.join("actual");
    std::fs::write(&src, program).unwrap();
    let build = Command::new("rustc")
        .args(["--edition=2024", "-Dwarnings", "-O"])
        .arg(&src)
        .arg("-o")
        .arg(&exe)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{root:?}: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&exe).output().unwrap();
    assert!(
        run.status.success(),
        "{root:?}: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    std::fs::remove_dir_all(root).unwrap();
}
