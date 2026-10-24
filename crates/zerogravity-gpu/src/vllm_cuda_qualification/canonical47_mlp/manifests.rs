//! Three original47 packets; no serial-splitK clear or old-output reads.
use super::*;
mod packets;

pub(super) fn require_memory_semantics() -> Result<()> {
    // ROOT reviewed exact47 predicates/partition/rings/epilogues and original
    // packet joins. This finite memory permission grants no numerical result.
    Ok(())
}

fn bind(bytes: usize, io: CubinIo) -> CubinBinding {
    CubinBinding {
        bytes: bytes as u64,
        alignment: 16,
        io,
    }
}
fn fix(
    parameter: usize,
    byte_offset: usize,
    binding: usize,
    extent: usize,
    io: CubinIo,
) -> CubinPointerFixup {
    CubinPointerFixup {
        parameter,
        byte_offset,
        binding,
        binding_offset: 0,
        extent: extent as u64,
        alignment: 16,
        io,
    }
}
fn gemm(index: usize) -> Launch {
    let gate = index == 8;
    let image = if gate {
        ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80
    } else {
        ProjectionImage::AmpereBf16Gemm128x64Sm80
    };
    let (w, x, y) = if gate {
        (14_680_064, 96_256, 673_792)
    } else {
        (7_340_032, 336_896, 96_256)
    };
    let mut bindings = vec![
        bind(w, CubinIo::Read),
        bind(x, CubinIo::Read),
        bind(y, CubinIo::Write),
    ];
    let mut roles = vec![Binding::W, Binding::X, Binding::Y];
    let mut fixups = vec![
        fix(0, 0, 0, w, CubinIo::Read),
        fix(0, 8, 1, x, CubinIo::Read),
    ];
    if gate {
        fixups.extend([
            fix(0, 16, 2, y, CubinIo::Write),
            fix(0, 168, 2, y, CubinIo::Write),
        ]);
    } else {
        roles.push(Binding::Workspace);
        bindings.push(bind(WORKSPACE_BYTES, CubinIo::Write));
        fixups.extend([
            fix(0, 16, 3, 192_512, CubinIo::Write),
            fix(0, 120, 3, 192_512, CubinIo::Write),
            fix(0, 168, 2, y, CubinIo::Write),
        ]);
    }
    Launch {
        image,
        roles,
        manifest: CubinKernelManifest {
            entry: if gate {
                "ampere_bf16_s16816gemm_bf16_256x64_ldg8_f2f_tn"
            } else {
                "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn"
            }
            .into(),
            binary_sm: image.binary_sm(),
            grid: if gate { [28, 1, 1] } else { [8, 1, 2] },
            block: [128, 1, 1],
            dynamic_shared_bytes: if gate { 32768 } else { 24576 },
            attributes: CubinLaunchAttributes::None,
            bindings,
            parameters: vec![CubinParameter {
                offset: 0,
                template: if gate {
                    packets::GATE_UP.to_vec()
                } else {
                    packets::DOWN.to_vec()
                },
            }],
            fixups,
        },
    }
}
fn reducer() -> Launch {
    let mut parameters = vec![CubinParameter {
        offset: 0,
        template: packets::REDUCE[..60].to_vec(),
    }];
    parameters.extend((64..224).step_by(8).map(|offset| CubinParameter {
        offset,
        template: packets::REDUCE[offset..offset + 8].to_vec(),
    }));
    Launch { image: ProjectionImage::Bf16SplitKReduceSm86,
        roles: vec![Binding::Y, Binding::Workspace],
        manifest: CubinKernelManifest {
            entry: "_ZN8cublasLt19splitKreduce_kernelILi32ELi16Ei13__nv_bfloat16S1_fS1_Lb0ES1_S1_S1_Lb1ELb0ELb0ELb0EEEvNS_18cublasSplitKParamsIT4_EEPKT2_PKT8_PT7_PT3_PKS3_SG_PKT9_S7_PSH_PvlPS3_PiSM_SM_SG_SG_SG_SG_SG_".into(),
            binary_sm: 86, grid: [32, 3, 1], block: [32, 16, 1],
            dynamic_shared_bytes: 0, attributes: CubinLaunchAttributes::None,
            bindings: vec![bind(96_256, CubinIo::Write), bind(WORKSPACE_BYTES, CubinIo::Read)],
            parameters, fixups: vec![fix(1, 0, 1, 192_512, CubinIo::Read),
                fix(2, 0, 0, 96_256, CubinIo::Write), fix(3, 0, 0, 96_256, CubinIo::Write),
                fix(4, 0, 0, 96_256, CubinIo::Write)],
        },
    }
}
pub(super) fn launches(index: usize) -> Result<Vec<Launch>> {
    match index {
        8 => Ok(vec![gemm(8)]),
        10 => Ok(vec![gemm(10), reducer()]),
        _ => Err(Error::Unsupported(
            "canonical47 MLP learned nodes8/10 only".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn word(raw: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(raw[offset..offset + 4].try_into().unwrap())
    }
    #[test]
    fn real_producer_consumer_extents_join_through_the_separate_activation() {
        let gate = gemm(8).manifest;
        let act = super::super::super::activation_manifest::canonical47_manifest();
        let down = gemm(10).manifest;
        assert_eq!(gate.bindings[2].bytes, act.bindings[0].bytes);
        assert_eq!(act.bindings[1].bytes, down.bindings[1].bytes);
        assert_eq!(gate.bindings[1].bytes, down.bindings[2].bytes);
        assert_ne!(gate.bindings[2].bytes, down.bindings[1].bytes);
        assert!(launches(7).is_err());
        assert!(launches(9).is_err());
        assert!(launches(11).is_err());
    }
    #[test]
    fn down_writes_two_bf16_planes_before_reducer_reads_them() {
        let down = gemm(10).manifest;
        let reduce = reducer().manifest;
        assert_eq!(down.grid[2], 2);
        assert_eq!(word(&packets::DOWN, 128), 1792);
        assert_eq!(word(&packets::DOWN, 112), 2 * 1792);
        assert_eq!(word(&packets::REDUCE, 36), 47 * 1024);
        assert_eq!(word(&packets::REDUCE, 40), 2);
        let produced = down.fixups.iter().find(|f| f.byte_offset == 16).unwrap();
        let consumed = &reduce.fixups[0];
        assert_eq!(produced.extent, 2 * 47 * 1024 * 2);
        assert_eq!(produced.extent, consumed.extent);
        assert_eq!(produced.io, CubinIo::Write);
        assert_eq!(consumed.io, CubinIo::Read);
        assert_eq!(down.bindings[3].bytes, WORKSPACE_BYTES as u64);
        assert_eq!(reduce.bindings[1].bytes, WORKSPACE_BYTES as u64);
        assert_ne!(produced.binding, down.fixups.last().unwrap().binding);
    }
    #[test]
    fn original_parallel_split_has_no_serial_semaphore_or_old_y_read() {
        assert_eq!(word(&packets::DOWN, 132), 0x24);
        assert_eq!(word(&packets::DOWN, 132) & 8, 0);
        assert_eq!(word(&packets::DOWN, 156), 0);
        assert_eq!(word(&packets::REDUCE, 4), 0);
        assert_eq!(word(&packets::REDUCE, 44), 0);
        assert_eq!(word(&packets::REDUCE, 56), 1);
        assert_eq!(packets::REDUCE[60..64], [0; 4]);
        let down = gemm(10).manifest;
        let reducer = reducer().manifest;
        assert_eq!(down.fixups[2].binding, down.fixups[3].binding);
        assert_eq!(reducer.parameters.len(), 21);
        assert_eq!(reducer.parameters[0].template.len(), 60);
        assert_eq!(reducer.parameters[1].offset, 64);
    }
}
