//! CPU differential checks of extracted CubeCL bodies; no device/shader proof.
use super::*;
use std::{
    io::Write,
    process::{Command, Stdio},
};
use zerogravity_core::accelerator::SourceIdentity;

fn host_body(source: &str) -> String {
    assert_eq!(
        source.matches("use cubecl::prelude::*;").count(),
        1,
        "helper imports changed; review the closed native substitution seam"
    );
    assert!(
        source.contains("#[cube]\n"),
        "expected scalar CubeCL helper source"
    );
    // Closed substitutions of scalar intrinsics only. Keep the helper bodies,
    // control flow, coefficients and operations byte-for-byte otherwise.
    let mut source = source
        .replace("#[cube]\n", "")
        .replace("use cubecl::prelude::*;", "use crate::native::*;")
        .replace("crate::source_arithmetic::", "super::float::");
    for (from, to) in [
        ("f32::reinterpret", "float32"),
        ("f64::reinterpret", "float64"),
        ("u32::reinterpret", "bits32"),
        ("u64::reinterpret", "bits64"),
        ("i32::reinterpret", "signed32"),
        ("i64::reinterpret", "signed64"),
        ("f32::cast_from", "cast_f32"),
        ("f64::cast_from", "cast_f64"),
        ("u32::cast_from", "cast_u32"),
        ("u64::cast_from", "cast_u64"),
        ("i32::cast_from", "cast_i32"),
        ("i64::cast_from", "cast_i64"),
        ("usize::cast_from", "cast_usize"),
    ] {
        source = source.replace(from, to);
    }
    for unexpected in [
        "cubecl::",
        "#[cube",
        "#[comptime]",
        "comptime!",
        "gpu_asm!",
        "::reinterpret",
        "::cast_from",
        "ABSOLUTE_POS",
        "CUBE_POS",
        "UNIT_POS",
        "Shared::",
    ] {
        assert!(
            !source.contains(unexpected),
            "unreviewed construct {unexpected} in extracted scalar helper"
        );
    }
    source
}
const NATIVE: &str = r#"
mod native {
    pub trait Cast {fn f32(self)->f32;fn f64(self)->f64;fn u32(self)->u32;fn u64(self)->u64;fn i32(self)->i32;fn i64(self)->i64;fn usize(self)->usize;}
    macro_rules! cast {($($t:ty),*)=>{$(impl Cast for $t {fn f32(self)->f32{self as f32}fn f64(self)->f64{self as f64}fn u32(self)->u32{self as u32}fn u64(self)->u64{self as u64}fn i32(self)->i32{self as i32}fn i64(self)->i64{self as i64}fn usize(self)->usize{self as usize}})*};}
    cast!(f32,f64,u32,u64,i32,i64,usize);
    pub fn cast_f32<T:Cast>(x:T)->f32{x.f32()}pub fn cast_f64<T:Cast>(x:T)->f64{x.f64()}pub fn cast_u32<T:Cast>(x:T)->u32{x.u32()}pub fn cast_u64<T:Cast>(x:T)->u64{x.u64()}pub fn cast_i32<T:Cast>(x:T)->i32{x.i32()}pub fn cast_i64<T:Cast>(x:T)->i64{x.i64()}pub fn cast_usize<T:Cast>(x:T)->usize{x.usize()}
    pub fn bits32(x:f32)->u32{x.to_bits()}pub fn bits64(x:f64)->u64{x.to_bits()}pub fn float32(x:u32)->f32{f32::from_bits(x)}pub fn float64(x:u64)->f64{f64::from_bits(x)}pub fn signed32(x:u32)->i32{x as i32}pub fn signed64(x:u64)->i64{x as i64}
    pub fn fma(a:f32,b:f32,c:f32)->f32{a.mul_add(b,c)}
}
"#;
const DRIVER: &str = r#"
fn next(s:&mut u64)->u32{*s=s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);(*s>>32)as u32}
fn finite(b:u32)->f32{f32::from_bits(if b&0x7f800000==0x7f800000{b^0x00800000}else{b})}
fn reduce(row:&[f32],maximum:bool)->f32 {
    let combine=|a:f32,b:f32|if maximum{if a>b{a}else{b}}else{float::add(a,b)};
    if row.len()<8{return row[1..].iter().fold(row[0],|a,&b|combine(a,b));}
    let mut lanes:[f32;8]=row[..8].try_into().unwrap();for i in 8..row.len(){lanes[i%8]=combine(lanes[i%8],row[i]);}
    combine(combine(combine(lanes[0],lanes[4]),combine(lanes[2],lanes[6])),combine(combine(lanes[1],lanes[5]),combine(lanes[3],lanes[7])))
}
pub fn run(){
    use std::io::Read;let mut bytes=Vec::new();std::io::stdin().read_to_end(&mut bytes).unwrap();let mut words=bytes.chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap()));
    let edge=[0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x00800001,0x3f7fffff,0x3f800000,0x3f800001,0x7f7fffff,0xff7fffff,0x40000000,0xbf800000];
    for &a in &edge{for &b in &edge{let(a,b)=(f32::from_bits(a),f32::from_bits(b));assert_eq!(float::add(a,b).to_bits(),(a+b).to_bits(),"add {a:?} + {b:?}");assert_eq!(float::mul(a,b).to_bits(),(a*b).to_bits(),"mul {a:?} * {b:?}");if b!=0.0{assert_eq!(float::div(a,b).to_bits(),(a/b).to_bits(),"{a:?}/{b:?}");}}}
    for &a in &edge{for &b in &edge{for &c in &edge{let(a,b,c)=(f32::from_bits(a),f32::from_bits(b),f32::from_bits(c));assert_eq!(float::fused(a,b,c).to_bits(),a.mul_add(b,c).to_bits(),"{a:?}*{b:?}+{c:?}");}}}
    let mut seed=0x4d595df4d0f33173;for _ in 0..1_000_000{let a=finite(next(&mut seed));let b=finite(next(&mut seed));let c=finite(next(&mut seed));assert_eq!(float::fused(a,b,c).to_bits(),a.mul_add(b,c).to_bits(),"{a:?}*{b:?}+{c:?}");}
    for bits in 0u32..65536{if bits&0x7f80!=0x7f80 {let x=f32::from_bits(bits<<16);let expected=words.next().unwrap();assert_eq!(transcendentals::gelu(x).to_bits(),expected,"GELU input={:08x}",x.to_bits());}}
    let cases=words.next().unwrap();for sample in 0..cases{let width=words.next().unwrap()as usize;let row=(0..width).map(|_|f32::from_bits(words.next().unwrap())).collect::<Vec<_>>();let expected=(0..width).map(|_|words.next().unwrap()).collect::<Vec<_>>();let maximum=reduce(&row,true);let e=row.iter().map(|&v|{let x=float::sub(v,maximum);if x==f32::NEG_INFINITY{0.0}else{transcendentals::exp(x)}}).collect::<Vec<_>>();let inverse=float::div(1.0,reduce(&e,false));for (actual,expected) in e.iter().map(|&v|float::mul(v,inverse)).zip(expected){assert_eq!(actual.to_bits(),expected,"softmax width{width} case{sample}");}}
    let exps=words.next().unwrap();let table=words.by_ref().take(256).collect::<Vec<_>>();
    for _ in 0..exps{let x=f32::from_bits(words.next().unwrap());let expected=words.next().unwrap();
        assert_eq!(standalone::mkl_ha_normal(x,&table).to_bits(),expected,"explicit standalone exp input={:08x}",x.to_bits());}
    let softplus=words.next().unwrap();for _ in 0..softplus{
        let x=f32::from_bits(words.next().unwrap());let expected=words.next().unwrap();
        assert_eq!(transcendentals::softplus(x).to_bits(),expected,"softplus input={:08x}",x.to_bits());}
    assert!(words.next().is_none());
}
"#;
#[test]
fn actual_helper_bodies_match_ieee_fma_and_original_cpu_gelu_softmax() {
    // CubeCL's ordinary Cast/Reinterpret/fma bodies intentionally panic.
    // Extract the unchanged helper bodies, replacing only those scalar
    // intrinsics with native IEEE casts/bitcasts. GPU expansion is a separate gate.
    let code = format!(
        "{NATIVE}\nmod source_float{{\n{}\n}}\nmod math{{pub mod float{{\n{}\n}}pub mod transcendentals{{\n{}\n}}pub mod standalone{{\n{}\n}}\n{DRIVER}\n}}\nfn main(){{math::run();}}",
        host_body(include_str!("../../../source_float.rs")),
        host_body(include_str!("../../../source_arithmetic.rs")),
        host_body(include_str!("../../../source_sleef.rs")),
        host_body(
            include_str!("../../../source_standalone_exp.rs")
                .split("#[cube(launch_unchecked)]")
                .next()
                .unwrap()
        )
        .replace("use crate::program_kernels::NONFINITE_VALUE;", "")
    );
    let dir = std::env::temp_dir().join(format!(
        "zerogravity-gemma-math-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&dir).unwrap();
    let src = dir.join("math.rs");
    let bin = dir.join("math");
    std::fs::write(&src, code).unwrap();
    let compiled = Command::new("rustc")
        .args(["--edition=2024", "-O", "--cap-lints=allow"])
        .arg(&src)
        .arg("-o")
        .arg(&bin)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let source = SourceIdentity::for_new_allocation().unwrap();
    let input = (0u32..65536)
        .filter(|v| v & 0x7f80 != 0x7f80)
        .map(|v| f32::from_bits(v << 16))
        .collect::<Vec<_>>();
    let gelu =
        zerogravity_cpu::original_hf::transcendentals::gelu_tanh_bf16_widened_f32_contiguous16(
            source, &input,
        )
        .unwrap();
    let mut data = gelu
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    data.extend_from_slice(&(64u32 * 32).to_le_bytes());
    let mut seed = 93u64;
    for width in 1..=64 {
        for sample in 0..32 {
            let mut row = (0..width)
                .map(|i| {
                    if i < sample % width {
                        f32::NEG_INFINITY
                    } else {
                        seed = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        (((seed >> 32) as u32 % 26000) as f32 - 13000.0) / 128.0
                    }
                })
                .collect::<Vec<_>>();
            if sample == 0 {
                row.fill(-0.0);
                row[0] = 0.0;
            }
            if sample == 1 {
                row.fill(f32::from_bits(0x00010000));
                row[0] = f32::from_bits(0x80010000);
            }
            let expected = zerogravity_cpu::original_hf::transcendentals::softmax_f32_lastdim_avx2(
                source, &row, width,
            )
            .unwrap();
            data.extend_from_slice(&(width as u32).to_le_bytes());
            for value in row.iter().chain(&expected) {
                data.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
    }
    let mut exp_input = (0..65536)
        .map(|i| ((i as f32 / 65535.0) * 174.0) - 87.0)
        .collect::<Vec<_>>();
    exp_input[..6].copy_from_slice(&[
        0.0,
        -0.0,
        -87.0,
        87.0,
        f32::from_bits(0xbcecdf1d),
        f32::from_bits(0xbf2461c6),
    ]);
    let exp_expected=zerogravity_cpu::original_hf::transcendentals::exp_f32_contiguous16_profiled(
        source,&exp_input,zerogravity_core::accelerator::program::StandaloneExpProfile::MklExHaSse2HaFtzOffNormalV1).unwrap();
    data.extend_from_slice(&(exp_input.len() as u32).to_le_bytes());
    for pair in zerogravity_core::accelerator::program::standalone_exp::MKL_EXHA_SSE2_EXP2_128 {
        for word in pair {
            data.extend_from_slice(&word.to_le_bytes());
        }
    }
    for (x, y) in exp_input.iter().zip(exp_expected) {
        data.extend_from_slice(&x.to_le_bytes());
        data.extend_from_slice(&y.to_le_bytes());
    }
    let softplus =
        zerogravity_cpu::original_hf::transcendentals::softplus_beta1_threshold20_f32_contiguous16(
            source, &input,
        )
        .unwrap();
    data.extend_from_slice(&(input.len() as u32).to_le_bytes());
    for (x, y) in input.iter().zip(softplus) {
        data.extend_from_slice(&x.to_le_bytes());
        data.extend_from_slice(&y.to_le_bytes());
    }
    let mut child = Command::new(&bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&data).unwrap();
    let result = child.wait_with_output().unwrap();
    assert!(
        result.status.success(),
        "{} (extracted source {})",
        String::from_utf8_lossy(&result.stderr),
        src.display()
    );
    std::fs::remove_file(src).unwrap();
    std::fs::remove_file(bin).unwrap();
    std::fs::remove_dir(dir).unwrap();
}
#[test]
fn scratch_and_tables_price_source_geometry_only() {
    assert_eq!(scratch_layout().len(), 14);
    assert_eq!(scratch_layout().iter().sum::<usize>(), 113436);
    let tables = immutable_tables();
    assert_eq!(
        tables.iter().map(Vec::len).collect::<Vec<_>>(),
        [512, 1024, 280]
    );
    assert!(tables[1][64 * 4..].iter().all(|b| *b == 0));
}
