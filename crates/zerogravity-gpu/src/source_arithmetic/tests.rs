use std::{path::PathBuf, process::Command, time::SystemTime};

fn plain(source: &str) -> String {
    source
        .split("#[cfg(")
        .next()
        .unwrap()
        .lines()
        .filter(|line| {
            !line.starts_with("//!")
                && !line.starts_with("use crate::source_")
                && *line != "use cubecl::prelude::*;"
                && *line != "#[cube]"
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn actual_attention_arithmetic_preserves_bits_with_and_without_denormals() {
    // Compile actual scalar bodies, not a second implementation of the fast path.
    let support = r#"
#![allow(dead_code)]
trait Cast<T> { fn cast_from(x:T)->Self; }
trait Reinterpret<T> { fn reinterpret(x:T)->Self; }
macro_rules! cast { ($a:ty,$b:ty) => { impl Cast<$a> for $b { fn cast_from(x:$a)->Self {x as Self} } }; }
cast!(u32,u64);cast!(u64,u32);cast!(u32,i32);cast!(i32,i32);cast!(i32,u64);cast!(i32,u32);
cast!(u32,f64);cast!(f32,f64);cast!(f64,f32);cast!(f64,u32);
impl Reinterpret<f32> for u32 {fn reinterpret(x:f32)->Self {x.to_bits()}}
impl Reinterpret<u32> for f32 {fn reinterpret(x:u32)->Self {Self::from_bits(x)}}
const FLUSH:bool=false;
fn physical(x:f32)->f32 {if FLUSH&&x.is_subnormal() {f32::from_bits(x.to_bits()&0x80000000)} else {x}}
fn fma(a:f32,b:f32,c:f32)->f32 {physical(physical(a).mul_add(physical(b),physical(c)))}
"#;
    let checks = r#"
fn check(a:f32,b:f32,c:f32,counts:&mut [usize;2]) {
 let p=fma(a,b,0.0);let r=fma(a,b,c);
 let fast=source_bf16(a)&&source_bf16(b)&&normal(a)&&normal(b)&&normal_or_zero(c)&&normal(p)&&normal(r);
 counts[usize::from(!fast)]+=1;
 if fast {assert_eq!(f64::from(p),f64::from(a)*f64::from(b),"normal BF16 product must be exact")}
 let actual=fused_bf16(a,b,c);
 let expected=if a.is_finite()&&b.is_finite()&&c.is_finite() {a.mul_add(b,c)} else {fused(a,b,c)};
 assert_eq!(actual.to_bits(),expected.to_bits(),"fma a={:08x} b={:08x} c={:08x}",a.to_bits(),b.to_bits(),c.to_bits());
 let expected=if a.is_finite()&&b.is_finite() {a*b} else {mul(a,b)};
 assert_eq!(mul_bf16(a,b).to_bits(),expected.to_bits(),"mul a={:08x} b={:08x}",a.to_bits(),b.to_bits());
 let expected=if a.is_finite()&&c.is_finite() {a+c} else {add(a,c)};
 assert_eq!(add_normal(a,c).to_bits(),expected.to_bits(),"add a={:08x} c={:08x}",a.to_bits(),c.to_bits());
 let expected=add(c,mul(a,b));
 assert_eq!(mul_then_add_normal(a,b,c).to_bits(),expected.to_bits(),"separate mul/add a={:08x} b={:08x} c={:08x}",a.to_bits(),b.to_bits(),c.to_bits());
}
fn next(state:&mut u32)->u32 {
 *state^=*state<<13;*state^=*state>>17;*state^=*state<<5;*state
}
fn finite_bits(mut bits:u32)->u32 {
 if bits&0x7f800000==0x7f800000 {bits^=0x00800000} bits
}
fn main() {
 let mut counts=[0;2];
 for raw in 0..=u16::MAX {
  let a=f32::from_bits(u32::from(raw)<<16);
  if !a.is_finite() {continue}
  for b in [0u32,1,0x007f,0x0080,0x3f00,0x3f80,0x3f81,0x4000,0x5f80,0x7f7f] {
   for sign in [0,0x80000000] {
    let b=f32::from_bits((b<<16)^sign);let p=mul(a,b);
    for c in [0.0,-0.0,f32::from_bits(1),-f32::from_bits(1),f32::MIN_POSITIVE,-f32::MIN_POSITIVE,f32::MAX,-f32::MAX,p,-p,(-p).next_up(),(-p).next_down()] {
     if c.is_finite() {check(a,b,c,&mut counts)}
    }
    if p.is_normal() {
     let exponent=((p.to_bits()>>23)&255) as i32-127;
     let half=2.0f32.powi(exponent-24);
     for c in [half,half.next_up(),half.next_down(),-half,-half.next_up(),-half.next_down()] {
      check(a,b,c,&mut counts);
     }
    }
   }
  }
 }
 let mut state=0x1f27a5b9;
 for index in 0..2_000_000 {
  let mask=if index%2==0 {0xffff0000} else {u32::MAX};
  let a=f32::from_bits(finite_bits(next(&mut state))&mask);
  let b=f32::from_bits(finite_bits(next(&mut state))&mask);
  let c=f32::from_bits(finite_bits(next(&mut state)));
  check(a,b,c,&mut counts);
 }
 // Invalid operands retain the general implementation, including NaN handling.
 for a in [0u32,0x80000000,0x3f800001,0x7f800000,0xff800000,0x7fc01234,0x7f800001] {
  for b in [0u32,0x80000000,0x3f800001,0x7f800000,0xff800000] {
   for c in [0u32,0x80000000,0x3f800001,0x7fc05678,0x7f800000,0xff800000] {
    check(f32::from_bits(a),f32::from_bits(b),f32::from_bits(c),&mut counts);
   }
  }
 }
 assert!(counts[0]>1_000_000&&counts[1]>1_000_000);
 // Preserve actual per-lane QK accumulation/reduction and four-way PV order.
 for length in [1usize,2,3,4,17,64,129,257,1025] {
  let mut q=[0.0f32;64];let mut reference=q;
  let mut pv=[0.0f32;4];let mut pv_reference=pv;
  for token in 0..length {
   let lane=token%64;
   let a=f32::from_bits(finite_bits(next(&mut state))&0xffff0000);
   let b=f32::from_bits(finite_bits(next(&mut state))&0xffff0000);
   q[lane]=fused_bf16(a,b,q[lane]);reference[lane]=fused(a,b,reference[lane]);
   assert_eq!(q[lane].to_bits(),reference[lane].to_bits());
   let p=f32::from_bits([0,0x00010000,0x00800000,0x3e800000,0x3f000000,0x3f800000][token%6]);
   let lane=token%4;
   pv[lane]=add_normal(pv[lane],mul_bf16(a,p));
   pv_reference[lane]=add(pv_reference[lane],mul(a,p));
   assert_eq!(pv[lane].to_bits(),pv_reference[lane].to_bits());
  }
  for half in [32,16,8] {
   for lane in 0..half {
    q[lane]=add_normal(q[lane],q[lane+half]);
    reference[lane]=add(reference[lane],reference[lane+half]);
    assert_eq!(q[lane].to_bits(),reference[lane].to_bits());
   }
  }
  let q0=add_normal(q[0],q[4]);let q1=add_normal(q[1],q[5]);
  let q2=add_normal(q[2],q[6]);let q3=add_normal(q[3],q[7]);
  let r0=add(reference[0],reference[4]);let r1=add(reference[1],reference[5]);
  let r2=add(reference[2],reference[6]);let r3=add(reference[3],reference[7]);
  assert_eq!(add_normal(add_normal(q0,q2),add_normal(q1,q3)).to_bits(),add(add(r0,r2),add(r1,r3)).to_bits());
  assert_eq!(add_normal(add_normal(add_normal(pv[0],pv[1]),pv[2]),pv[3]).to_bits(),add(add(add(pv_reference[0],pv_reference[1]),pv_reference[2]),pv_reference[3]).to_bits());
 }
 println!("ATTENTION_ARITHMETIC_BODY flush={} fused_normal={} fused_fallback={} scalar_checks={} mismatches=0",FLUSH,counts[0],counts[1],4*(counts[0]+counts[1]));
}
"#;
    let code = format!(
        "{support}\n{}\n{}\n{checks}",
        plain(include_str!("../source_float.rs")),
        plain(include_str!("../source_arithmetic.rs")),
    );
    let nonce = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("zg-bf16-fused-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let source = root.join("body.rs");
    let binary = root.join("body");
    // Both runs compile the same actual bodies; only the physical FMA shim differs.
    let compiler = std::env::var_os("RUSTC")
        .map(PathBuf::from)
        .unwrap_or_else(|| "rustc".into());
    for flush in [false, true] {
        std::fs::write(
            &source,
            code.replace(
                "const FLUSH:bool=false;",
                &format!("const FLUSH:bool={flush};"),
            ),
        )
        .unwrap();
        let built = Command::new(&compiler)
            .args(["--edition=2024", "-Dwarnings", "-O"])
            .arg(&source)
            .arg("-o")
            .arg(&binary)
            .output()
            .unwrap();
        assert!(
            built.status.success(),
            "{}",
            String::from_utf8_lossy(&built.stderr)
        );
        let ran = Command::new(&binary).output().unwrap();
        assert!(
            ran.status.success(),
            "{}",
            String::from_utf8_lossy(&ran.stderr)
        );
        println!("{}", String::from_utf8_lossy(&ran.stdout));
    }
    std::fs::remove_dir_all(root).unwrap();
}
