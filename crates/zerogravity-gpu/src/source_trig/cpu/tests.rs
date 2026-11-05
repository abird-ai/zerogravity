use std::{
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn plain(source: &str) -> String {
    source
        .lines()
        .filter(|line| {
            !line.starts_with("//!")
                && !line.starts_with("use crate::source_")
                && *line != "use cubecl::prelude::*;"
                && *line != "#[cube]"
        })
        .collect::<Vec<_>>()
        .join("\n")
        .replace("#[comptime] ", "")
}

#[test]
fn actual_source_body_preserves_original_f32_rounding_and_signed_zero() {
    let body = include_str!("../cpu.rs")
        .split("#[cube(launch_unchecked)]")
        .next()
        .unwrap();
    let tables = include_str!("tables.rs").replace("//!", "//");
    let body = body.replace("mod tables;", &format!("mod tables {{\n{tables}\n}}"));
    let floats = include_str!("../../source_float.rs")
        .split("/// Correct a physically stored quotient")
        .next()
        .unwrap();
    let arithmetic = include_str!("../../source_arithmetic.rs")
        .split("#[cube]\npub(crate) fn div")
        .next()
        .unwrap();
    let support = r#"
#![allow(dead_code)]
trait Cast<T> { fn cast_from(x:T)->Self; }
trait Reinterpret<T> { fn reinterpret(x:T)->Self; }
macro_rules! cast { ($a:ty,$b:ty) => { impl Cast<$a> for $b { fn cast_from(x:$a)->Self {x as Self} } }; }
cast!(u32,u64);cast!(u64,u32);cast!(u32,usize);cast!(u32,f64);cast!(f32,f64);cast!(f64,f32);cast!(f64,u32);
impl Reinterpret<f32> for u32 {fn reinterpret(x:f32)->Self {x.to_bits()}}
impl Reinterpret<f64> for u64 {fn reinterpret(x:f64)->Self {x.to_bits()}}
impl Reinterpret<u32> for f32 {fn reinterpret(x:u32)->Self {Self::from_bits(x)}}
impl Reinterpret<u64> for f64 {fn reinterpret(x:u64)->Self {Self::from_bits(x)}}
fn fma(a:f64,b:f64,c:f64)->f64 {a.mul_add(b,c)}
"#;
    // Independent original CPU words; no input/value dispatch in the primitive.
    let assertions = r#"
fn main() {
 for (input,sine,cosine) in [
  (0x00000000,0x00000000,0x3f800000),
  (0x00000001,0x00000001,0x3f800000),
  (0x3431af44,0x3431af44,0x3f800000),
  (0x3f800000,0x3f576aa4,0x3f0a5141),
  (0x461c4000,0xbe9c797d,0xbf73c074),
  (0x461c4400,0xbf7761c0,0xbe83bb01),
  (0x461ca000,0x3f3b8cba,0xbf2e3e90),
  (0x44e6e635,0xbd7fe06b,0x3f7f7fff),
  (0x467ffc00,0x3eca0fc1,0xbf6b3881),
  (0x46fffe00,0x3e4001b8,0x3f7b759c),
 ] {
  for sign in [0u32,0x80000000] {
   let input=f32::from_bits(input^sign);
   assert_eq!(value(input,&TABLES,false).to_bits(),sine^sign,"sin {input:?}");
   assert_eq!(value(input,&TABLES,true).to_bits(),cosine,"cos {input:?}");
  }
 }
}
"#;
    let code = format!(
        "{support}\n{}\n{}\n{}\n{assertions}",
        plain(floats),
        plain(arithmetic),
        plain(&body)
    );
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("zg-source-trig-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let source = root.join("body.rs");
    let binary = root.join("body");
    std::fs::write(&source, code).unwrap();
    let compiler = std::env::var_os("RUSTC")
        .map(PathBuf::from)
        .unwrap_or_else(|| "rustc".into());
    let built = Command::new(compiler)
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
    std::fs::remove_dir_all(root).unwrap();
}
