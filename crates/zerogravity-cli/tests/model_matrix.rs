//! The source-neutral model roadmap must work in default CPU-only builds.
use std::process::Command;
fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_zerogravity"))
        .args(args)
        .env("HF_TOKEN", "DO_NOT_ECHO_SECRET")
        .env("GITHUB_TOKEN", "DO_NOT_ECHO_SECRET")
        .output()
        .unwrap()
}
#[test]
fn ordered_family_scaffolds_are_offline_and_truthful_in_default_and_optional_builds() {
    let output = run(&["models", "list"]);
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    let qwen = text.find("qwen3.5-0.8b").unwrap();
    let gemma = text.find("gemma4-e2b-it").unwrap();
    let deepseek = text.find("deepseek-v4.1-flash").unwrap();
    let glm = text.find("glm5.3-flash").unwrap();
    let muse = text.find("meta-muse-spark-open").unwrap();
    let mistral = text.find("mistral").unwrap();
    let llama = text.find("llama").unwrap();
    assert!(
        qwen < gemma
            && gemma < deepseek
            && deepseek < glm
            && glm < muse
            && muse < mistral
            && mistral < llama
    );
    for slug in [
        "qwen-image-2.1",
        "qwen3-tts-1.7b",
        "qwen3-asr-1.7b",
        "whisper-large-v3-turbo",
        "nvidia-canary-qwen-2.5b",
        "nvidia-parakeet-tdt-1.1b",
    ] {
        assert!(text.contains(slug));
    }
    assert!(text.contains("NOT IMPLEMENTED"));
    assert!(!text.contains("DO_NOT_ECHO_SECRET"));
}
#[test]
fn every_unqualified_target_has_an_explicit_nonexecuting_failure() {
    for name in [
        "qwen3.8-flash-next",
        "gemma4-31b-it",
        "qwen-image-2.1",
        "nvidia-canary-qwen-2.5b",
        "deepseek-v4.1-flash",
        "glm5.3-flash",
        "llama",
    ] {
        let output = run(&["models", "inspect", name]);
        assert!(output.status.success(), "{name}");
        let text = String::from_utf8(output.stdout).unwrap();
        assert!(text.contains("NOT IMPLEMENTED"), "{name}");
        if matches!(name, "glm5.3-flash" | "deepseek-v4.1-flash") {
            assert!(text.contains("YES - PIPELINE ONLY"));
        }
        for operation in ["run", "stream"] {
            let output = run(&["models", operation, name]);
            assert_eq!(output.status.code(), Some(2), "{operation} {name}");
            let diagnostic = String::from_utf8_lossy(&output.stderr);
            assert!(diagnostic.contains("NOT IMPLEMENTED"));
            assert!(!diagnostic.contains("DO_NOT_ECHO_SECRET"));
        }
    }
    assert_eq!(run(&["models", "inspect", "../bad"]).status.code(), Some(2));
}
