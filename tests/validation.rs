use hotbatch_server::{ServeArgs, ServeMode};
use std::process::Command;

fn valid_args() -> ServeArgs {
    ServeArgs {
        host: "127.0.0.1".to_string(),
        port: 0,
        mode: ServeMode::Continuous,
        model: "gpt2".to_string(),
        device: "cpu".to_string(),
        max_running_seqs: 8,
        max_queue_depth: 128,
        max_seq_len: 256,
        max_new_tokens: 32,
    }
}

#[test]
fn valid_startup_configuration_and_model_aliases() {
    for model in [
        "gpt2",
        "openai-community/gpt2",
        "tiny-gpt2",
        "tiny-random-gpt2",
        "sshleifer/tiny-gpt2",
        "hf-internal-testing/tiny-random-gpt2",
    ] {
        let mut args = valid_args();
        args.model = model.to_string();
        args.validate()
            .unwrap_or_else(|error| panic!("expected model alias {model:?} to be valid: {error}"));
    }

    let mut ipv6 = valid_args();
    ipv6.host = "::1".to_string();
    ipv6.validate().expect("IPv6 bind hosts should be valid");
}

#[test]
fn invalid_startup_configuration_is_rejected() {
    type MutateArgs = fn(&mut ServeArgs);
    let invalid_cases: Vec<(&str, MutateArgs)> = vec![
        ("max-running-seqs", |args| args.max_running_seqs = 0),
        ("max-queue-depth", |args| args.max_queue_depth = 0),
        ("max-seq-len", |args| args.max_seq_len = 0),
        ("max-new-tokens", |args| args.max_new_tokens = 0),
        ("less than --max-seq-len", |args| {
            args.max_new_tokens = args.max_seq_len
        }),
        ("1024-token", |args| args.max_seq_len = 1_025),
        ("GPT-2 models only", |args| args.model = "scripted".into()),
        ("unsupported model", |args| args.model = "llama".into()),
        ("cpu only", |args| args.device = "cuda".into()),
        ("invalid --host", |args| args.host = "not a host".into()),
    ];

    for (expected, mutate) in invalid_cases {
        let mut args = valid_args();
        mutate(&mut args);
        let error = args.validate().expect_err("configuration should fail");
        assert!(
            error.to_string().contains(expected),
            "expected {expected:?} in error, got {error:#}"
        );
    }
}

#[test]
fn cli_rejects_configuration_before_starting() {
    let output = Command::new(env!("CARGO_BIN_EXE_hotbatch"))
        .args(["serve", "--max-running-seqs", "0"])
        .output()
        .expect("run hotbatch CLI");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--max-running-seqs must be greater than zero"));
}
