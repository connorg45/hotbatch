use anyhow::{anyhow, Context, Result};
use clap::Args;
use futures_util::StreamExt;
use hotbatch_server::{ServeArgs, ServeMode};
use plotters::prelude::*;
use reqwest::Client;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use tokio::sync::Barrier;

#[derive(Debug, Clone, Args)]
pub struct BenchArgs {
    #[arg(long, default_value = "bench")]
    pub output_dir: PathBuf,
    #[arg(long, default_value = "gpt2")]
    pub model: String,
    #[arg(long, default_value_t = false)]
    pub smoke: bool,
}

#[derive(Debug, Clone)]
struct RunResult {
    mode: ServeMode,
    concurrency: usize,
    tokens_per_sec: f64,
    first_p50_ms: f64,
    first_p95_ms: f64,
    inter_p50_ms: f64,
    inter_p95_ms: f64,
}

#[derive(Debug, Default)]
struct RequestStats {
    tokens: usize,
    first_token_ms: Option<f64>,
    inter_token_ms: Vec<f64>,
}

pub async fn run(args: BenchArgs) -> Result<()> {
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("creating {}", args.output_dir.display()))?;
    let smoke = args.smoke || std::env::var("HOTBATCH_BENCH_SMOKE").is_ok();
    let concurrencies = if smoke {
        vec![1, 2]
    } else {
        vec![1, 2, 4, 8, 16, 32]
    };
    let max_tokens = if smoke { 8 } else { 64 };
    let prompt = "The capital of France is a useful benchmark prompt because";

    let mut results = Vec::new();
    for mode in [ServeMode::Naive, ServeMode::Continuous] {
        let server = hotbatch_server::spawn_server(ServeArgs {
            host: "127.0.0.1".to_string(),
            port: 0,
            mode,
            model: args.model.clone(),
            device: "cpu".to_string(),
            max_running_seqs: 32,
            max_queue_depth: 2048,
            max_seq_len: 512,
            max_new_tokens: max_tokens,
            forward_base_us: 1_200,
            forward_per_seq_us: 130,
        })
        .await?;
        let base_url = format!("http://{}", server.addr);
        warmup(&base_url, prompt, max_tokens).await?;
        for concurrency in &concurrencies {
            let result = run_concurrency(&base_url, mode, *concurrency, prompt, max_tokens).await?;
            println!(
                "{:?}\tconcurrency={}\ttok/s={:.2}\tft p50/p95={:.2}/{:.2}ms\tit p50/p95={:.2}/{:.2}ms",
                result.mode,
                result.concurrency,
                result.tokens_per_sec,
                result.first_p50_ms,
                result.first_p95_ms,
                result.inter_p50_ms,
                result.inter_p95_ms
            );
            results.push(result);
        }
        server.stop().await?;
    }

    write_markdown(&args.output_dir, prompt, max_tokens, &results)?;
    write_plot(&args.output_dir, &results)?;
    Ok(())
}

async fn warmup(base_url: &str, prompt: &str, max_tokens: usize) -> Result<()> {
    let client = Client::new();
    let _stats = one_streaming_request(&client, base_url, prompt, max_tokens, 7).await?;
    Ok(())
}

async fn run_concurrency(
    base_url: &str,
    mode: ServeMode,
    concurrency: usize,
    prompt: &str,
    max_tokens: usize,
) -> Result<RunResult> {
    let client = Client::new();
    let barrier = std::sync::Arc::new(Barrier::new(concurrency + 1));
    let mut joins = Vec::with_capacity(concurrency);
    for index in 0..concurrency {
        let barrier = barrier.clone();
        let client = client.clone();
        let base_url = base_url.to_string();
        let prompt = prompt.to_string();
        joins.push(tokio::spawn(async move {
            barrier.wait().await;
            one_streaming_request(&client, &base_url, &prompt, max_tokens, index as u64).await
        }));
    }
    let start = Instant::now();
    barrier.wait().await;
    let mut request_stats = Vec::with_capacity(concurrency);
    for join in joins {
        request_stats.push(join.await.context("bench task join failed")??);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let total_tokens: usize = request_stats.iter().map(|stats| stats.tokens).sum();
    let first_latencies: Vec<f64> = request_stats
        .iter()
        .filter_map(|stats| stats.first_token_ms)
        .collect();
    let inter_latencies: Vec<f64> = request_stats
        .iter()
        .flat_map(|stats| stats.inter_token_ms.iter().copied())
        .collect();

    Ok(RunResult {
        mode,
        concurrency,
        tokens_per_sec: if elapsed > 0.0 {
            total_tokens as f64 / elapsed
        } else {
            0.0
        },
        first_p50_ms: percentile(first_latencies.clone(), 0.50),
        first_p95_ms: percentile(first_latencies, 0.95),
        inter_p50_ms: percentile(inter_latencies.clone(), 0.50),
        inter_p95_ms: percentile(inter_latencies, 0.95),
    })
}

async fn one_streaming_request(
    client: &Client,
    base_url: &str,
    prompt: &str,
    max_tokens: usize,
    seed: u64,
) -> Result<RequestStats> {
    let url = format!("{base_url}/v1/completions");
    let start = Instant::now();
    let response = client
        .post(url)
        .json(&json!({
            "model": "gpt2",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": true,
            "temperature": 0,
            "seed": seed
        }))
        .send()
        .await
        .context("sending streaming bench request")?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(anyhow!("bench request failed: {status} {body}"));
    }

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut stats = RequestStats::default();
    let mut last_token_at: Option<Instant> = None;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("reading SSE chunk")?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(frame_end) = buffer.find("\n\n") {
            let frame = buffer[..frame_end].to_string();
            buffer = buffer[frame_end + 2..].to_string();
            for line in frame.lines() {
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();
                if data == "[DONE]" {
                    return Ok(stats);
                }
                let value: serde_json::Value =
                    serde_json::from_str(data).context("parsing SSE JSON")?;
                let Some(choice) = value.get("choices").and_then(|choices| choices.get(0)) else {
                    continue;
                };
                if !choice
                    .get("finish_reason")
                    .map(|reason| reason.is_null())
                    .unwrap_or(false)
                {
                    continue;
                }
                let text = choice
                    .get("text")
                    .and_then(|text| text.as_str())
                    .or_else(|| {
                        choice
                            .get("delta")
                            .and_then(|delta| delta.get("content"))
                            .and_then(|content| content.as_str())
                    })
                    .unwrap_or("");
                if text.is_empty() {
                    continue;
                }
                let now = Instant::now();
                if stats.first_token_ms.is_none() {
                    stats.first_token_ms =
                        Some(now.saturating_duration_since(start).as_secs_f64() * 1000.0);
                }
                if let Some(last) = last_token_at {
                    stats
                        .inter_token_ms
                        .push(now.saturating_duration_since(last).as_secs_f64() * 1000.0);
                }
                last_token_at = Some(now);
                stats.tokens += 1;
            }
        }
    }
    Ok(stats)
}

fn percentile(mut values: Vec<f64>, p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let index = ((values.len().saturating_sub(1)) as f64 * p.clamp(0.0, 1.0)).round() as usize;
    values[index]
}

fn write_markdown(
    output_dir: &Path,
    prompt: &str,
    max_tokens: usize,
    results: &[RunResult],
) -> Result<()> {
    let mut markdown = String::new();
    markdown.push_str("# Hotbatch Benchmark Results\n\n");
    markdown.push_str(&format!("Hardware: {}\n\n", hardware_spec()));
    markdown.push_str(&format!(
        "Prompt: `{}`\n\nMax tokens: `{}`. Warm-up request excluded. Each measured client uses `stream=true` and records first-token and inter-token latency from SSE frame arrival times.\n\n",
        prompt, max_tokens
    ));
    markdown.push_str("| mode | concurrency | agg tok/s | first p50 ms | first p95 ms | inter p50 ms | inter p95 ms |\n");
    markdown.push_str("|---|---:|---:|---:|---:|---:|---:|\n");
    for result in results {
        markdown.push_str(&format!(
            "| {:?} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |\n",
            result.mode,
            result.concurrency,
            result.tokens_per_sec,
            result.first_p50_ms,
            result.first_p95_ms,
            result.inter_p50_ms,
            result.inter_p95_ms
        ));
    }
    fs::write(output_dir.join("results.md"), markdown).context("writing benchmark markdown")
}

fn write_plot(output_dir: &Path, results: &[RunResult]) -> Result<()> {
    let path = output_dir.join("results.png");
    let path_string = path.to_string_lossy().to_string();
    let root = BitMapBackend::new(&path_string, (960, 540)).into_drawing_area();
    root.fill(&WHITE)?;
    let max_x = results
        .iter()
        .map(|result| result.concurrency)
        .max()
        .unwrap_or(1) as i32;
    let max_y = results
        .iter()
        .map(|result| result.tokens_per_sec)
        .fold(1.0_f64, f64::max)
        * 1.15;
    let mut chart = ChartBuilder::on(&root)
        .caption("Hotbatch throughput by concurrency", ("sans-serif", 28))
        .margin(24)
        .x_label_area_size(42)
        .y_label_area_size(58)
        .build_cartesian_2d(1_i32..max_x, 0_f64..max_y)?;
    chart
        .configure_mesh()
        .x_desc("concurrent clients")
        .y_desc("aggregate tokens/sec")
        .draw()?;

    for (mode, color) in [(ServeMode::Naive, &RED), (ServeMode::Continuous, &BLUE)] {
        let series: Vec<(i32, f64)> = results
            .iter()
            .filter(|result| result.mode == mode)
            .map(|result| (result.concurrency as i32, result.tokens_per_sec))
            .collect();
        chart
            .draw_series(LineSeries::new(series.clone(), color))?
            .label(format!("{mode:?}"))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], color));
        chart.draw_series(
            series
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 4, color.filled())),
        )?;
    }
    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.9))
        .draw()?;
    root.present()?;
    Ok(())
}

fn hardware_spec() -> String {
    let cpu = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| format!("{} {}", std::env::consts::OS, std::env::consts::ARCH));
    let mem = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|value| value.trim().parse::<f64>().ok())
        .map(|bytes| format!("{:.1} GiB RAM", bytes / 1024.0 / 1024.0 / 1024.0))
        .unwrap_or_else(|| "RAM unknown".to_string());
    format!("{cpu}, {mem}")
}
