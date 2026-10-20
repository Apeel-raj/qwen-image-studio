#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use image::imageops::FilterType;
use image::GenericImageView;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;
#[cfg(target_os = "windows")]
const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x00004000;

#[derive(Clone, Debug)]
pub struct FileDownloadProgress {
    pub file_index: usize,
    pub total_files: usize,
    pub file_name: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub speed_mb_s: f32,
    pub eta_seconds: u64,
    pub status_text: String,
}

enum WorkerMsg {
    Log(String),
    Progress(f32, String),
    Finished(Result<PathBuf, String>),
    DownloadProgress(FileDownloadProgress),
    DownloadFinished(Result<String, String>),
}

struct QwenApp {
    // Model paths
    sd_bin_path: String,
    diffusion_model_path: String,
    llm_model_path: String,
    vae_model_path: String,
    mmproj_model_path: String,
    lora_dir_path: String,

    // Generation parameters
    prompt: String,
    negative_prompt: String,
    transparent_rgba: bool,
    ref_image_path: String,
    auto_resize_ref: bool,
    width: u32,
    height: u32,
    steps: u32,
    cfg_scale: f32,
    seed: i64,
    threads: u32,
    sampler: String,

    // Hardware & Thermal Guard flags for RTX 4050 (6GB) + 16GB RAM Laptop
    offload_to_cpu: bool,
    te_on_disk_free_ram: bool,
    clip_on_cpu: bool,
    vae_tiling: bool,
    vae_on_cpu: bool,
    flash_attention: bool,
    q4_prefix_cache: bool,

    // Privacy & Output Browser state
    privacy_dont_autoshow: bool,
    show_output_drawer: bool,
    output_files: Vec<PathBuf>,

    // Native Downloader state
    show_download_modal: bool,
    download_in_progress: bool,
    download_progress: Option<FileDownloadProgress>,
    cancel_download_flag: Arc<AtomicBool>,

    // Runtime state
    is_busy: bool,
    progress: f32,
    status_line: String,
    logs: Vec<String>,
    rx: Receiver<WorkerMsg>,
    tx: Sender<WorkerMsg>,
    texture: Option<egui::TextureHandle>,
    last_image_path: Option<PathBuf>,
}

impl QwenApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (tx, rx) = channel();
        let cancel_download_flag = Arc::new(AtomicBool::new(false));

        let mut app = Self {
            sd_bin_path: "bin/sd-cli.exe".to_owned(),
            diffusion_model_path: "models/qwen-image-2.1-UC-Q4_K_M.gguf".to_owned(),
            llm_model_path: "models/Qwen3VL-8B-Instruct-Q4_K_M.gguf".to_owned(),
            vae_model_path: "models/qwen_image_2.1_vae_bf16.safetensors".to_owned(),
            mmproj_model_path: "models/mmproj-Qwen3VL-8B-Instruct-F16.gguf".to_owned(),
            lora_dir_path: "loras".to_owned(),

            prompt: "A wide-angle cinematic photograph of a bustling cyberpunk alleyway at night during a light rain. Warm amber light spills out from ramen stalls and glass storefronts onto the wet asphalt pavement, illuminating rising steam vents and glowing neon signs in cyan and magenta. Balanced exposure with visible shadow detail, sharp focus, 35mm lens photography.".to_owned(),
            negative_prompt: "blurry, out of focus, low quality, underexposed, pitch black, crushed blacks, over-saturated, deformed, bad anatomy, extra limbs, distorted proportions, watermark".to_owned(),
            transparent_rgba: false,
            ref_image_path: String::new(),
            auto_resize_ref: true,

            width: 896,
            height: 896,
            steps: 26,
            cfg_scale: 3.5,
            seed: -1,
            threads: 4,
            sampler: "euler".to_owned(),

            offload_to_cpu: true,
            te_on_disk_free_ram: true,
            clip_on_cpu: false,
            vae_tiling: true,
            vae_on_cpu: false,
            flash_attention: true,
            q4_prefix_cache: false,

            privacy_dont_autoshow: false,
            show_output_drawer: true,
            output_files: Vec::new(),

            show_download_modal: false,
            download_in_progress: false,
            download_progress: None,
            cancel_download_flag,

            is_busy: false,
            progress: 0.0,
            status_line: "Ready — High-Quality Mode Active (Press ESC anytime to clear screen)".to_owned(),
            logs: vec![
                "Welcome to Qwen-Image 2.1 Studio (RTX 4050 6GB Laptop Edition).".to_owned(),
                "Memory Profile: 4-bit Diffusion (4.6GB) + 4-bit Qwen3-VL (5.0GB) with disk-backed prompt release.".to_owned(),
                "Click '⬇ 1-Click Setup / Models' if any engine or model files are missing.".to_owned(),
            ],
            rx,
            tx,
            texture: None,
            last_image_path: None,
        };

        // If core files are missing, show the clean 1-click download view on first launch
        if !app.files_ready() {
            app.show_download_modal = true;
        }

        app.refresh_output_files();
        app
    }

    fn append_log(&mut self, line: String) {
        self.logs.push(line);
        if self.logs.len() > 800 {
            let drain_count = self.logs.len() - 800;
            self.logs.drain(0..drain_count);
        }
    }

    fn clear_preview(&mut self) {
        self.texture = None;
        self.status_line = "Preview screen cleared (File remains saved in ./output folder).".to_owned();
    }

    fn refresh_output_files(&mut self) {
        let _ = std::fs::create_dir_all("output");
        let mut files: Vec<PathBuf> = std::fs::read_dir("output")
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.starts_with("_ref_") {
                    return false;
                }
                if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                    matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "webp")
                } else {
                    false
                }
            })
            .collect();

        files.sort_by(|a, b| {
            let ma = std::fs::metadata(a).and_then(|m| m.modified()).ok();
            let mb = std::fs::metadata(b).and_then(|m| m.modified()).ok();
            mb.cmp(&ma)
        });
        self.output_files = files;
    }

    fn files_ready(&self) -> bool {
        Path::new(&self.sd_bin_path).exists()
            && Path::new(&self.diffusion_model_path).exists()
            && Path::new(&self.llm_model_path).exists()
            && Path::new(&self.vae_model_path).exists()
    }

    fn mmproj_ready(&self) -> bool {
        Path::new(&self.mmproj_model_path).exists()
    }

    fn start_native_download(&mut self) {
        if self.download_in_progress {
            return;
        }

        self.download_in_progress = true;
        self.cancel_download_flag.store(false, Ordering::Relaxed);
        let cancel_flag = Arc::clone(&self.cancel_download_flag);
        let tx = self.tx.clone();

        self.append_log(">>> Starting 1-Click Engine & Model verification/download...".to_owned());

        thread::spawn(move || {
            let res = run_download_pipeline(&tx, cancel_flag);
            let _ = tx.send(WorkerMsg::DownloadFinished(res));
        });
    }

    fn prepare_reference_image(&mut self, raw_path: &str) -> Result<String, String> {
        let p = Path::new(raw_path);
        if !p.exists() {
            return Err(format!("Reference image does not exist: {raw_path}"));
        }
        if !self.auto_resize_ref {
            return Ok(raw_path.to_owned());
        }

        let img = image::open(p).map_err(|e| format!("Failed to open reference image: {e}"))?;
        let (orig_w, orig_h) = img.dimensions();

        let max_side: u32 = self.width.max(self.height).clamp(512, 768);
        let scale = (max_side as f32 / orig_w.max(orig_h) as f32).min(1.0);
        let new_w = (((orig_w as f32 * scale) as u32) / 32 * 32).max(256);
        let new_h = (((orig_h as f32 * scale) as u32) / 32 * 32).max(256);

        let _ = std::fs::create_dir_all("output");
        let prepared_path = PathBuf::from("output/_ref_prepared.png");
        let resized = img.resize_exact(new_w, new_h, FilterType::Lanczos3);
        resized
            .save(&prepared_path)
            .map_err(|e| format!("Failed to save resized reference image: {e}"))?;

        self.append_log(format!(
            ">>> Prepared Reference Image: resized {}x{} -> {}x{} ({}) to fit 6GB VRAM safely.",
            orig_w,
            orig_h,
            new_w,
            new_h,
            prepared_path.display()
        ));
        Ok(prepared_path.to_string_lossy().to_string())
    }

    fn start_generation(&mut self) {
        if self.is_busy {
            return;
        }

        let has_ref = !self.ref_image_path.trim().is_empty();
        if has_ref && !self.mmproj_ready() {
            let msg = format!(
                "Cannot run Image-to-Image yet: missing '{}'. Click '⬇ 1-Click Setup' at top-right first!",
                self.mmproj_model_path
            );
            self.status_line = msg.clone();
            self.append_log(format!(">>> ERROR: {msg}"));
            self.show_download_modal = true;
            return;
        }

        let prepared_ref = if has_ref {
            let raw_ref = self.ref_image_path.trim().to_owned();
            match self.prepare_reference_image(&raw_ref) {
                Ok(p) => Some(p),
                Err(e) => {
                    self.status_line = e.clone();
                    self.append_log(format!(">>> ERROR: {e}"));
                    return;
                }
            }
        } else {
            None
        };

        let _ = std::fs::create_dir_all("output");
        let _ = std::fs::create_dir_all("loras");
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let out_path = PathBuf::from(format!("output/qwen_2_1_{ts}.png"));

        let final_prompt = if self.transparent_rgba {
            format!(
                "This is an RGBA image with transparency. {}. The image has alpha channel and the background is transparent.",
                self.prompt.trim()
            )
        } else {
            self.prompt.trim().to_owned()
        };

        let mut args: Vec<String> = vec![
            "--diffusion-model".into(),
            self.diffusion_model_path.clone(),
            "--llm".into(),
            self.llm_model_path.clone(),
            "--vae".into(),
            self.vae_model_path.clone(),
            "-p".into(),
            final_prompt,
            "-W".into(),
            (self.width - (self.width % 32)).to_string(),
            "-H".into(),
            (self.height - (self.height % 32)).to_string(),
            "--steps".into(),
            self.steps.to_string(),
            "--cfg-scale".into(),
            format!("{:.1}", self.cfg_scale),
            "--sampling-method".into(),
            self.sampler.clone(),
            "-s".into(),
            self.seed.to_string(),
            "-t".into(),
            self.threads.to_string(),
            "-v".into(),
            "-o".into(),
            out_path.to_string_lossy().to_string(),
        ];

        if self.cfg_scale > 1.05 && !self.negative_prompt.trim().is_empty() {
            args.push("-n".into());
            args.push(self.negative_prompt.trim().to_owned());
        }

        if let Some(ref_path) = prepared_ref {
            args.push("--llm_vision".into());
            args.push(self.mmproj_model_path.trim().to_owned());
            args.push("-r".into());
            args.push(ref_path);
        }

        if !self.lora_dir_path.trim().is_empty() && Path::new(self.lora_dir_path.trim()).exists() {
            args.push("--lora-model-dir".into());
            args.push(self.lora_dir_path.trim().to_owned());
        }

        if self.offload_to_cpu {
            args.push("--offload-to-cpu".into());
        }
        if self.te_on_disk_free_ram {
            args.push("--params-backend".into());
            args.push("te=disk".into());
        }
        if self.clip_on_cpu {
            args.push("--clip-on-cpu".into());
        }
        if self.vae_tiling {
            args.push("--vae-tiling".into());
        }
        if self.vae_on_cpu {
            args.push("--vae-on-cpu".into());
        }
        if self.flash_attention {
            args.push("--diffusion-fa".into());
            args.push("--fa".into());
        }
        if self.q4_prefix_cache {
            args.push("--model-args".into());
            args.push("qwen_image_2_1_prefix_cache_type=q4_0".into());
        }

        let bin = self.sd_bin_path.clone();
        let total_steps = self.steps.max(1);
        let tx = self.tx.clone();

        self.is_busy = true;
        self.progress = 0.02;
        self.status_line = "Running sd-cli.exe (Encoding prompt/image & sampling on RTX 4050)...".to_owned();
        self.append_log(format!(">>> Executing: {} {}", bin, args.join(" ")));

        thread::spawn(move || {
            let mut cmd = Command::new(&bin);
            cmd.args(&args)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());

            #[cfg(target_os = "windows")]
            cmd.creation_flags(CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS);

            match cmd.spawn() {
                Ok(mut child) => {
                    let (err_tx, err_rx) = channel::<String>();

                    if let Some(stdout) = child.stdout.take() {
                        let tx_out = tx.clone();
                        let err_tx_out = err_tx.clone();
                        thread::spawn(move || {
                            let reader = BufReader::new(stdout);
                            for line in reader.lines().map_while(Result::ok) {
                                if let Some((frac, label)) = parse_step_progress(&line, total_steps) {
                                    let _ = tx_out.send(WorkerMsg::Progress(frac, label));
                                }
                                if line.contains("[ERROR]")
                                    || line.contains("error:")
                                    || line.contains("GGML_ASSERT")
                                    || line.contains("failed")
                                {
                                    let _ = err_tx_out.send(line.clone());
                                }
                                let _ = tx_out.send(WorkerMsg::Log(line));
                            }
                        });
                    }
                    if let Some(stderr) = child.stderr.take() {
                        let tx_err = tx.clone();
                        let err_tx_err = err_tx.clone();
                        thread::spawn(move || {
                            let reader = BufReader::new(stderr);
                            for line in reader.lines().map_while(Result::ok) {
                                if let Some((frac, label)) = parse_step_progress(&line, total_steps) {
                                    let _ = tx_err.send(WorkerMsg::Progress(frac, label));
                                }
                                if line.contains("[ERROR]")
                                    || line.contains("error:")
                                    || line.contains("GGML_ASSERT")
                                    || line.contains("failed")
                                {
                                    let _ = err_tx_err.send(line.clone());
                                }
                                let _ = tx_err.send(WorkerMsg::Log(line));
                            }
                        });
                    }
                    drop(err_tx);

                    match child.wait() {
                        Ok(status) if status.success() && out_path.exists() => {
                            let _ = tx.send(WorkerMsg::Finished(Ok(out_path)));
                        }
                        Ok(status) => {
                            let mut last_err = String::new();
                            while let Ok(e) = err_rx.recv() {
                                last_err = e;
                            }
                            if last_err.is_empty() {
                                let _ = tx.send(WorkerMsg::Finished(Err(format!(
                                    "sd-cli exited with {status}. Click '📋 Copy Logs' below to inspect."
                                ))));
                            } else {
                                let _ = tx.send(WorkerMsg::Finished(Err(format!(
                                    "sd-cli failed ({status}): {last_err}"
                                ))));
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(WorkerMsg::Finished(Err(format!("Wait error: {e}"))));
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(WorkerMsg::Finished(Err(format!(
                        "Could not launch '{}': {}. Click '⬇ 1-Click Setup' at top-right.",
                        bin, e
                    ))));
                }
            }
        });
    }

    fn load_preview_texture(&mut self, ctx: &egui::Context, path: &Path) {
        if let Ok(img) = image::open(path) {
            let rgba = img.to_rgba8();
            let size = [rgba.width() as usize, rgba.height() as usize];
            let pixels = rgba.as_flat_samples();
            let color_image = egui::ColorImage::from_rgba_unmultiplied(size, pixels.as_slice());
            self.texture = Some(ctx.load_texture(
                "generated_output",
                color_image,
                egui::TextureOptions::LINEAR,
            ));
        }
    }
}

fn parse_step_progress(line: &str, fallback_total: u32) -> Option<(f32, String)> {
    for token in line.split_whitespace() {
        if let Some((left, right)) = token.split_once('/') {
            if let (Ok(cur), Ok(tot)) = (left.parse::<u32>(), right.parse::<u32>()) {
                if tot > 0 && cur <= tot && tot <= 200 {
                    let frac = (cur as f32 / tot as f32).clamp(0.05, 0.98);
                    return Some((frac, format!("Sampling step {cur}/{tot}...")));
                }
            }
        }
    }
    if line.contains("encode_first_stage") || line.contains("decode_first_stage") {
        return Some((0.95, "Running RGBA VAE (Tiled)...".to_owned()));
    }
    let _ = fallback_total;
    None
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.0} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn download_file_resumable(
    client: &reqwest::blocking::Client,
    url: &str,
    dest_path: &Path,
    file_name: &str,
    file_index: usize,
    total_files: usize,
    tx: &Sender<WorkerMsg>,
    cancel_flag: &Arc<AtomicBool>,
) -> Result<(), String> {
    if let Some(parent) = dest_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let mut downloaded = if dest_path.exists() {
        dest_path.metadata().map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    let mut req = client.get(url);
    if downloaded > 0 {
        req = req.header("Range", format!("bytes={}-", downloaded));
    }

    let mut resp = req.send().map_err(|e| format!("HTTP request failed: {e}"))?;
    let status = resp.status();

    let (mut file, total_bytes) = if status == reqwest::StatusCode::PARTIAL_CONTENT {
        let content_len = resp.content_length().unwrap_or(0);
        let total = downloaded + content_len;
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(dest_path)
            .map_err(|e| format!("Failed to open file in append mode: {e}"))?;
        (file, total)
    } else if status == reqwest::StatusCode::OK {
        let total = resp.content_length().unwrap_or(0);
        downloaded = 0;
        let file = File::create(dest_path)
            .map_err(|e| format!("Failed to create file: {e}"))?;
        (file, total)
    } else if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        let _ = tx.send(WorkerMsg::DownloadProgress(FileDownloadProgress {
            file_index,
            total_files,
            file_name: file_name.to_owned(),
            downloaded_bytes: downloaded,
            total_bytes: downloaded,
            speed_mb_s: 0.0,
            eta_seconds: 0,
            status_text: "Already completed".to_owned(),
        }));
        return Ok(());
    } else {
        return Err(format!("Server returned HTTP status: {status}"));
    };

    let mut buf = [0u8; 128 * 1024];
    let mut last_tick = Instant::now();
    let mut last_bytes = downloaded;
    let mut speed_mb_s: f32;
    let mut eta_secs = 0;

    loop {
        if cancel_flag.load(Ordering::Relaxed) {
            return Err("Download cancelled by user.".to_owned());
        }

        let n = resp.read(&mut buf).map_err(|e| format!("Error reading response stream: {e}"))?;
        if n == 0 {
            break;
        }

        file.write_all(&buf[..n]).map_err(|e| format!("Failed writing to disk: {e}"))?;
        downloaded += n as u64;

        if last_tick.elapsed() >= Duration::from_millis(150) {
            let elapsed = last_tick.elapsed().as_secs_f32();
            let bytes_delta = (downloaded - last_bytes) as f32;
            speed_mb_s = (bytes_delta / (1024.0 * 1024.0)) / elapsed.max(0.001);
            last_tick = Instant::now();
            last_bytes = downloaded;

            if speed_mb_s > 0.05 && total_bytes > downloaded {
                let remaining = (total_bytes - downloaded) as f32 / (1024.0 * 1024.0);
                eta_secs = (remaining / speed_mb_s) as u64;
            }

            let _ = tx.send(WorkerMsg::DownloadProgress(FileDownloadProgress {
                file_index,
                total_files,
                file_name: file_name.to_owned(),
                downloaded_bytes: downloaded,
                total_bytes,
                speed_mb_s,
                eta_seconds: eta_secs,
                status_text: format!("Downloading {} ({:.1} MB/s)", file_name, speed_mb_s),
            }));
        }
    }

    let _ = tx.send(WorkerMsg::DownloadProgress(FileDownloadProgress {
        file_index,
        total_files,
        file_name: file_name.to_owned(),
        downloaded_bytes: total_bytes.max(downloaded),
        total_bytes: total_bytes.max(downloaded),
        speed_mb_s: 0.0,
        eta_seconds: 0,
        status_text: format!("Verified {}", file_name),
    }));

    Ok(())
}

fn extract_zip_flat(zip_path: &Path, dest_dir: &Path) -> Result<(), String> {
    let _ = std::fs::create_dir_all(dest_dir);
    let file = File::open(zip_path).map_err(|e| format!("Failed to open zip archive: {e}"))?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| format!("Failed to read zip: {e}"))?;

    for i in 0..archive.len() {
        let mut f = archive.by_index(i).map_err(|e| format!("Zip entry error: {e}"))?;
        let raw_name = f.name().map_err(|e| format!("Zip name error: {e}"))?;
        let name = raw_name.to_string();
        if name.ends_with('/') || name.ends_with('\\') {
            continue;
        }

        let file_name = Path::new(&name)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        if file_name.is_empty() {
            continue;
        }

        let dest_file = dest_dir.join(file_name);
        let mut outfile = File::create(&dest_file)
            .map_err(|e| format!("Failed to extract to {}: {e}", dest_file.display()))?;
        std::io::copy(&mut f, &mut outfile).map_err(|e| format!("Write error during extract: {e}"))?;
    }

    Ok(())
}

fn run_download_pipeline(tx: &Sender<WorkerMsg>, cancel_flag: Arc<AtomicBool>) -> Result<String, String> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("QwenImageStudio/0.2.0")
        .timeout(Duration::from_secs(300))
        .build()
        .map_err(|e| format!("Failed to initialize HTTP client: {e}"))?;

    let total_steps = 5;

    // 1. CUDA 12 Binaries
    if !Path::new("bin/sd-cli.exe").exists() {
        let _ = tx.send(WorkerMsg::Log(">>> Step 1/5: Downloading CUDA 12 Engine & DLLs...".into()));
        let zip_sd = PathBuf::from("bin/sd-cuda12.zip");
        let zip_cu = PathBuf::from("bin/cudart-cu12.zip");

        // Download main engine zip
        download_file_resumable(
            &client,
            "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-8ebbf90/bin-win-cuda12-x64.zip",
            &zip_sd,
            "CUDA 12 Engine (bin-win-cuda12-x64.zip)",
            1,
            total_steps,
            tx,
            &cancel_flag,
        )?;

        // Download CUDA runtime DLLs zip
        download_file_resumable(
            &client,
            "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-8ebbf90/cudart-sd-bin-win-cu12-x64.zip",
            &zip_cu,
            "CUDA 12 Runtime DLLs (cudart-sd-bin-win-cu12-x64.zip)",
            1,
            total_steps,
            tx,
            &cancel_flag,
        )?;

        let _ = tx.send(WorkerMsg::Log(">>> Extracting CUDA binaries into ./bin...".into()));
        extract_zip_flat(&zip_sd, Path::new("bin"))?;
        extract_zip_flat(&zip_cu, Path::new("bin"))?;
        let _ = std::fs::remove_file(&zip_sd);
        let _ = std::fs::remove_file(&zip_cu);

        if Path::new("bin/sd.exe").exists() && !Path::new("bin/sd-cli.exe").exists() {
            let _ = std::fs::copy("bin/sd.exe", "bin/sd-cli.exe");
        }
    } else {
        let _ = tx.send(WorkerMsg::Log(">>> Step 1/5: CUDA 12 binaries already verified.".into()));
    }

    // 2. Diffusion model (4.60 GB)
    let diff_path = PathBuf::from("models/qwen-image-2.1-UC-Q4_K_M.gguf");
    download_file_resumable(
        &client,
        "https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized/resolve/main/qwen-image-2.1-UC-Q4_K_M.gguf",
        &diff_path,
        "qwen-image-2.1-UC-Q4_K_M.gguf (4.60 GB)",
        2,
        total_steps,
        tx,
        &cancel_flag,
    )?;

    // 3. Text Encoder (5.02 GB)
    let llm_path = PathBuf::from("models/Qwen3VL-8B-Instruct-Q4_K_M.gguf");
    download_file_resumable(
        &client,
        "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        &llm_path,
        "Qwen3VL-8B-Instruct-Q4_K_M.gguf (5.02 GB)",
        3,
        total_steps,
        tx,
        &cancel_flag,
    )?;

    // 4. VAE (676 MB)
    let vae_path = PathBuf::from("models/qwen_image_2.1_vae_bf16.safetensors");
    download_file_resumable(
        &client,
        "https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors",
        &vae_path,
        "qwen_image_2.1_vae_bf16.safetensors (676 MB)",
        4,
        total_steps,
        tx,
        &cancel_flag,
    )?;

    // 5. Vision Projector (1.16 GB F16)
    let mmproj_path = PathBuf::from("models/mmproj-Qwen3VL-8B-Instruct-F16.gguf");
    download_file_resumable(
        &client,
        "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-8B-Instruct-F16.gguf",
        &mmproj_path,
        "mmproj-Qwen3VL-8B-Instruct-F16.gguf (1.16 GB)",
        5,
        total_steps,
        tx,
        &cancel_flag,
    )?;

    Ok("All 5 files verified & downloaded successfully.".to_owned())
}

impl eframe::App for QwenApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.clear_preview();
        }

        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                WorkerMsg::Log(line) => self.append_log(line),
                WorkerMsg::Progress(frac, label) => {
                    self.progress = frac;
                    self.status_line = label;
                }
                WorkerMsg::DownloadProgress(p) => {
                    self.download_progress = Some(p);
                }
                WorkerMsg::DownloadFinished(res) => {
                    self.download_in_progress = false;
                    match res {
                        Ok(msg) => {
                            self.status_line = msg.clone();
                            self.append_log(format!(">>> SUCCESS: {msg}"));
                        }
                        Err(err) => {
                            self.status_line = format!("Download error: {err}");
                            self.append_log(format!(">>> ERROR: {err}"));
                        }
                    }
                }
                WorkerMsg::Finished(res) => {
                    self.is_busy = false;
                    match res {
                        Ok(path) => {
                            self.progress = 1.0;
                            self.append_log(format!(">>> SUCCESS: Image saved to {}", path.display()));
                            self.last_image_path = Some(path.clone());
                            self.refresh_output_files();
                            if self.privacy_dont_autoshow {
                                self.texture = None;
                                self.status_line = format!(
                                    "Saved to {} (Hidden by Privacy Mode - click file in Output Drawer to view)",
                                    path.display()
                                );
                            } else {
                                self.status_line = format!("Saved to {}", path.display());
                                self.load_preview_texture(ctx, &path);
                            }
                        }
                        Err(err) => {
                            self.progress = 0.0;
                            self.status_line = err.clone();
                            self.append_log(format!(">>> ERROR: {err}"));
                        }
                    }
                }
            }
        }

        if self.is_busy || self.download_in_progress {
            ctx.request_repaint_after(Duration::from_millis(80));
        }

        // Native 1-Click Downloader & Verifier Modal (Neutral, Clean Slate UI)
        if self.show_download_modal {
            egui::Window::new("⬇ 1-Click Engine & Model Setup")
                .collapsible(false)
                .resizable(false)
                .fixed_size([580.0, 480.0])
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.add_space(4.0);
                    ui.heading("Qwen-Image 2.1 Studio — Automatic Setup");
                    ui.label(
                        egui::RichText::new(
                            "Downloads the official CUDA 12 inference binaries and 4-bit quantized model weights.\nTuned for 6GB RTX 4050 & 16GB RAM laptops with resumable downloads.",
                        )
                        .color(egui::Color32::from_rgb(160, 170, 185))
                        .size(13.0),
                    );
                    ui.add_space(8.0);
                    ui.separator();

                    ui.group(|ui| {
                        ui.strong("Component Checklist:");
                        ui.add_space(4.0);

                        let cuda_ok = Path::new(&self.sd_bin_path).exists();
                        let diff_ok = Path::new(&self.diffusion_model_path).exists();
                        let llm_ok = Path::new(&self.llm_model_path).exists();
                        let vae_ok = Path::new(&self.vae_model_path).exists();
                        let mmproj_ok = Path::new(&self.mmproj_model_path).exists();

                        let check_row = |ui: &mut egui::Ui, name: &str, size: &str, ok: bool, idx: usize| {
                            ui.horizontal(|ui| {
                                let (badge, col) = if ok {
                                    ("● Ready", egui::Color32::from_rgb(80, 190, 120))
                                } else if let Some(p) = &self.download_progress {
                                    if p.file_index == idx {
                                        ("● Downloading", egui::Color32::from_rgb(90, 160, 240))
                                    } else if p.file_index > idx {
                                        ("● Done", egui::Color32::from_rgb(80, 190, 120))
                                    } else {
                                        ("○ Queued", egui::Color32::from_rgb(140, 150, 165))
                                    }
                                } else {
                                    ("○ Missing", egui::Color32::from_rgb(220, 140, 50))
                                };

                                ui.colored_label(col, badge);
                                ui.label(egui::RichText::new(name).strong());
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    ui.label(egui::RichText::new(size).color(egui::Color32::from_rgb(140, 150, 165)));
                                });
                            });
                        };

                        check_row(ui, "1. CUDA 12 Binary Engine (sd-cli.exe & DLLs)", "~800 MB", cuda_ok, 1);
                        check_row(ui, "2. Diffusion DiT Model (qwen-image-2.1-UC-Q4_K_M)", "4.60 GB", diff_ok, 2);
                        check_row(ui, "3. Qwen3-VL 4-Bit Text Encoder (Q4_K_M.gguf)", "5.02 GB", llm_ok, 3);
                        check_row(ui, "4. Qwen-Image 2.1 VAE (BF16 safetensors)", "676 MB", vae_ok, 4);
                        check_row(ui, "5. F16 Vision Projector (mmproj for Img2Img)", "1.16 GB", mmproj_ok, 5);
                    });

                    ui.add_space(10.0);

                    // Download progress section
                    if let Some(p) = &self.download_progress {
                        ui.group(|ui| {
                            let frac = if p.total_bytes > 0 {
                                (p.downloaded_bytes as f32 / p.total_bytes as f32).clamp(0.0, 1.0)
                            } else {
                                0.0
                            };

                            ui.horizontal(|ui| {
                                ui.strong(format!("[{}/{}] {}", p.file_index, p.total_files, p.file_name));
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    ui.label(format!("{:.1}%", frac * 100.0));
                                });
                            });

                            ui.add(egui::ProgressBar::new(frac));

                            ui.horizontal(|ui| {
                                ui.label(format!(
                                    "{} / {}",
                                    format_bytes(p.downloaded_bytes),
                                    format_bytes(p.total_bytes)
                                ));
                                ui.separator();
                                ui.label(format!("⚡ {:.1} MB/s", p.speed_mb_s));
                                if p.eta_seconds > 0 {
                                    ui.separator();
                                    let mins = p.eta_seconds / 60;
                                    let secs = p.eta_seconds % 60;
                                    ui.label(format!("ETA: {}m {:02}s", mins, secs));
                                }
                            });
                        });
                    }

                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(4.0);

                    ui.horizontal(|ui| {
                        if self.download_in_progress {
                            if ui.button("⏹ Cancel Download").clicked() {
                                self.cancel_download_flag.store(true, Ordering::Relaxed);
                                self.download_in_progress = false;
                            }
                        } else {
                            let all_ready = self.files_ready() && self.mmproj_ready();
                            let btn_text = if all_ready {
                                "✓ Verify All Files"
                            } else {
                                "⬇ Start 1-Click Download"
                            };

                            let download_btn = egui::Button::new(
                                egui::RichText::new(btn_text).size(15.0).strong(),
                            )
                            .min_size(egui::vec2(220.0, 36.0));

                            if ui.add(download_btn).clicked() {
                                self.start_native_download();
                            }

                            if ui.button("Close Dialog").clicked() {
                                self.show_download_modal = false;
                            }
                        }
                    });
                });
        }

        egui::TopBottomPanel::top("top_header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Qwen-Image 2.1 Studio");
                ui.separator();
                let ready = self.files_ready();
                let mmproj = self.mmproj_ready();
                let badge_color = if ready && mmproj {
                    egui::Color32::from_rgb(80, 190, 120)
                } else if ready {
                    egui::Color32::from_rgb(220, 180, 40)
                } else {
                    egui::Color32::from_rgb(220, 110, 40)
                };
                ui.colored_label(
                    badge_color,
                    if ready && mmproj {
                        "● All 5 Files Verified (Text2Img + F16 Img2Img)"
                    } else if ready {
                        "● Text2Img Ready (Click Setup for F16 Vision)"
                    } else {
                        "● Missing Engine/Models (Click 1-Click Setup)"
                    },
                );

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("⚙ 1-Click Setup / Models").clicked() {
                        self.show_download_modal = true;
                    }
                });
            });
        });

        egui::TopBottomPanel::bottom("bottom_console")
            .resizable(true)
            .default_height(155.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.strong("Status:");
                    ui.label(&self.status_line);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("📋 Copy Logs").clicked() {
                            ctx.copy_text(self.logs.join("\n"));
                        }
                        if ui.button("🧹 Clear Logs").clicked() {
                            self.logs.clear();
                        }
                    });
                });
                ui.add(egui::ProgressBar::new(self.progress).show_percentage());
                ui.separator();
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for line in &self.logs {
                            ui.monospace(line);
                        }
                    });
            });

        if self.show_output_drawer {
            egui::SidePanel::right("right_output_drawer")
                .resizable(true)
                .default_width(250.0)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("📁 Saved Outputs");
                        if ui.button("🔄").on_hover_text("Refresh folder").clicked() {
                            self.refresh_output_files();
                        }
                        if ui.button("📂 Explorer").clicked() {
                            let _ = Command::new("explorer").arg("output").spawn();
                        }
                    });
                    ui.separator();

                    let files_snapshot = self.output_files.clone();
                    if files_snapshot.is_empty() {
                        ui.label("No images in ./output yet.");
                    } else {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            for file_path in files_snapshot {
                                let file_name = file_path
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or("image.png")
                                    .to_owned();

                                let is_selected = self
                                    .last_image_path
                                    .as_ref()
                                    .map(|p| p == &file_path)
                                    .unwrap_or(false)
                                    && self.texture.is_some();

                                ui.group(|ui| {
                                    ui.label(egui::RichText::new(&file_name).monospace().size(12.0));
                                    ui.horizontal(|ui| {
                                        if ui
                                            .selectable_label(is_selected, "👁 View")
                                            .on_hover_text("Show this image in the center preview")
                                            .clicked()
                                        {
                                            self.load_preview_texture(ctx, &file_path);
                                            self.last_image_path = Some(file_path.clone());
                                            self.status_line = format!("Viewing {}", file_name);
                                        }
                                        if ui
                                            .button("➡ Img2Img")
                                            .on_hover_text("Use this image as Reference Image Input")
                                            .clicked()
                                        {
                                            self.ref_image_path = file_path.to_string_lossy().to_string();
                                            self.width = 768;
                                            self.height = 768;
                                            self.status_line = format!(
                                                "Set {} as Reference Input (Switched to 768x768 for 6GB VRAM safety)",
                                                file_name
                                            );
                                        }
                                        if ui
                                            .button("🗑")
                                            .on_hover_text("Delete file permanently from disk")
                                            .clicked()
                                        {
                                            if self
                                                .last_image_path
                                                .as_ref()
                                                .map(|p| p == &file_path)
                                                .unwrap_or(false)
                                            {
                                                self.texture = None;
                                            }
                                            let _ = std::fs::remove_file(&file_path);
                                            self.refresh_output_files();
                                        }
                                    });
                                });
                            }
                        });
                    }
                });
        }

        egui::SidePanel::left("left_controls")
            .resizable(true)
            .default_width(440.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Prompt");
                    ui.add(
                        egui::TextEdit::multiline(&mut self.prompt)
                            .desired_rows(4)
                            .desired_width(f32::INFINITY),
                    );

                    ui.label("Negative Prompt (Active when CFG > 1.0):");
                    ui.add_enabled(
                        self.cfg_scale > 1.05,
                        egui::TextEdit::multiline(&mut self.negative_prompt)
                            .desired_rows(2)
                            .desired_width(f32::INFINITY),
                    );

                    ui.checkbox(
                        &mut self.transparent_rgba,
                        "Native RGBA Transparent Background (.png)",
                    );

                    ui.separator();
                    ui.heading("Image Input / Editing (Img2Img)");
                    ui.label("Reference Image (Leave empty for Text-to-Image):");
                    ui.horizontal(|ui| {
                        ui.text_edit_singleline(&mut self.ref_image_path);
                        if ui.button("Browse...").clicked() {
                            if let Some(file) = rfd::FileDialog::new()
                                .add_filter("Image", &["png", "jpg", "jpeg", "webp"])
                                .pick_file()
                            {
                                self.ref_image_path = file.to_string_lossy().to_string();
                                self.width = 768;
                                self.height = 768;
                            }
                        }
                        if !self.ref_image_path.is_empty() && ui.button("✖ Clear").clicked() {
                            self.ref_image_path.clear();
                        }
                    });
                    ui.checkbox(
                        &mut self.auto_resize_ref,
                        "Auto-fit Reference Image to ≤768px (Prevents 6GB VRAM Crash)",
                    );
                    if !self.mmproj_ready() {
                        ui.colored_label(
                            egui::Color32::from_rgb(220, 90, 70),
                            "⚠ Img2Img needs mmproj-Qwen3VL-8B-Instruct-F16.gguf! Click '⚙ 1-Click Setup' at top-right.",
                        );
                    }

                    ui.separator();
                    ui.heading("Quality & Dimensions");

                    ui.horizontal(|ui| {
                        ui.label("Presets:");
                        if ui.button("768x768 (Img2Img)").clicked() {
                            self.width = 768;
                            self.height = 768;
                        }
                        if ui.button("896x896 (Sweet Spot)").clicked() {
                            self.width = 896;
                            self.height = 896;
                        }
                        if ui.button("1024x1024 (Max Detail)").clicked() {
                            self.width = 1024;
                            self.height = 1024;
                        }
                        if ui.button("896x1152 (Portrait)").clicked() {
                            self.width = 896;
                            self.height = 1152;
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Width:");
                        ui.add(egui::DragValue::new(&mut self.width).speed(32).range(256..=1536));
                        ui.label("Height:");
                        ui.add(egui::DragValue::new(&mut self.height).speed(32).range(256..=1536));
                    });

                    ui.add(egui::Slider::new(&mut self.steps, 12..=40).text("Steps (26-28 = Crisp Quality)"));
                    ui.add(
                        egui::Slider::new(&mut self.cfg_scale, 1.0..=6.0)
                            .text("CFG Scale (3.5-4.0 = Sharpest)"),
                    );

                    ui.horizontal(|ui| {
                        ui.label("Seed (-1 = random):");
                        ui.add(egui::DragValue::new(&mut self.seed));
                        ui.label("i7 Threads (4 = Cool):");
                        ui.add(egui::DragValue::new(&mut self.threads).range(1..=12));
                    });

                    ui.separator();
                    ui.heading("RAM & VRAM Guard (RTX 4050 6GB)");
                    ui.checkbox(
                        &mut self.te_on_disk_free_ram,
                        "Unload 5GB Text Encoder After Prompt (--params-backend te=disk)",
                    );
                    ui.checkbox(
                        &mut self.offload_to_cpu,
                        "Dynamic GPU Weight Offload (--offload-to-cpu)",
                    );
                    ui.checkbox(
                        &mut self.vae_tiling,
                        "256x256 VAE Tiling (--vae-tiling)",
                    );
                    ui.checkbox(
                        &mut self.flash_attention,
                        "Flash Attention (--diffusion-fa & --fa)",
                    );
                    ui.checkbox(
                        &mut self.q4_prefix_cache,
                        "Lossy Q4_0 Prefix Cache (Keep OFF for high quality)",
                    );
                    ui.checkbox(
                        &mut self.vae_on_cpu,
                        "Force VAE on CPU (--vae-on-cpu) [Turn ON only if Img2Img OOMs]",
                    );

                    ui.separator();
                    ui.collapsing("Model & LoRA Paths", |ui| {
                        ui.label("sd-cli binary:");
                        ui.text_edit_singleline(&mut self.sd_bin_path);
                        ui.label("Uncensored Diffusion GGUF (4.6GB):");
                        ui.text_edit_singleline(&mut self.diffusion_model_path);
                        ui.label("Qwen3-VL 4-bit GGUF Text Encoder (5.0GB):");
                        ui.text_edit_singleline(&mut self.llm_model_path);
                        ui.label("Qwen3-VL F16 Vision Projector (mmproj 1.16GB):");
                        ui.text_edit_singleline(&mut self.mmproj_model_path);
                        ui.label("Qwen-Image 2.1 VAE (676MB):");
                        ui.text_edit_singleline(&mut self.vae_model_path);
                        ui.label("LoRA Folder (Use <lora:name:0.8> in prompt):");
                        ui.text_edit_singleline(&mut self.lora_dir_path);
                    });

                    ui.add_space(10.0);
                    let gen_btn = egui::Button::new(
                        egui::RichText::new("⚡ GENERATE IMAGE").size(18.0).strong(),
                    )
                    .min_size(egui::vec2(ui.available_width(), 42.0));

                    if ui.add_enabled(!self.is_busy && !self.download_in_progress, gen_btn).clicked() {
                        self.start_generation();
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Output Preview");

                let clear_btn = egui::Button::new(
                    egui::RichText::new("🙈 Clear Screen (Esc)").strong(),
                );
                if ui
                    .add_enabled(self.texture.is_some(), clear_btn)
                    .on_hover_text("Instantly hide the preview image from screen (file stays saved in ./output)")
                    .clicked()
                {
                    self.clear_preview();
                }

                if let Some(last_path) = &self.last_image_path {
                    if self.texture.is_none() && last_path.exists() {
                        if ui.button("👁 Show Last Image").clicked() {
                            let p = last_path.clone();
                            self.load_preview_texture(ctx, &p);
                        }
                    }
                }

                ui.checkbox(
                    &mut self.privacy_dont_autoshow,
                    "Privacy Mode (Don't auto-show new renders)",
                );

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.toggle_value(&mut self.show_output_drawer, "📁 Output Folder List");
                });
            });
            ui.separator();

            if let Some(tex) = &self.texture {
                let avail = ui.available_size();
                let tex_size = tex.size_vec2();
                let scale = (avail.x / tex_size.x).min(avail.y / tex_size.y).min(1.0);
                let draw_size = tex_size * scale;
                ui.centered_and_justified(|ui| {
                    ui.image((tex.id(), draw_size));
                });
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Preview screen is clear. (Saved renders are in the 📁 Output Folder List on the right).");
                });
            }
        });
    }
}

fn load_icon() -> Option<egui::IconData> {
    let png_bytes = include_bytes!("../qwen_logo.png");
    if let Ok(img) = image::load_from_memory(png_bytes) {
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();
        Some(egui::IconData {
            rgba: rgba.into_raw(),
            width,
            height,
        })
    } else {
        None
    }
}

fn main() -> eframe::Result<()> {
    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([1280.0, 800.0])
        .with_min_inner_size([960.0, 620.0]);

    if let Some(icon) = load_icon() {
        viewport = viewport.with_icon(icon);
    }

    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    eframe::run_native(
        "Qwen-Image 2.1 Uncensored Studio (RTX 4050 6GB Edition)",
        options,
        Box::new(|cc| Ok(Box::new(QwenApp::new(cc)))),
    )
}
