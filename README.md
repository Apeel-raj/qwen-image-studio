# 🎨 Qwen-Image 2.1 Studio (RTX 4050 6GB Laptop Edition)

[![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange?logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Platform](https://img.shields.io/badge/Platform-Windows_x64-0078D6?logo=windows)](https://github.com/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Model](https://img.shields.io/badge/Model-Qwen--Image--2.1--Uncensored-8A2BE2)](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized)

A blazing-fast, memory-optimized native **Rust desktop studio** for generating and editing images with **Qwen-Image 2.1 Uncensored**.

Engineered specifically for **6 GB VRAM GPUs (NVIDIA GeForce RTX 4050 Laptop)** and **16 GB System RAM Windows machines**, running 100% locally with zero Python, zero PyTorch bloat, and zero SSD pagefile thrashing.

---

## ⚡ The Problem with Running Qwen-Image 2.1 on 16GB Laptops

Standard Python/ComfyUI setups for Qwen-Image 2.1 load the unquantized 9.35 GB safetensors text encoder and 20 GB DiT model into memory. On a 16 GB Windows laptop (where Windows uses ~4.5 GB RAM), this immediately exceeds 16 GB, causing **catastrophic SSD pagefile thrashing (100% disk usage, severe heat spikes, and memory crash freezes)**.

### How This Studio Solves It:
1. **4-Bit GGUF Text Encoder (`Qwen3VL-8B-Instruct-Q4_K_M.gguf`, ~5.0 GB):** Replaces the 9.35 GB FP16/INT8 safetensors encoder, saving over 4.3 GB of System RAM.
2. **Dynamic RAM Offloading (`--params-backend te=disk`):** Encodes your prompt once, then **instantly unloads the 5 GB text encoder from RAM** before the diffusion sampling loop begins. Windows never touches your SSD swap file.
3. **RTX 4050 6GB VRAM Optimization:** Uses 4-bit DiT weights (`qwen-image-2.1-UC-Q4_K_M.gguf`, 4.60 GB), flash attention (`--diffusion-fa --fa`), and tiled VAE decoding (`--vae-tiling`) to comfortably fit inside 6 GB VRAM.
4. **Laptop Heatsink & Thermal Protection:** Caps background compute to 4 CPU threads at `BELOW_NORMAL_PRIORITY_CLASS`, preventing CPU/GPU shared heatpipe baking.

---

## ✨ Features

- **1-Click Native Downloader & Verifier:** No command-line commands or PowerShell scripts needed. The built-in downloader streams CUDA 12 binaries and 4-bit models directly with real-time transfer speed (MB/s), percentage, and byte-level resume support.
- **Text-to-Image & Image-to-Image (Img2Img):** Full support for reference images using the official `mmproj-Qwen3VL-8B-Instruct-F16.gguf` vision projector.
- **Smart Image Preprocessing:** Reference photos are automatically downscaled to $\le 768\text{px}$ (divisible by 32) before vision encoding to prevent VRAM out-of-memory errors on 6 GB GPUs.
- **Built-in Output History Drawer:** View, inspect, re-use as Img2Img input, or delete saved renders right from the side panel.
- **Instant Privacy ("Boss Key" / Esc):** Press **`Esc`** at any time to instantly wipe the canvas preview while keeping your files safely stored in `./output`. Includes an optional **Privacy Mode** (renders without automatically showing the image on screen).
- **LoRA Support:** Drop your fine-tuned `.safetensors` LoRAs into `./loras` and trigger them via `<lora:name:weight>` in your prompt.
- **Copy Logs Diagnostic:** 1-click clipboard log copy for fast troubleshooting.

---

## 🖥️ System Requirements

| Component | Minimum Specification | Recommended Specification |
|---|---|---|
| **OS** | Windows 10 / 11 64-bit | Windows 11 64-bit |
| **GPU** | NVIDIA GeForce RTX 3050/4050 (6 GB VRAM) | NVIDIA RTX 4050 Laptop (6 GB VRAM) or higher |
| **System RAM** | 16 GB DDR4/DDR5 | 16 GB - 32 GB |
| **Storage** | 25 GB free space (NVMe SSD recommended) | Fast NVMe SSD |
| **Drivers** | NVIDIA Game Ready or Studio Driver (CUDA 12.x+) | Latest NVIDIA Studio Driver |

---

## 🚀 Quick Start (1-Click Run)

1. Download the latest `qwen_image_studio.exe` from **[Releases](https://github.com/Apeel-raj/qwen-image-studio/releases)**.
2. Double-click `qwen_image_studio.exe` to launch the app.
3. On first launch, click **"⬇ 1-Click Download / Verify Models"**.
4. The studio will automatically download and verify all 5 necessary components:
   - `sd-cli.exe` + CUDA 12 Runtime DLLs (`./bin`)
   - `qwen-image-2.1-UC-Q4_K_M.gguf` (4.60 GB DiT model)
   - `Qwen3VL-8B-Instruct-Q4_K_M.gguf` (5.02 GB 4-bit Text Encoder)
   - `qwen_image_2.1_vae_bf16.safetensors` (676 MB VAE)
   - `mmproj-Qwen3VL-8B-Instruct-F16.gguf` (1.16 GB Vision Projector for Img2Img)
5. Type your prompt, choose your resolution (e.g. `896x896`), and click **⚡ GENERATE IMAGE**!

---

## 🛠️ Building from Source

### Prerequisites
- [Rust & Cargo](https://rustup.rs/) (version 1.80+)
- Windows C++ Build Tools (MSVC)

### Build Steps
```powershell
# Clone the repository
git clone https://github.com/Apeel-raj/qwen-image-studio.git
cd qwen-image-studio

# Build optimized release binary
cargo build --release

# Run the studio
.\target\release\qwen_image_studio.exe
```

---

## 🎯 Recommended Generation Settings (RTX 4050 6GB)

| Task | Resolution | Steps | CFG Scale | VRAM / RAM Behavior |
|---|---|---|---|---|
| **Text-to-Image (Sweet Spot)** | `896x896` | 26 | 3.5 | ~5.2 GB VRAM • ~9.5 GB RAM peak |
| **Text-to-Image (Max Detail)** | `1024x1024` | 28 | 3.5 | ~5.8 GB VRAM • ~10.5 GB RAM peak |
| **Image-to-Image (Img2Img)** | `768x768` | 24 | 3.0 – 3.5 | ~5.4 GB VRAM (auto-resizes ref photo) |
| **Fast / Cool Eco Mode** | `768x768` | 20 | 2.5 | Minimal thermal footprint |

---

## 📜 Acknowledgments & License

- **Qwen Team (Alibaba Cloud):** Creators of the Qwen-Image-2.1 and Qwen3-VL models.
- **[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp):** High-performance C/C++ GGML diffusion inference engine by @leejet.
- **[abenzerps](https://huggingface.co/abenzerps):** Quantization & packaging of the uncensored Qwen-Image 2.1 weights.
- **[egui / eframe](https://github.com/emilk/egui):** Immediate-mode Rust GUI framework by @emilk.

This project is licensed under the **[Apache License 2.0](LICENSE)**.
