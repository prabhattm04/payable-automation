# Phase 6C-1: Hardware & Environment Feasibility Report
**Target Model**: `Qwen/Qwen3-VL-4B-Instruct`  
**Execution Target**: Zero-Cost Local Inference via Hugging Face Transformers  
**Date**: September 14, 2026  
**Document Status**: Review Required Before Model Download or Environment Creation  

---

## 1. Executive Summary & Feasibility Verdict

| Evaluation Parameter | Result | Assessment |
| :--- | :--- | :--- |
| **Model** | `Qwen/Qwen3-VL-4B-Instruct` (~4.44B parameters) | Multimodal Vision-Language Model |
| **Inference Target** | Local execution (0 API cost, 100% offline) | Mandatory per project directive |
| **Primary Device** | **CPU Only** (`device_map="cpu"`) | Integrated AMD GPU lacks sufficient VRAM (512 MB); no NVIDIA GPU |
| **Recommended Dtype** | `torch.bfloat16` | FP32 requires ~17.8 GB and will cause immediate OOM |
| **Flash-Attention** | **NOT USABLE** | Unsupported on CPU and non-CUDA environments |
| **Transformers Support** | **INSUFFICIENT in current base** (`4.48.0`) | Requires `transformers >= 4.57.0` |
| **Environment Status** | Current environment shared with Phase 6B | **Dedicated isolated virtual environment required** |
| **Local Feasibility** | **FEASIBLE WITH CONSTRAINTS (CPU Execution)** | See Memory Considerations & Latency Profile below |

---

## 2. Hardware Inspection

All metrics below were collected directly from the active host machine via Windows WMI/CIM and Python system diagnostic probes.

| Hardware Component | Detected System Metric | Technical Note |
| :--- | :--- | :--- |
| **Operating System** | **Windows 11 Home Single Language (64-bit)** | Version 10.0.26200-SP0 (Build 26200) |
| **CPU Model** | **AMD Ryzen 7 5800HS with Radeon Graphics** | 8 Physical Cores, 16 Logical Processors, Zen 3 architecture |
| **CPU Instruction Sets** | AVX, AVX2, FMA3, BMI1, BMI2 | Enables vectorized bfloat16/float32 tensor operations |
| **Total System RAM** | **16.00 GB Physical** (`16,153,948 KB` visible / `15.41 GiB`) | Shared system unified memory |
| **Available System RAM** | **~3.65 GB to 3.85 GB Free Physical** (`3,830,680 KB`) | Background OS and applications currently consume ~11.7 GB |
| **Virtual Memory (Commit)** | Total: `20.40 GB`, Available Commit: `~4.07 GB` | Pagefile configured at ~4.99 GB |
| **GPU Availability** | **Integrated GPU ONLY** (No discrete GPU) | Dedicated discrete GPU: **None** |
| **GPU Model** | **AMD Radeon(TM) Graphics** | PCI Device ID: `PCI\VEN_1002&DEV_1638` |
| **GPU VRAM** | **512 MB** dedicated video memory (`536,870,912 bytes`) | Insufficient to hold even a 0.5B model |
| **CUDA Availability** | **None** (`torch.cuda.is_available() == False`) | `nvidia-smi` not found, no NVIDIA hardware present |
| **Disk Storage (Drive D:)** | **216.9 GB Free** (Total: 238.5 GB) | Excellent capacity for model weights and caches |
| **Disk Storage (Drive C:)** | **36.4 GB Free** (Total: 237.5 GB) | Recommended to set `HF_HOME` to `D:\` to save OS drive |

---

## 3. Current Python Environment Audit

| Environment Dimension | Current Machine State | Impact on Phase 6C |
| :--- | :--- | :--- |
| **Python Version** | `3.12.1 (AMD64)` (`C:\Program Files\Python312\python.exe`) | Standard modern runtime |
| **Virtual Environment** | `sys.prefix == sys.base_prefix` (`In venv: False`) | Currently running in base global Python interpreter |
| **Phase 6B Sharing** | **YES — Shared with Phase 6B** | Phase 6B libraries (`rapidocr-onnxruntime==1.4.4`, `onnxruntime==1.29.0`, `PyMuPDF==1.26.3`, `pydantic==2.9.2`, `paddlepaddle==3.3.1`, `paddlex==3.7.2`) reside in this environment |
| **Installed PyTorch** | `torch == 2.14.0+cpu` | CPU-only build installed |
| **Installed Transformers** | `transformers == 4.48.0` | Installed in user site-packages |
| **Qwen3-VL Support** | **FAILED**: `cannot import name 'Qwen3VLForConditionalGeneration'` | `transformers==4.48.0` predates Qwen3-VL release (introduced in `transformers>=4.57.0`) |
| **Installed Accelerate** | `Not installed` | Required for Hugging Face multi-device dispatch and memory offloading |

---

## 4. Flash-Attention Compatibility Assessment

**Verdict: NOT USABLE — Must be strictly excluded.**

1. **Hardware Incompatibility**: `flash-attn` requires NVIDIA GPUs with Tensor Cores (Compute Capability $\ge 8.0$: Ampere, Ada Lovelace, Hopper). This machine has an **AMD Ryzen CPU and AMD integrated Radeon graphics**.
2. **Platform & Compiler Incompatibility**: Official `flash-attn` wheels are built for Linux CUDA environments. Windows installations require MSVC C++ Build Tools with matching CUDA Toolkit headers and custom PyTorch extensions.
3. **No CPU Support**: Flash-Attention kernel optimizations are written in CUDA/Triton specifically for NVIDIA GPU hardware; there is no native CPU implementation of `flash-attn`.
4. **Resolution**: `Qwen3VLForConditionalGeneration` will use PyTorch's native SDPA (`Scaled Dot Product Attention`) or standard eager attention (`attn_implementation="sdpa"`), which is natively supported on CPU.

---

## 5. Recommended Execution Configuration

### 5.1 Recommended Device: `cpu`
- **Configuration**: `device_map="cpu"` (or `device_map="auto"`, which automatically binds all layers to CPU given zero CUDA devices).
- **Rationale**: The AMD integrated Radeon graphics only has 512 MB of VRAM. Attempting DirectML or Vulkan mapping would fail due to insufficient VRAM. The 8-core AMD Ryzen 7 5800HS CPU (16 threads, AVX2) is the only viable execution target.

### 5.2 Recommended Dtype: `torch.bfloat16`
- **Configuration**: `torch_dtype=torch.bfloat16`
- **Rejection of `torch.float32`**:
  $$\text{Memory}_{\text{FP32}} \approx 4.44 \times 10^9 \text{ params} \times 4 \text{ bytes} \approx 17.76 \text{ GB}$$
  This strictly exceeds the entire system physical memory (15.41 GB), guaranteeing an unrecoverable `OutOfMemoryError` or severe Windows pagefile thrashing.
- **Suitability of `torch.bfloat16`**:
  $$\text{Memory}_{\text{BF16}} \approx 4.44 \times 10^9 \text{ params} \times 2 \text{ bytes} \approx 8.88 \text{ GB}$$
  Modern PyTorch (`>=2.4.0`) supports native bfloat16 tensor arithmetic on x86 CPUs with AVX2. If bfloat16 emulation produces any numerical precision warnings on Zen 3, `torch.float16` serves as a verified drop-in fallback.

---

## 6. Memory Considerations & Local Feasibility Analysis

### 6.1 Memory Breakdown During Inference
| Component | Estimated Footprint (BF16) | Notes |
| :--- | :--- | :--- |
| **Model Weights** | ~8.88 GB | Raw model parameter weights loaded into RAM |
| **Vision Processor / Patches** | ~0.50 – 1.00 GB | Image patch embeddings from rendered PDF page |
| **KV Cache & Context Window** | ~0.80 – 1.50 GB | Context tokens for document prompt and reasoning |
| **PyTorch CPU Overhead** | ~0.50 GB | Framework runtime buffers |
| **Total Inference RAM Required**| **~10.7 GB – 11.8 GB** | Dynamic peak footprint during `.generate()` |

### 6.2 The Free RAM Constraint
- **Current Physical RAM Free**: **~3.65 GB to 3.85 GB** (out of 15.41 GB total).
- **Available Virtual Memory**: **~4.07 GB commit limit remaining** before Windows paging limit is reached.
- **Key Risk**: Loading an 8.88 GB model when only ~3.65 GB physical RAM is free will force Windows to page out existing background memory to disk, and if the commit charge exceeds the virtual memory limit, PyTorch will terminate with `RuntimeError: [enforce fail at ... CPUAllocator.cpp] DefaultCPUAllocator: can't allocate memory`.

### 6.3 Mandatory Pre-Flight Memory Recommendations
To ensure reliable local execution without memory exhaustion:
1. **Close Memory-Intensive Applications**:
   - Close active browser instances, background Electron applications, and development servers during vision benchmarking to free up at least 8.5–10.0 GB of physical RAM.
2. **Windows Pagefile Expansion**:
   - Ensure the Windows paging file (virtual memory) on SSD is configured to at least 16–24 GB (currently ~4.99 GB) to provide a safe virtual commit buffer.
3. **Targeted Benchmarking**:
   - In Phase 6C, execute spot-check benchmarking on representative document pages (e.g., INV-01, HLD-01, DU-02) rather than attempting bulk continuous batch inference.
4. **Expected Inference Latency**:
   - On an 8-core AMD Ryzen 7 5800HS CPU, generating 200–300 tokens for document extraction is estimated to take **45 to 90 seconds per page** (approximately 3–5 tokens/second). Single-page benchmarks are feasible; running all 111 pages consecutively would require ~2.5 hours of continuous CPU execution.

---

## 7. Proposed Isolated Environment

### 7.1 The Need for Strict Isolation
- Upgrading `transformers` from `4.48.0` to `4.57+` or `5.x` in the global environment risks breaking dependencies pinned by Phase 6B's installed packages (such as `paddlex 3.7.2`, `paddleocr 3.7.0`, `langchain`, etc.).
- Creating a dedicated virtual environment ensures **zero modifications** to the existing Phase 6B OCR pipeline, preserving all 86 passing tests.

### 7.2 Proposed Environment Architecture
- **Environment Path**: `d:\candidate_kit\.venv-vision`
- **Interpreter**: Python 3.12 (virtual environment isolated from global packages via `--no-site-packages`)
- **Creation Command**:
  ```powershell
  python -m venv d:\candidate_kit\.venv-vision
  ```

---

## 8. Exact Packages & Versions Required for Phase 6C

The following pinned packages will be installed **only** inside `.venv-vision`:

```text
# PyTorch CPU (Stable official wheel)
torch>=2.4.0
torchvision>=0.19.0

# Hugging Face Core (Qwen3-VL support requires >=4.57.0)
transformers>=4.57.0
accelerate>=1.15.0
safetensors>=0.4.5
huggingface-hub>=0.28.0

# Qwen Vision Model Utilities
qwen-vl-utils>=0.0.14

# Image & Data Processing
pillow>=10.4.0
numpy>=1.26.0,<3.0.0
psutil>=5.9.0

# Explicitly Excluded
# flash-attn (NOT supported on CPU / Windows AMD)
```

---

## 9. Verification & Stop Notice

1. **Current Codebase Status**:
   - Phase 6A (Renderer): Untouched, stable.
   - Phase 6B (OCR Pipeline): Untouched, stable.
   - `erp.py`: Untouched.
   - Test Suite: **86/86 tests passing** (verified at 13:53:52).
2. **Current Model Status**:
   - Model download: **NOT STARTED** (awaiting authorization).
   - Environment creation: **NOT STARTED** (awaiting authorization).

**STOPPING POINT**: Phase 6C-1 hardware and dependency assessment is complete. No code modifications or downloads have been executed. Awaiting user review and authorization to create the isolated environment and proceed to Phase 6C-2.
