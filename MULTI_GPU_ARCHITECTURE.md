# Multi-GPU Architecture for Moshi Real-Time Speech

## Executive Summary

This branch adds **layer-sharded model parallelism** to Moshi's language model inference server, distributing transformer layers across multiple GPUs so the model can run on machines where **no single GPU has enough memory** to hold the entire model. This is a memory-capacity solution, not a throughput optimization --- GPUs execute serially, so multi-GPU inference is expected to be slower than a hypothetical single GPU large enough to hold the model, due to serial execution and cross-device transfer overhead. Before this change, the entire model had to fit on one device --- the only alternative was CPU offloading, which is too slow for real-time audio.

> **Note on terminology:** This is *not* pipeline parallelism in the classical sense (where different GPUs process different frames concurrently to increase throughput). Here, each frame passes through all GPUs sequentially --- GPU 0 is idle while GPU 1 computes and vice versa. The motivation is purely fitting the model in memory.

The core challenge was that Moshi uses **CUDA graphs** (pre-recorded GPU command sequences) to eliminate per-frame CPU overhead, and CUDA graphs fundamentally cannot contain cross-device operations. The solution splits the model across GPUs, selectively disables CUDA graphs for the cross-device transformer pass, keeps CUDA graphs enabled for the single-device depformer pass, and uses **CUDA events** for non-blocking GPU-to-GPU synchronization that adds only ~4--8 microseconds of overhead per frame.

## Background: Why This Is Hard

### The 80ms Real-Time Budget

Moshi is a streaming speech-to-speech model. Audio arrives in frames, and each frame must be processed and returned within **80 milliseconds** --- otherwise the user hears silence or glitches. This budget covers encoding audio to tokens, running the language model, and decoding tokens back to audio. There is zero room for blocking waits or inefficient synchronization.

### CUDA Graphs

To meet this budget, the single-GPU code uses **CUDA graphs**. Normally, each PyTorch operation issues a separate GPU command through the CUDA driver, and the CPU must orchestrate every step. A CUDA graph records an entire sequence of GPU operations once, then replays the whole thing with a single CPU call. This eliminates thousands of microseconds of per-frame CPU overhead.

The catch: a CUDA graph is a recording of operations **on a single device, on a single CUDA stream**. You cannot record a `tensor.to(other_device)` call inside a graph --- the CUDA runtime will raise `cudaErrorStreamCaptureIsolation`. This is a hard constraint from NVIDIA, not a PyTorch limitation.

### Single-Device Assumption

The original code assumes everything lives on one GPU. The `LMModel` class places all parameters on a single `device`, and `LMGen` (the streaming inference wrapper) wraps `forward_codes`, `forward_embeddings`, and `depformer_step` each in a `CUDAGraphed` wrapper. There is no concept of tensors moving between devices during a forward pass.

## The Architecture Before (Single GPU)

The inference pipeline on `main` looks like this:

```
Audio Frame
    |
    v
[Mimi Encoder] -- encodes audio to tokens
    |
    v
[LMModel.forward_codes]     -- embed tokens (CUDAGraphed)
    |
    v
[LMModel.forward_embeddings] -- transformer layers 0..N (CUDAGraphed)
    |
    v
[LMGen.depformer_step]       -- depformer loop (CUDAGraphed)
    |
    v
[Mimi Decoder] -- decode tokens to audio
    |
    v
Audio Output
```

All three `CUDAGraphed` wrappers capture and replay on the same device. The `CUDAGraphed` class (in `compile.py`) records the function's GPU operations once, then on subsequent calls copies new inputs into the recorded buffers and replays the graph. This is the fastest possible execution path on a single GPU.

Key code path (`lm.py:715-719` on `main`):
```python
disable = lm_model.device.type != 'cuda'
graphed_main = CUDAGraphed(lm_model.forward_codes, disable=disable)
graphed_embeddings = CUDAGraphed(lm_model.forward_embeddings, disable=disable)
graphed_depth = CUDAGraphed(self.depformer_step, disable=disable)
```

All three are graphed or all three are not --- there's no per-function decision.

## What Changed: Layer-Sharded Model Parallelism

### New Execution Model (Multi-GPU)

```
Audio Frame
    |
    v
[Mimi Encoder]  (cuda:0)
    |
    v
[Embedding]     (cuda:0)  -- embed_codes()
    |
    v
[Layers 0..15]  (cuda:0)  -- forward_embeddings() with CUDA event sync
    |                         (NOT CUDAGraphed -- has cross-device transfers)
    | --- CUDA event record on cuda:0, event wait on cuda:1 ---
    | --- tensor.to(cuda:1, non_blocking=True) ---
    v
[Layers 16..31] (cuda:1)
    |
    v
[Out Norm + Text Linear] (cuda:1)
    |
    +--> text_logits.to(cuda:0) for sampling  (via CUDA events)
    |
    v
[Depformer]     (cuda:1)  -- depformer_step() (CUDAGraphed!)
    |                         Inputs pre-transferred before graph entry
    v
sampled_tokens.to(cuda:0)  -- small tensor, ~64 bytes
    |
    v
[Mimi Decoder]  (cuda:0)
```

The transformer layers are split across GPUs. The depformer and its associated parameters (depformer embeddings, linear heads) live on the **last GPU** --- the same device where the transformer output lands. This is a deliberate placement: the transformer output tensor is ~16KB and transferring it back to cuda:0 just to run the depformer would waste time. Instead, only the final sampled tokens (~64 bytes) are transferred back.

### CUDA Graph Strategy

| Function | Single-GPU | Multi-GPU | Why |
|---|---|---|---|
| `forward_codes` | CUDAGraphed | **Not** graphed | Contains `embed_codes` which transfers sequence to primary device |
| `forward_embeddings` | CUDAGraphed | **Not** graphed | Transfers tensors between GPUs at layer boundaries |
| `depformer_step` | CUDAGraphed | **CUDAGraphed** | All inputs pre-transferred to depformer device *before* entering the graph |

The depformer optimization is key: rather than disabling CUDA graphs entirely, the code moves all device transfers *outside* the graphed function. In `lm.py:896-907`:

```python
# Pre-transfer inputs to depformer device BEFORE calling the graphed function
depformer_device = getattr(lm_model, '_depformer_device', lm_model.device)
transformer_out_dep = transformer_out.to(depformer_device, non_blocking=True)
target_dep = target_[:, lm_model.audio_offset:, 0].to(depformer_device, non_blocking=True)
provided_dep = provided_[:, lm_model.audio_offset:, 0].to(depformer_device, non_blocking=True)
next_text_token_dep = next_text_token.to(depformer_device, non_blocking=True)

# This call IS CUDAGraphed -- no cross-device ops inside
sampled_audio_tokens = state.graphed_depth(next_text_token_dep, transformer_out_dep, target_dep, provided_dep)
```

### CUDA Event Synchronization

The `MultiGPULMModel.forward_embeddings()` method implements GPU-to-GPU synchronization using CUDA events. Here's why this matters and what the alternatives are:

**Option 1: `torch.cuda.synchronize()`** --- blocks the CPU until all GPU work finishes. Measured at 18--90ms per frame. This alone can exceed the 80ms budget.

**Option 2: `non_blocking=True` only** --- the CPU never waits, but neither does the destination GPU. GPU 1 may start computing on data that GPU 0 hasn't finished transferring. This causes silent data corruption (data races).

**Option 3: CUDA events (this implementation)** --- the source GPU records an event when it finishes. The destination GPU waits for that event before starting. The CPU never blocks. Overhead is ~4--8 microseconds.

The implementation in `loaders.py` (inside `MultiGPULMModel`):

```python
# Each GPU gets a dedicated stream and event
self._gpu_streams = {}      # one Stream per GPU
self._boundary_events = {}  # one Event per GPU

# At each device boundary during forward_embeddings():
with torch.cuda.stream(current_stream):
    self._boundary_events[str(current_device)].record()   # GPU A done

current_stream = self._gpu_streams[str(new_device)]
with torch.cuda.stream(current_stream):
    self._boundary_events[str(prev_device)].wait()        # GPU B waits for A
    x = x.to(new_device, non_blocking=True)               # transfer tensor
```

## Why It Works

The multi-GPU design exists because the model doesn't fit on a single GPU. It trades execution speed for memory capacity --- serial execution across GPUs plus cross-device transfer overhead means this will always be slower than running on a single GPU with sufficient VRAM. The goal is to meet the 80ms real-time budget, not to outperform single-GPU inference.

The key insight is that you can split the problem into two parts:

1. **Transformer forward pass** (cross-device) --- cannot be CUDA-graphed, but the overhead of issuing individual PyTorch ops is acceptable because transformer layers are computationally heavy. The GPU compute time dominates the CPU dispatch time.

2. **Depformer loop** (single-device) --- can and should be CUDA-graphed, because the depformer runs many small sequential steps (one per audio codebook, typically 8 iterations) where CPU dispatch overhead would be significant.

By pre-transferring all depformer inputs *outside* the graph and passing `skip_transfer=True` into `forward_depformer`, the graphed depformer sees a purely single-device computation.

The depformer placement on the last GPU is the other critical decision. The transformer output is the largest tensor that needs to cross a device boundary. By placing the depformer where the transformer output already lives, only the small sampled token tensor (~64 bytes) needs to transfer back to cuda:0.

## Why It Wasn't Possible Before

Three blockers had to be solved simultaneously:

### 1. CUDA Graphs Were All-or-Nothing

The original code used a single `disable` flag for all three graphed functions. There was no way to say "graph the depformer but not the transformer." The fix: separate `disable_cross_device` (for cross-device functions) from the existing device-type check (for depformer).

### 2. Device Transfers Inside Graphed Functions

The depformer received tensors from the transformer on the primary device and assumed they were already on the correct device. In multi-GPU mode, the transformer output is on the last GPU. If the depformer tried to do `.to(depformer_device)` inside the graphed function, CUDA graph capture would fail. The fix: pre-transfer all inputs outside the graph boundary, add `skip_transfer` parameter to `forward_depformer`.

### 3. No Cross-Device Synchronization Mechanism

The original code had exactly one synchronization point: `torch.cuda.synchronize()` in the server's warmup. There was no mechanism for ordered data movement between GPUs during inference. The fix: CUDA events with per-GPU streams in `MultiGPULMModel`, plus `wait_stream()` calls to synchronize with default streams before the depformer executes.

## Implementation Details

### File-by-File Walkthrough

#### `moshi/moshi/models/loaders.py`

**Lines of change:** ~500 added, ~90 removed (mostly deduplication).

1. **`_patch_state_dict()` (new function)** --- Extracted from the duplicated weight-patching logic that existed in both `get_moshi_lm()` and `_get_moshi_lm_with_offload()`. Handles two cases: expanding depformer self-attention weights when shapes don't match, and copying weights from layers 0--7 to layers 8--15 for certain parameter groups. This is a pure refactor --- no behavioral change.

2. **`get_moshi_lm()` signature** --- Added `multi_gpu: bool = False` and `gpus: int | None = None` parameters. When `multi_gpu=True`, dispatches to `_get_moshi_lm_multi_gpu()` early.

3. **`_get_moshi_lm_multi_gpu()` (new function)** --- The core multi-GPU loader:
   - Creates the model on CPU
   - Loads and patches weights
   - Computes layer-to-GPU assignments: `gpu_idx = min(i // layers_per_gpu, num_gpus - 1)`
   - Moves each transformer layer to its assigned GPU
   - Places embeddings on cuda:0 (primary device)
   - Places depformer, output norm, text linear, and all depformer-related modules on the last GPU
   - Wraps the model in `MultiGPULMModel`

4. **`MultiGPULMModel` (new class, ~200 lines)** --- A `StreamingContainer` subclass that wraps the real `LMModel` and intercepts forward passes to handle device transitions:
   - **`forward_embeddings()`**: Reimplements the transformer forward pass with explicit device transitions and CUDA event synchronization. Each layer executes within a `torch.cuda.stream()` context on its GPU's dedicated stream. At device boundaries, records an event on the source stream and waits for it on the destination stream.
   - **`forward_depformer()`**: Delegates to the underlying model's `forward_depformer()`, optionally transferring inputs to the depformer device. Accepts `skip_transfer=True` when called from the CUDA-graphed depformer step.
   - **`embed_codes()`**: Transfers input sequence to primary device before embedding.
   - **`__getattr__()`**: Proxies attribute access to the underlying `_model` for all attributes not explicitly defined on the wrapper (e.g., `dep_q`, `audio_offset`, `card`).
   - **`parameters()`**: Yields embedding parameters first so `next(iter(model.parameters())).device` returns `cuda:0`, which downstream code uses for device detection.
   - Flags `_is_multi_gpu = True` and `_depformer_device` for use by `LMGen`.

#### `moshi/moshi/models/lm.py`

**Lines of change:** ~40 added.

1. **`_init_streaming_state()` (lines 718--726)** --- Replaced single `disable` flag with two:
   - `disable_cross_device`: disables CUDA graphs for `forward_codes` and `forward_embeddings` when `_is_multi_gpu` is set
   - Depformer remains graphed (only disabled for non-CUDA devices)

2. **Inside `LMGen.process_transformer_output()` (around line 899)** --- Added a pre-transfer block that moves `transformer_out`, `target_`, `provided_`, and `next_text_token` to the depformer device *before* calling `state.graphed_depth()`. Also transfers results back to primary device after.

3. **`depformer_step()` (lines 1155--1170)** --- Added `skip_transfer=True` to the `forward_depformer()` call, with a docstring explaining that inputs are pre-transferred by the caller.

#### `moshi/moshi/utils/compile.py`

**Lines of change:** ~40 added.

1. **`_get_device_from_args()` (new function)** --- Inspects function arguments to find the first CUDA tensor and returns its device. Used to ensure CUDA graph capture happens on the correct GPU (not necessarily cuda:0).

2. **`CUDAGraphed.__call__()` (lines 291--325)** --- Wrapped the graph capture logic in a `torch.cuda.device(device)` context manager so that `CUDAGraph()` and `Stream()` are created on the correct device. Added explicit stream creation and synchronization for the capture: `cuda.current_stream(device).synchronize()` before capture prevents "CUDA Graph is empty" warnings on secondary GPUs where no prior work may have been queued.

#### `moshi/moshi/server.py`

**Lines of change:** ~30 added.

1. **CLI arguments** --- Added `--multi-gpu` (boolean flag) and `--gpus` (optional int) to the argument parser. Passed through to `loaders.get_moshi_lm()`.

2. **Warmup synchronization (lines 131--134)** --- Changed from `torch.cuda.synchronize()` (single device) to a loop that synchronizes all CUDA devices, which is necessary when model parameters span multiple GPUs.

3. **Performance logging (lines 221--248)** --- Added per-step timing that logs a warning when any frame exceeds the 80ms budget, broken down into encode, LM step, and decode components. This is diagnostic instrumentation, not part of the multi-GPU logic itself, but is essential for validating that multi-GPU inference meets real-time constraints.

## Known Limitations

### Load Imbalance (Configurable via `--depformer-cost-ratio`)

The last GPU carries more work than just its share of transformer layers --- it also runs `out_norm`, `text_linear`, the 6-layer depformer, depformer embeddings, and depformer linear heads. To compensate, the layer split is weighted using `--depformer-cost-ratio` (default 0.5).

The algorithm treats each depformer layer as a fraction of a transformer layer and distributes work accordingly:

```
depformer_equivalent = int(num_depformer_layers * depformer_cost_ratio)
total_work = num_transformer_layers + depformer_equivalent
work_per_gpu = total_work / num_gpus
layers_for_last_gpu = round(work_per_gpu - depformer_equivalent)
```

Examples (32 transformer layers, 6 depformer layers, 2 GPUs):
- `--depformer-cost-ratio 0.0` → 16/16 split (even, ignores depformer cost)
- `--depformer-cost-ratio 0.5` → 17/15 split (default)
- `--depformer-cost-ratio 1.0` → 19/13 split (aggressive)

**Per-GPU timing metrics** are logged every 100 frames at INFO level, showing mean/max transformer time per GPU, mean/max depformer time, and a GPU idle ratio (`abs(gpu0 - gpu1) / max(gpu0, gpu1)`). An idle ratio approaching 0 means balanced. Use these metrics to tune the cost ratio for your hardware.

### Serial GPU Execution

GPU 0 is idle while GPU 1 runs, and vice versa. This is inherent to the memory-motivated design --- the goal is fitting the model across GPUs, not achieving speedup over a hypothetical single large GPU. Utilization of each individual GPU will be well below 100%.

## Usage

```bash
# 2 GPUs (auto-detect)
python -m moshi.server --multi-gpu

# Limit to specific number of GPUs
python -m moshi.server --multi-gpu --gpus 2

# Single GPU with CPU offload (fallback for machines with 1 GPU)
python -m moshi.server --cpu-offload
```
