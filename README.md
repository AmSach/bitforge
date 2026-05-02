# BitForge

BitForge shrinks a model export so it can fit on tiny hardware, then keeps it fast by doing three things well:

1. **Weight compression** — packs model weights into low-bit storage.
2. **Context trimming** — keeps the useful part of the active prompt/cache and drops the rest.
3. **Block pruning** — removes low-value blocks first when a device budget is tight.

That third part matters if you want to actually fit and run on ESP-class boards or make a Raspberry Pi demo feel fast. Tiny hardware does not want a giant active context or dead weight.

## What this is for

BitForge is useful when you want:

- a small, fast local demo
- a model export that fits a strict RAM/flash budget
- a way to prove “this runs on tiny gear” without pretending it is magic
- a faster prompt loop by trimming unnecessary context before inference

## What it can do

- compress weights with bit packing
- prune low-value blocks first when a device budget is tight
- compact prompt/context tokens so only the useful part stays active
- generate an embedded C project for ESP32, Arduino, or STM32
- run a local simulator to estimate speed and memory use
- give you a sane path to a fast Raspberry Pi proof-of-concept

## What it cannot do

- It cannot make a big model truly “free” on tiny hardware.
- It cannot turn a normal 7B model into a real-time ESP32 model.
- It cannot guarantee high tokens/sec on every board.

If you want **super fast**, the trick is to use a **small model** and then reduce everything around it: context, weights, memory churn, and pointless blocks.

## Best practical targets

- **Raspberry Pi**: good for proving the idea and getting actual speed.
- **ESP32-S3**: good for tiny demos and tight memory budgets.
- **Arduino**: only for extremely small toy examples.

## Install

```bash
pip install -e .
```

## Quick start

### 1) Compress weights

```python
import numpy as np
from bitforge import Quantizer, QuantizationConfig

weights = np.random.randn(256, 256).astype(np.float32)
q = Quantizer(QuantizationConfig(mode="adaptive"))
packed, scale, zero_point = q.quantize_tensor(weights, bits=4)
```

### 2) Trim active context

```python
from bitforge.context import ContextCompressor, ContextCompressionConfig

compressor = ContextCompressor(ContextCompressionConfig(max_tokens=128))
result = compressor.compress_tokens(list(range(500)))
print(result.compressed_tokens)
```

### 3) Prune to a hardware budget

```python
from bitforge.prune import BlockPruner, PruningConfig

pruner = BlockPruner(PruningConfig(block_size=64, target_keep_ratio=0.15))
export = pruner.prune_to_budget({"layer": weights}, budget_bytes=200_000)
print(export.compression_ratio)
```

### 4) Access it from the package root

```python
from bitforge import ContextCompressor, ContextCompressionConfig, BlockPruner, PruningConfig
```

## How to prove it runs fast

For a Raspberry Pi demo, use:

- a small model
- short context windows
- 4-bit or 2-bit weights
- block pruning for the least important layers
- a strict max-token limit

The simulator reports throughput estimates, and the pruning/context tools reduce the amount of work before the model even starts.

## CLI you can use

```bash
bitforge compress gpt2 --target esp32-s3 --bits 4 --output ./compressed_model
bitforge prune ./compressed_model --budget-bytes 200000
bitforge simulate ./compressed_model --prompt "Hello"
```

## Testing

```bash
PYTHONPATH=. pytest -q
```

## Structure

- `bitforge/compress/quantize.py` — low-bit weight packing
- `bitforge/dequantize.py` — restore packed values
- `bitforge/context.py` — keep the useful part of the active context
- `bitforge/prune.py` — prune low-value blocks to fit a budget
- `bitforge/generate/c_codegen.py` — embedded C export
- `bitforge/simulator.py` — local speed/memory simulator
- `tests/` — correctness tests

## Honest status

BitForge is now designed to be useful for real tiny-device demos, not just a flashy claim.

If you want the fastest proof, the winning strategy is:

1. use a small model
2. prune it hard
3. trim context aggressively
4. deploy to Raspberry Pi or ESP32-S3
5. keep the active window tiny so tokens/sec stays high

That’s the actual path to a convincing demo on small hardware.
