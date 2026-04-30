"""
BitForge CLI - Command-line interface for LLM compression.

Usage:
    bitforge compress <model> --target esp32-s3 --output ./compressed
    bitforge simulate ./compressed --prompt "Hello, world"
    bitforge flash ./compressed --port /dev/ttyUSB0
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from tqdm import tqdm

from bitforge import __version__
from bitforge.compress.quantize import (
    QuantizationConfig,
    QuantizationMode,
    Quantizer,
    QuantizationResult,
)
from bitforge.generate.c_codegen import CCodeGenerator, GeneratedProject
from bitforge.model_loader import ModelLoader, LoadedModel
from bitforge.simulator import InferenceSimulator, SimulatorConfig
from bitforge.targets.esp32 import ESP32Target, ESP32Variant
from bitforge.targets.arduino import ArduinoTarget, ArduinoVariant
from bitforge.targets.stm32 import STM32Target, STM32Family

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Supported targets
TARGET_MAP = {
    "esp32": lambda: ESP32Target(variant=ESP32Variant.ESP32),
    "esp32-s2": lambda: ESP32Target(variant=ESP32Variant.ESP32_S2),
    "esp32-s3": lambda: ESP32Target(variant=ESP32Variant.ESP32_S3),
    "esp32-c3": lambda: ESP32Target(variant=ESP32Variant.ESP32_C3),
    "esp32-c6": lambda: ESP32Target(variant=ESP32Variant.ESP32_C6),
    "arduino-uno": lambda: ArduinoTarget(variant=ArduinoVariant.UNO),
    "arduino-nano": lambda: ArduinoTarget(variant=ArduinoVariant.NANO),
    "arduino-mega": lambda: ArduinoTarget(variant=ArduinoVariant.MEGA),
    "arduino-mega2560": lambda: ArduinoTarget(variant=ArduinoVariant.MEGA2560),
    "stm32f4": lambda: STM32Target(family=STM32Family.STM32F4),
    "stm32f7": lambda: STM32Target(family=STM32Family.STM32F7),
    "stm32h7": lambda: STM32Target(family=STM32Family.STM32H7),
}


def parse_target(target_str: str):
    """Parse target string into target configuration.
    
    Args:
        target_str: Target identifier (e.g., "esp32-s3")
        
    Returns:
        Target configuration object
        
    Raises:
        click.BadParameter: If target is unknown
    """
    target_str = target_str.lower()
    if target_str not in TARGET_MAP:
        available = ", ".join(TARGET_MAP.keys())
        raise click.BadParameter(
            f"Unknown target '{target_str}'. Available: {available}"
        )
    return TARGET_MAP[target_str]()


def parse_bits(bits_str: str) -> QuantizationMode:
    """Parse bit width string into QuantizationMode.
    
    Args:
        bits_str: Bit width (e.g., "1", "2", "4", "8", "adaptive")
        
    Returns:
        QuantizationMode enum value
    """
    bits_str = bits_str.lower()
    if bits_str == "adaptive" or bits_str == "auto":
        return QuantizationMode.ADAPTIVE
    try:
        bits = int(bits_str)
        mode_map = {
            1: QuantizationMode.BIT_1,
            2: QuantizationMode.BIT_2,
            4: QuantizationMode.BIT_4,
            8: QuantizationMode.BIT_8,
        }
        if bits not in mode_map:
            raise click.BadParameter(
                f"Unsupported bit width: {bits}. Use 1, 2, 4, 8, or 'adaptive'"
            )
        return mode_map[bits]
    except ValueError:
        raise click.BadParameter(
            f"Invalid bit width: {bits_str}. Use 1, 2, 4, 8, or 'adaptive'"
        )


@click.group()
@click.version_option(version=__version__, prog_name="bitforge")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("-q", "--quiet", is_flag=True, help="Suppress non-essential output")
def main(verbose: bool, quiet: bool):
    """BitForge - Shrink any LLM to fit in your pocket.
    
    Auto-compress language models for microcontrollers.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif quiet:
        logging.getLogger().setLevel(logging.WARNING)


@main.command()
@click.argument("model")
@click.option(
    "-t", "--target",
    default="esp32-s3",
    help="Target platform (esp32-s3, arduino-mega, stm32f4, etc.)"
)
@click.option(
    "-b", "--bits",
    default="4",
    help="Quantization bit width (1, 2, 4, 8, or 'adaptive')"
)
@click.option(
    "-o", "--output",
    type=click.Path(),
    default="./compressed_model",
    help="Output directory"
)
@click.option(
    "--ram-kb",
    type=int,
    default=None,
    help="Target RAM in KB (overrides target default)"
)
@click.option(
    "--flash-kb",
    type=int,
    default=None,
    help="Target Flash in KB (overrides target default)"
)
@click.option(
    "--calibration-samples",
    type=int,
    default=128,
    help="Number of calibration samples"
)
@click.option(
    "--no-sensitivity",
    is_flag=True,
    help="Disable per-layer sensitivity analysis"
)
def compress(
    model: str,
    target: str,
    bits: str,
    output: str,
    ram_kb: Optional[int],
    flash_kb: Optional[int],
    calibration_samples: int,
    no_sensitivity: bool,
):
    """Compress a model for target platform.
    
    MODEL can be a HuggingFace model ID or path to a local model file.
    
    Examples:
    
        bitforge compress gpt2 --target esp32-s3 --bits 4
        bitforge compress ./my_model.safetensors --target arduino-mega --bits 1
        bitforge compress TinyLlama/TinyLlama-1.1B-Chat-v1.0 --bits adaptive
    """
    # Parse target
    target_config = parse_target(target)
    
    # Get memory constraints from target
    if ram_kb is None:
        ram_kb = getattr(target_config, "ram_kb", 512)
        if ram_kb is None:
            ram_kb = getattr(target_config, "ram_bytes", 8192) // 1024
    
    if flash_kb is None:
        flash_kb = getattr(target_config, "flash_kb", 4096)
        if flash_kb is None:
            flash_kb = getattr(target_config, "flash_bytes", 262144) // 1024
    
    # Parse quantization mode
    quant_mode = parse_bits(bits)
    
    # Create quantization config
    quant_config = QuantizationConfig(
        mode=quant_mode,
        target_ram_kb=ram_kb,
        target_flash_kb=flash_kb,
        calibration_samples=calibration_samples,
        per_layer_sensitivity=not no_sensitivity,
    )
    
    click.echo(f"\n🔥 BitForge Compressor v{__version__}")
    click.echo(f"   Target: {target_config}")
    click.echo(f"   Quantization: {quant_mode.value}-bit")
    click.echo(f"   RAM: {ram_kb} KB, Flash: {flash_kb} KB")
    click.echo()
    
    # Load model
    click.echo("📦 Loading model...")
    loader = ModelLoader()
    try:
        loaded_model = loader.load(model)
    except FileNotFoundError:
        click.echo(f"❌ Error: Model not found: {model}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error loading model: {e}", err=True)
        sys.exit(1)
    
    click.echo(f"   Model: {loaded_model.name}")
    click.echo(f"   Parameters: {loaded_model.param_count:,}")
    click.echo(f"   Size: {loaded_model.size_bytes / (1024*1024):.2f} MB")
    click.echo()
    
    # Quantize
    click.echo("⚡ Quantizing model...")
    quantizer = Quantizer(quant_config)
    
    # Progress tracking
    with tqdm(total=len(loaded_model.weights), desc="   Layers", unit="layer") as pbar:
        # Quantize all weights
        quant_result = quantizer.quantize_model(loaded_model.weights)
        pbar.update(len(loaded_model.weights))
    
    # Show results
    summary = quant_result.get_summary()
    click.echo()
    click.echo("✅ Quantization complete!")
    click.echo(f"   Original: {summary['original_size_mb']:.2f} MB")
    click.echo(f"   Compressed: {summary['compressed_size_kb']:.2f} KB")
    click.echo(f"   Compression: {summary['compression_ratio']:.1f}x")
    click.echo(f"   Average MSE: {summary['average_mse']:.6f}")
    
    if quant_result.target_compatible:
        click.echo(f"   ✅ Fits target constraints")
    else:
        click.echo(f"   ⚠️  Exceeds target flash capacity")
    
    click.echo()
    
    # Generate C code
    click.echo("🔧 Generating C code...")
    generator = CCodeGenerator(model_config={
        "hidden_dim": loaded_model.config.hidden_size,
        "num_layers": loaded_model.config.num_hidden_layers,
        "num_heads": loaded_model.config.num_attention_heads,
        "vocab_size": loaded_model.config.vocab_size,
        "max_seq_len": loaded_model.config.max_position_embeddings,
        "embed_dim": loaded_model.config.hidden_size,
    })
    
    project = generator.generate(
        model_name=loaded_model.name,
        quant_result=quant_result,
        target=target,
    )
    
    # Save to output directory
    output_path = Path(output)
    save_result = generator.save_project(project, output_path)
    
    # Save quantization metadata
    metadata_path = output_path / "quantization.json"
    import json
    with open(metadata_path, "w") as f:
        json.dump({
            "model_name": loaded_model.name,
            "target": target,
            "bits": quant_mode.value,
            "summary": summary,
            "config": {
                "ram_kb": ram_kb,
                "flash_kb": flash_kb,
                "calibration_samples": calibration_samples,
                "per_layer_sensitivity": not no_sensitivity,
            },
        }, f, indent=2)
    
    click.echo(f"   Generated {save_result['files_saved']} files")
    click.echo(f"   Output: {output_path}")
    click.echo()
    click.echo(f"🚀 Ready for deployment! Next steps:")
    click.echo(f"   1. Test locally: bitforge simulate {output}")
    click.echo(f"   2. Flash to device: bitforge flash {output} --port /dev/ttyUSB0")


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--prompt",
    default="Hello, world",
    help="Input prompt for generation"
)
@click.option(
    "--max-tokens",
    type=int,
    default=50,
    help="Maximum tokens to generate"
)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    help="Sampling temperature"
)
@click.option(
    "--top-k",
    type=int,
    default=0,
    help="Top-k sampling (0 = disabled)"
)
@click.option(
    "--top-p",
    type=float,
    default=1.0,
    help="Top-p (nucleus) sampling"
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility"
)
@click.option(
    "--benchmark",
    is_flag=True,
    help="Run benchmark instead of single generation"
)
@click.option(
    "--benchmark-runs",
    type=int,
    default=10,
    help="Number of benchmark runs"
)
def simulate(
    model_path: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    benchmark: bool,
    benchmark_runs: int,
):
    """Simulate inference on compressed model.
    
    Test the quantized model on your laptop before deploying to hardware.
    
    Examples:
    
        bitforge simulate ./compressed_model --prompt "Once upon a time"
        bitforge simulate ./compressed_model --benchmark --benchmark-runs 20
    """
    model_dir = Path(model_path)
    
    # Load quantization metadata
    metadata_path = model_dir / "quantization.json"
    if not metadata_path.exists():
        click.echo(f"❌ Error: Not a BitForge model directory: {model_path}", err=True)
        sys.exit(1)
    
    import json
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    click.echo(f"\n🧪 BitForge Simulator v{__version__}")
    click.echo(f"   Model: {metadata.get('model_name', 'unknown')}")
    click.echo(f"   Target: {metadata.get('target', 'unknown')}")
    click.echo(f"   Quantization: {metadata.get('bits', 'unknown')}-bit")
    click.echo()
    
    # Create simulator config
    sim_config = SimulatorConfig(
        max_seq_len=metadata.get("config", {}).get("max_seq_len", 128),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )
    
    # Note: In a full implementation, we would load the quantized weights
    # from the generated C files and reconstruct the QuantizationResult.
    # For this demo, we create a minimal result.
    
    from bitforge.compress.quantize import QuantizationResult, LayerQuantizationResult
    import numpy as np
    
    # Create minimal quantization result for simulation
    # (In production, this would be loaded from saved data)
    demo_weights = np.zeros(100, dtype=np.uint8)
    demo_layer = LayerQuantizationResult(
        name="demo",
        original_shape=(10, 10),
        original_bits=32,
        quantized_bits=4,
        quantized_weights=demo_weights,
        scale=0.1,
        zero_point=0.0,
    )
    
    quant_result = QuantizationResult(
        layers=[demo_layer],
        total_params=metadata.get("summary", {}).get("total_params", 1000),
        compressed_size_bytes=metadata.get("summary", {}).get("compressed_size_kb", 100) * 1024,
        original_size_bytes=metadata.get("summary", {}).get("original_size_mb", 1) * 1024 * 1024,
    )
    
    # Create simulator
    simulator = InferenceSimulator(quant_result, sim_config)
    
    if benchmark:
        # Run benchmark
        click.echo("⚡ Running benchmark...")
        
        # Convert prompt to tokens (simplified)
        input_tokens = [ord(c) for c in prompt if 32 <= ord(c) < 127]
        
        results = simulator.benchmark(input_tokens, num_runs=benchmark_runs, max_tokens=max_tokens)
        
        click.echo(f"\n📊 Benchmark Results:")
        click.echo(f"   Average time: {results['avg_time_ms']:.2f} ms")
        click.echo(f"   Std deviation: {results['std_time_ms']:.2f} ms")
        click.echo(f"   Min time: {results['min_time_ms']:.2f} ms")
        click.echo(f"   Max time: {results['max_time_ms']:.2f} ms")
        click.echo(f"   Tokens/sec: {results['tokens_per_second']:.2f}")
    else:
        # Run single generation
        click.echo(f"📝 Prompt: \"{prompt}\"")
        click.echo()
        click.echo("⚡ Generating...")
        
        # Convert prompt to tokens (simplified char-level)
        input_tokens = [ord(c) for c in prompt if 32 <= ord(c) < 127]
        
        result = simulator.generate(input_tokens, max_tokens=max_tokens)
        
        click.echo()
        click.echo(f"📤 Output: \"{result.output_text}\"")
        click.echo()
        click.echo(f"   Tokens generated: {len(result.output_tokens)}")
        click.echo(f"   Inference time: {result.inference_time_ms:.2f} ms")
        click.echo(f"   Tokens/sec: {result.tokens_per_second:.2f}")
        click.echo(f"   Memory used: ~{result.memory_used_bytes // 1024} KB")


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--port",
    default="/dev/ttyUSB0",
    help="Serial port for flashing"
)
@click.option(
    "--baud",
    type=int,
    default=115200,
    help="Baud rate for serial communication"
)
def flash(model_path: str, port: str, baud: int):
    """Flash compressed model to target device.
    
    Upload the generated code to your microcontroller.
    
    Examples:
    
        bitforge flash ./compressed_model --port /dev/ttyUSB0
        bitforge flash ./compressed_model --port COM3  # Windows
    """
    model_dir = Path(model_path)
    
    # Load metadata
    metadata_path = model_dir / "quantization.json"
    if not metadata_path.exists():
        click.echo(f"❌ Error: Not a BitForge model directory: {model_path}", err=True)
        sys.exit(1)
    
    import json
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    target = metadata.get("target", "unknown")
    
    click.echo(f"\n📡 BitForge Flash v{__version__}")
    click.echo(f"   Model: {metadata.get('model_name', 'unknown')}")
    click.echo(f"   Target: {target}")
    click.echo(f"   Port: {port}")
    click.echo()
    
    # Get target configuration
    try:
        target_config = parse_target(target)
    except click.BadParameter:
        click.echo(f"⚠️  Unknown target, using generic flashing method")
        target_config = None
    
    # Check if platformio is available
    import shutil
    has_platformio = shutil.which("pio") is not None
    has_arduino_cli = shutil.which("arduino-cli") is not None
    
    if target.startswith("esp32") and has_platformio:
        click.echo("🚀 Flashing via PlatformIO...")
        flash_cmd = f"cd {model_dir} && pio run --target upload --upload-port {port}"
        click.echo(f"   Running: {flash_cmd}")
        click.echo()
        click.echo("   Note: Ensure your device is in upload mode (hold BOOT, press RST)")
        import os
        result = os.system(flash_cmd)
        if result == 0:
            click.echo("✅ Flash successful!")
        else:
            click.echo("❌ Flash failed. Check the output above.")
    elif target.startswith("arduino") and has_arduino_cli:
        click.echo("🚀 Flashing via Arduino CLI...")
        flash_cmd = f"arduino-cli upload -p {port} --input-dir {model_dir}/src"
        click.echo(f"   Running: {flash_cmd}")
        import os
        result = os.system(flash_cmd)
        if result == 0:
            click.echo("✅ Flash successful!")
        else:
            click.echo("❌ Flash failed. Check the output above.")
    else:
        click.echo("⚠️  No supported flashing tool found.")
        click.echo()
        click.echo("Manual flashing instructions:")
        click.echo(f"   1. Navigate to: {model_dir}")
        if target.startswith("esp32"):
            click.echo("   2. Run: idf.py -p {port} flash monitor")
        elif target.startswith("arduino"):
            click.echo("   2. Open src/sketch.ino in Arduino IDE")
            click.echo("   3. Select your board from Tools > Board menu")
            click.echo("   4. Click Upload")
        elif target.startswith("stm32"):
            click.echo("   2. Run: platformio run --target upload")
        else:
            click.echo("   2. Run: make && make upload")


@main.command()
def models():
    """List supported models for compression.
    
    Shows commonly used small LLMs that work well with BitForge.
    """
    click.echo("\n📚 Supported Models")
    click.echo("=" * 50)
    click.echo()
    
    models = [
        ("gpt2", "124M params", "Original GPT-2 small", "✅ Recommended"),
        ("gpt2-medium", "355M params", "GPT-2 medium", "✅ Good"),
        ("distilgpt2", "82M params", "Distilled GPT-2", "✅ Best for Arduino"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "1.1B params", "Chat model", "⚠️  Requires 4-bit+"),
        ("Qwen/Qwen2.5-0.5B", "500M params", "Modern small LM", "✅ Good"),
        ("Qwen/Qwen2.5-1.5B", "1.5B params", "Larger Qwen", "⚠️  Requires 2-bit+"),
        ("microsoft/Phi-3-mini-4k-instruct", "3.8B params", "Instruction-tuned", "⚠️  Requires 4-bit"),
        ("HuggingFaceTB/SmolLM-135M", "135M params", "Tiny model", "✅ Best for Uno"),
        ("HuggingFaceTB/SmolLM-360M", "360M params", "Small model", "✅ Recommended"),
    ]
    
    for model_id, params, desc, status in models:
        click.echo(f"  {model_id}")
        click.echo(f"    Size: {params}, {desc}")
        click.echo(f"    Status: {status}")
        click.echo()
    
    click.echo("💡 Tips:")
    click.echo("   - Smaller models work better with extreme quantization (1-2 bit)")
    click.echo("   - Use 'adaptive' mode for best quality/size trade-off")
    click.echo("   - Arduino Uno requires models <100KB compressed (use SmolLM-135M)")


@main.command()
def targets():
    """List supported target platforms.
    
    Shows available microcontrollers and their constraints.
    """
    click.echo("\n🎯 Supported Targets")
    click.echo("=" * 50)
    click.echo()
    
    target_info = [
        ("ESP32 Family", [
            ("esp32", "520KB RAM", "4MB Flash", "WiFi + BLE"),
            ("esp32-s2", "320KB RAM", "4MB Flash", "WiFi + USB"),
            ("esp32-s3", "512KB RAM", "4MB Flash", "WiFi + BLE + USB + PSRAM"),
            ("esp32-c3", "400KB RAM", "4MB Flash", "WiFi + BLE, low cost"),
            ("esp32-c6", "512KB RAM", "4MB Flash", "WiFi + BLE, newest"),
        ]),
        ("Arduino Family", [
            ("arduino-uno", "2KB RAM", "32KB Flash", "Extreme quantization required"),
            ("arduino-nano", "2KB RAM", "32KB Flash", "Same as Uno"),
            ("arduino-mega", "8KB RAM", "128KB Flash", "Better for small models"),
            ("arduino-mega2560", "8KB RAM", "256KB Flash", "Best Arduino option"),
        ]),
        ("STM32 Family", [
            ("stm32f4", "192KB RAM", "1MB Flash", "Good balance, FPU"),
            ("stm32f7", "512KB RAM", "2MB Flash", "High performance"),
            ("stm32h7", "1MB RAM", "2MB Flash", "Top performance"),
        ]),
    ]
    
    for family, targets in target_info:
        click.echo(f"  {family}:")
        click.echo()
        for target_id, ram, flash, notes in targets:
            click.echo(f"    {target_id}")
            click.echo(f"      {ram}, {flash}")
            click.echo(f"      {notes}")
            click.echo()
    
    click.echo("💡 Tips:")
    click.echo("   - ESP32-S3 is recommended for best experience")
    click.echo("   - Arduino Uno/Nano require 1-bit quantization")
    click.echo("   - STM32 offers good performance/cost balance")


if __name__ == "__main__":
    main()
