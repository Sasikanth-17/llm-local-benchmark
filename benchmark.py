import time
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import csv
import os
import logging
import platform
import numpy as np
from huggingface_hub import login
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_system_info():
    """Log system information for reproducibility."""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "cpu": platform.processor(),
        "ram_gb": psutil.virtual_memory().total / 1024**3,
        "disk_free_gb": psutil.disk_usage('.').free / 1024**3,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU",
    }
    if info["gpu"] == "No GPU":
        logger.warning("No GPU detected. Disabling quantization and using max_new_tokens=50.")
    if info["disk_free_gb"] < 20:
        logger.warning(f"Low disk space ({info['disk_free_gb']:.2f} GB). Deleting model cache after each run.")
    return info

def get_memory_usage():
    """Return current process memory usage in MB."""
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 / 1024
    logger.debug(f"Current memory usage: {mem:.2f} MB")
    return mem

def clear_model_cache(model_name):
    """Delete model cache to free disk space."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    model_cache = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
    if os.path.exists(model_cache):
        shutil.rmtree(model_cache)
        logger.info(f"Deleted cache for {model_name} at {model_cache}")
    else:
        logger.debug(f"No cache found for {model_name}")

def load_model(model_name, use_quantization=True):
    """Load model and tokenizer with optional 4-bit quantization."""
    logger.info(f"Loading model: {model_name}")
    try:
        # Check disk space
        disk_free = psutil.disk_usage('.').free / 1024**3
        if disk_free < 10:
            logger.error(f"Insufficient disk space ({disk_free:.2f} GB). Need ~20GB for {model_name}. Skipping.")
            return None, None, 0
        
        # Authenticate with Hugging Face token
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.error(f"HF_TOKEN not set. Required for {model_name}. Set it via `set HF_TOKEN=your_token`.")
            return None, None, 0
        login(hf_token)
        logger.info("Authenticated with Hugging Face token.")
        
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Disable quantization on CPU
        use_quantization = use_quantization and torch.cuda.is_available()
        if not use_quantization:
            logger.info(f"Quantization disabled for {model_name} (CPU).")
        
        # Check available memory
        available_mem = psutil.virtual_memory().available / 1024**3
        if available_mem < 10:
            logger.warning(f"Low memory ({available_mem:.2f} GB). May cause OOM for {model_name}.")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_4bit=use_quantization,
            torch_dtype=torch.float16
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s")
        return model, tokenizer, load_time
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        return None, None, 0

def run_inference(model, tokenizer, prompt, max_new_tokens=50):
    """Run inference and measure latency and throughput."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        end_time = time.time()
        
        generated_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
        time_taken = end_time - start_time
        tpm = (generated_tokens / time_taken) * 60 if time_taken > 0 else 0
        tps = generated_tokens / time_taken if time_taken > 0 else 0
        return tpm, tps
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        return 0, 0

def benchmark_model(model_name, prompt, iterations=5):
    """Benchmark a single model and return metrics."""
    results = []
    model, tokenizer, load_time = load_model(model_name)
    if model is None or tokenizer is None:
        logger.error(f"Skipping {model_name} due to load failure.")
        return results
    
    max_new_tokens = 50  # Fixed for CPU
    logger.info(f"Using max_new_tokens={max_new_tokens} for {model_name}")
    
    for _ in tqdm(range(iterations), desc=f"Benchmarking {model_name}"):
        memory_before = get_memory_usage()
        tpm, tps = run_inference(model, tokenizer, prompt, max_new_tokens)
        if tpm == 0 and tps == 0:
            logger.error(f"Skipping iteration for {model_name} due to inference failure.")
            continue
        memory_after = get_memory_usage()
        results.append({
            "model": model_name,
            "tpm": tpm,
            "tokens_per_second": tps,
            "peak_memory_mb": memory_after - memory_before,
            "load_time_s": load_time
        })
        logger.info(f"Iteration complete: TPM={tpm:.2f}, TPS={tps:.2f}, Memory={memory_after - memory_before:.2f}MB")
    
    # Clear memory and cache
    del model
    torch.cuda.empty_cache()
    clear_model_cache(model_name)
    return results

def save_results(results, output_file="results/benchmark.csv"):
    """Save benchmark results to CSV."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "tpm", "tokens_per_second", "peak_memory_mb", "load_time_s"])
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    logger.info(f"Results saved to {output_file}")

def main():
    # Log system info
    system_info = get_system_info()
    logger.info("System Info: %s", system_info)
    
    models = [
        "meta-llama/Llama-3.1-8B",
        "Qwen/Qwen2.5-7B",
        "google/gemma-2b"
    ]
    prompt = "This is a sample prompt for benchmarking LLMs. It contains exactly 100 tokens to ensure consistency across models. The prompt is designed to be neutral and representative of typical input, such as a short paragraph describing a common scenario or question. The goal is to evaluate how quickly and efficiently the model can generate coherent text in response to this input, simulating real-world usage."[:100]
    
    all_results = []
    for model_name in models:
        results = benchmark_model(model_name, prompt)
        if results:
            all_results.extend(results)
    
    if all_results:
        save_results(all_results)
        # Compute and log averages
        for model_name in models:
            model_results = [r for r in all_results if r["model"] == model_name]
            if model_results:
                avg_tpm = np.mean([r["tpm"] for r in model_results])
                avg_tps = np.mean([r["tokens_per_second"] for r in model_results])
                avg_memory = np.mean([r["peak_memory_mb"] for r in model_results])
                logger.info(f"Average for {model_name}: TPM={avg_tpm:.2f}, TPS={avg_tps:.2f}, Memory={avg_memory:.2f}MB")
    else:
        logger.error("No results collected due to errors.")

if __name__ == "__main__":
    main()
