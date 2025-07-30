import time
import tracemalloc
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
import gc
import argparse

# Suppress symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_SYMLINKS_WARNING"] = "1"

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
        logger.warning("No GPU detected. Using max_new_tokens=100. Expect slower performance.")
    if info["disk_free_gb"] < 20:
        logger.warning(f"Low disk space ({info['disk_free_gb']:.2f} GB). Recommend keeping cache off.")
    return info

def get_memory_usage(use_tracemalloc=True):
    """Return current memory usage in MB using tracemalloc or psutil."""
    if use_tracemalloc:
        tracemalloc.start()
        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics('lineno')
        memory_mb = sum(stat.size for stat in stats) / 1024 / 1024
        tracemalloc.stop()
    else:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        # Estimate GPU memory if applicable
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            memory_mb += gpu_memory
    logger.debug(f"Memory usage ({'tracemalloc' if use_tracemalloc else 'psutil'}): {memory_mb:.2f}MB")
    return memory_mb

def clear_model_cache(model_name, keep_cache=False):
    """Delete model cache to free disk space if not keeping cache."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    model_cache = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
    if keep_cache:
        if os.path.exists(model_cache):
            logger.info(f"Keeping cache for {model_name} at {model_cache} (disk_free={psutil.disk_usage('.').free / 1024**3:.2f}GB)")
        else:
            logger.debug(f"No cache found for {model_name}")
        return
    if os.path.exists(model_cache):
        shutil.rmtree(model_cache)
        logger.info(f"Deleted cache for {model_name} at {model_cache}")
    else:
        logger.debug(f"No cache found for {model_name}")

def verify_cache(model_name):
    """Verify if model cache exists and is intact."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    model_cache = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
    if os.path.exists(model_cache):
        model_files = [f for f in os.listdir(model_cache) if f.endswith((".safetensors", ".bin", ".pt"))]
        if not model_files:
            logger.warning(f"Cache for {model_name} missing model files. Forcing re-download.")
            return False
        total_size = sum(os.path.getsize(os.path.join(model_cache, f)) / 1024**3 for f in os.listdir(model_cache) if os.path.isfile(os.path.join(model_cache, f)))
        logger.info(f"Cache hit for {model_name} at {model_cache}, size={total_size:.2f}GB, files={model_files}")
        return True
    logger.info(f"Cache miss for {model_name}. Downloading model.")
    return False

def load_model(model_name, use_quantization=True, keep_cache=False):
    """Load model and tokenizer with optional 4-bit quantization."""
    logger.info(f"Loading model: {model_name}")
    try:
        disk_free = psutil.disk_usage('.').free / 1024**3
        if disk_free < 10:
            logger.error(f"Insufficient disk space ({disk_free:.2f} GB). Need ~20GB for {model_name}. Skipping.")
            return None, None, 0
        
        if not hasattr(load_model, "logged_in"):
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                logger.error(f"HF_TOKEN not set. Required for {model_name}. Set it via `set HF_TOKEN=your_token`.")
                return None, None, 0
            login(hf_token)
            load_model.logged_in = True
            logger.info("Authenticated with Hugging Face token.")
        
        start_time = time.time()
        memory_baseline = get_memory_usage(use_tracemalloc=False)
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", f"models--{model_name.replace('/', '--')}")
        cache_hit = verify_cache(model_name)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir if cache_hit else None)
        
        use_quantization = use_quantization and torch.cuda.is_available()
        if not use_quantization:
            logger.info(f"Quantization disabled for {model_name} (CPU).")
        
        available_mem = psutil.virtual_memory().available / 1024**3
        if available_mem < 8:
            logger.warning(f"Low memory ({available_mem:.2f} GB). May cause OOM for {model_name}.")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_4bit=use_quantization,
            torch_dtype=torch.float16,
            offload_folder="offload" if not use_quantization else None,
            cache_dir=cache_dir if cache_hit else None
        )
        
        load_time = time.time() - start_time
        memory_after = get_memory_usage(use_tracemalloc=False)
        logger.info(f"Model loaded in {load_time:.2f}s, Memory used: {max(memory_after - memory_baseline, 0):.2f}MB")
        return model, tokenizer, load_time
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        return None, None, 0

def run_inference(model, tokenizer, prompt, max_new_tokens=100, model_name=""):
    """Run inference and measure latency and throughput."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        input_tokens = len(inputs["input_ids"][0])
        
        temperature = 1.0 if "gemma" in model_name.lower() else 0.7
        top_k = 100 if "gemma" in model_name.lower() else None
        
        tracemalloc.start()
        start_time = time.time()
        memory_before = get_memory_usage(use_tracemalloc=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=2 if "gemma" in model_name.lower() else 1,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            top_k=top_k
        )
        end_time = time.time()
        memory_after = get_memory_usage(use_tracemalloc=True)
        tracemalloc.stop()
        
        generated_tokens = len(outputs[0]) - input_tokens
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        time_taken = end_time - start_time
        tpm = (generated_tokens / time_taken) * 60 if time_taken > 0 else 0
        tps = generated_tokens / time_taken if time_taken > 0 else 0
        peak_memory = max(memory_after - memory_before, 0)
        logger.info(f"Input tokens: {input_tokens}, Generated {generated_tokens} tokens in {time_taken:.2f}s: {generated_text[:100]}...")
        return tpm, tps, peak_memory, generated_tokens, generated_text
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        tracemalloc.stop()
        return 0, 0, 0, 0, ""

def save_text_output(model_name, texts):
    """Save full generated texts to a file."""
    os.makedirs("results", exist_ok=True)
    output_file = f"results/generated_text_{model_name.replace('/', '_')}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for i, text in enumerate(texts, 1):
            f.write(f"Iteration {i}:\n{text}\n{'-'*50}\n")
    logger.info(f"Generated texts saved to {output_file}")

def benchmark_model(model_name, prompt, iterations=5, keep_cache=False):
    """Benchmark a single model and return metrics."""
    results = []
    full_texts = []
    model, tokenizer, load_time = load_model(model_name, keep_cache=keep_cache)
    if model is None or tokenizer is None:
        logger.error(f"Skipping {model_name} due to load failure.")
        return results
    
    max_new_tokens = 100
    logger.info(f"Using max_new_tokens={max_new_tokens} for {model_name}")
    
    logger.info(f"Running 4 warmup iterations for {model_name}")
    for _ in range(4):
        run_inference(model, tokenizer, prompt, max_new_tokens, model_name)
        gc.collect()
        torch.cuda.empty_cache()
    
    try:
        memory_baseline = get_memory_usage(use_tracemalloc=True)
        for i in tqdm(range(iterations), desc=f"Benchmarking {model_name}"):
            tpm, tps, peak_memory, gen_tokens, gen_text = run_inference(model, tokenizer, prompt, max_new_tokens, model_name)
            if tpm == 0 and tps == 0:
                logger.error(f"Skipping iteration {i+1} for {model_name} due to inference failure.")
                continue
            results.append({
                "model": model_name,
                "tpm": tpm,
                "tokens_per_second": tps,
                "peak_memory_mb": peak_memory,
                "load_time_s": load_time,
                "generated_tokens": gen_tokens,
                "generated_text": gen_text[:200] + "..." if len(gen_text) > 200 else gen_text
            })
            full_texts.append(gen_text)
            logger.info(f"Iteration {i+1} complete: TPM={tpm:.2f}, TPS={tps:.2f}, Memory={peak_memory:.2f}MB, Tokens={gen_tokens}")
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        clear_model_cache(model_name, keep_cache)
        if results:
            save_results(results, f"results/benchmark_partial_{model_name.replace('/', '_')}.csv")
            save_text_output(model_name, full_texts)
    
    return results

def save_results(results, output_file="results/benchmark.csv"):
    """Save benchmark results to CSV."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "tpm", "tokens_per_second", "peak_memory_mb", "load_time_s", "generated_tokens", "generated_text"])
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    logger.info(f"Results saved to {output_file}")

def main():
    """Run benchmark for all models."""
    parser = argparse.ArgumentParser(description="Benchmark LLMs")
    parser.add_argument("--keep-cache", action="store_true", help="Keep model cache (faster if >100GB free)")
    args = parser.parse_args()
    
    system_info = get_system_info()
    logger.info("System Info: %s", system_info)
    
    models = [
        "meta-llama/Llama-3.1-8B",
        "Qwen/Qwen2.5-7B",
        "google/gemma-2b"
    ]
    prompt = "Write a short story about a scientist discovering a new energy source to save the planet."
    
    all_results = []
    for model_name in models:
        results = benchmark_model(model_name, prompt, keep_cache=args.keep_cache)
        if results:
            all_results.extend(results)
    
    if all_results:
        save_results(all_results)
        for model_name in models:
            model_results = [r for r in all_results if r["model"] == model_name]
            if model_results:
                avg_tpm = np.mean([r["tpm"] for r in model_results])
                avg_tps = np.mean([r["tokens_per_second"] for r in model_results])
                avg_memory = np.mean([r["peak_memory_mb"] for r in model_results])
                avg_tokens = np.mean([r["generated_tokens"] for r in model_results])
                logger.info(f"Average for {model_name}: TPM={avg_tpm:.2f}, TPS={avg_tps:.2f}, Memory={avg_memory:.2f}MB, Tokens={avg_tokens:.2f}")
    else:
        logger.error("No results collected due to errors.")

if __name__ == "__main__":
    main()
