LLM Local Benchmark
This repository benchmarks open-source LLMs (Llama 3.1 8B, Qwen 2.5 7B, Gemma 2B) for latency (TPM), throughput (TPS), memory usage, load time, and generated text on a local machine.
Setup

Install Dependencies:
pip install -r requirements.txt


Set Hugging Face Token (required for Llama 3.1 8B, Gemma 2B):
set HF_TOKEN=  YOUR TOKEN# Windows
export HF_TOKEN= YOUR TOKEN # Linux/Mac


Accept terms: Llama 3.1 8B, Gemma 2B.


Pre-download Models:
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B', cache_dir='C:/Users/Sasi Kanth/.cache/huggingface/hub')"
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B', cache_dir='C:/Users/Sasi Kanth/.cache/huggingface/hub')"
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('google/gemma-2b', cache_dir='C:/Users/Sasi Kanth/.cache/huggingface/hub')"


Run the Script:
python src/benchmark.py --keep-cache


Outputs: results/benchmark.csv, results/benchmark_partial_<model>.csv, results/generated_text_<model>.txt.



Disk Space Notes

Models require ~10–20GB each.
Use --keep-cache if >100GB free.
Check disk space:python -c "import psutil; print(psutil.disk_usage('.').free / 1024**3)"



CPU-Specific Notes

No GPU: Quantization disabled, max_new_tokens=100.
Ensure 32GB RAM (Llama: ~20GB, Qwen: ~14GB, Gemma: ~5GB).
Install accelerate for Qwen:pip install accelerate


Iteration times: ~40–60s (Llama/Qwen), ~15–25s (Gemma).

Output

CSV (results/benchmark.csv): Columns: model, tpm, tokens_per_second, peak_memory_mb, load_time_s, generated_tokens, generated_text (first 200 chars).
Partial CSVs: Per-model results.
Text Logs: Full generated text per iteration.
Logs include system info, cache hits, input/output tokens, and metrics.

Troubleshooting

HF_TOKEN: Ensure token set and terms accepted.
Low Disk: Free ~20GB (delete C:\Users\Sasi Kanth\.cache\huggingface\hub).
OOM: Set max_new_tokens=50.
Cache: Verify *.safetensors or *.bin in cache, pre-download, use --keep-cache.
Memory: psutil for loading (20GB Llama, ~14GB Qwen, ~5GB Gemma), tracemalloc for inference (100–500MB).
TPM Variability: Check logs for outliers.
Gemma Text: Uses temperature=0.9, top_k=50 for diversity.
