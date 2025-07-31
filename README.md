LLM Local Benchmark
This project benchmarks three large language models—Llama 3.1 8B, Qwen 2.5 7B, and Gemma 2B—for local inference on a CPU-only system. The goal is to evaluate their viability, latency, text quality, and efficiency for generating creative stories, without relying on GPU acceleration.
Purpose
The benchmark assesses:

Viability: Can models run locally on a CPU with 31.7GB RAM without crashing?
Latency: Tokens per minute (TPM) targets: Llama/Qwen 70–150, Gemma 190–350, stddev <5.
Text Quality: Generate coherent, diverse 50-word stories (Llama: 2/3+, Qwen: 3/3, Gemma: 2/3+).
Efficiency: Complete run in ~30 minutes, model load times <10s.

Prompt: "Write a 50-word creative story about a scientist discovering a new energy source to save the planet."
System Setup

OS: Windows 10 (version 10.0.26100)
CPU: Intel64 Family 6 Model 183 Stepping 1
RAM: 31.7GB
Disk: ~697GB free
GPU: None
Python: 3.x (compatible with dependencies)
Run Date: July 31, 2025, 16:47–17:02 IST
Run Time: ~15 minutes

Results
The benchmark ran 3 iterations per model, generating 50 tokens per iteration. Key findings:

Qwen 2.5 7B:
TPM: 66.73 (range: 62.12–69.35, stddev 3.95), slightly below 70–150, stable.
Text: 3/3 coherent, diverse stories (e.g., Dr. Sarah’s orb, Dr. Ethan’s Quantum Flux, Dr. Cynthia’s element).
Load Time: 14.89s (misses <10s due to cache issues).
Verdict: Best for creative tasks on CPU, near-perfect performance.


Llama 3.1 8B:
TPM: 70.47 (range: 61.79–84.78, stddev 11.86), meets 70–150, unstable (stddev >5).
Text: 0/3 coherent stories (irrelevant outputs: contest instructions, penguin example).
Load Time: 16.61s (misses <10s).
Verdict: Viable for TPM but fails text quality; needs GPU or better prompting.


Gemma 2B:
TPM: 103.63 (range: 95.16–109.04, stddev 7.37), below 190–350, unstable.
Text: 2/3 partial stories (incomplete, some prompt echoing), misses 2/3+ coherent target.
Load Time: 9.37s (meets <10s).
Verdict: Fastest but inconsistent text and low TPM.



Files:

results/benchmark.csv: Aggregated metrics (TPM, TPS, load times, etc.).
results/benchmark_partial_<model>.csv: Model-specific metrics.
results/generated_text_<model>.txt: Full text outputs.

Project Structure
llm-local-benchmark/
├── src/
│   └── benchmark.py
├── results/
│   ├── benchmark.csv
│   ├── benchmark_partial_meta-llama_Llama-3.1-8B.csv
│   ├── benchmark_partial_Qwen_Qwen2.5-7B.csv
│   ├── benchmark_partial_google_gemma-2b.csv
│   ├── generated_text_meta-llama_Llama-3.1-8B.txt
│   ├── generated_text_Qwen_Qwen2.5-7B.txt
│   ├── generated_text_google_gemma-2b.txt
├── README.md
├── requirements.txt
├── .gitignore

Setup Instructions

Clone the Repository:git clone https://github.com/<your-username>/llm-local-benchmark.git
cd llm-local-benchmark


Install Dependencies:pip install -r requirements.txt


Set Hugging Face Token:set HF_TOKEN=your_huggingface_token


Run the Benchmark:python src/benchmark.py --keep-cache


Outputs saved to results/.
Use --keep-cache if >100GB disk space available; otherwise, clear cache:rmdir /s /q %USERPROFILE%\.cache\huggingface\hub





Notes

Memory Tracking: Ignored due to unreliable measurements (0.00–2.00MB vs. expected ~100–500MB).
Cache Issues: Cache misses increased load times for Llama/Qwen. Clear cache before runs for consistency.
Improvements:
Use GPU for better Llama/Gemma text quality and TPM.
Add negative prompting (e.g., “Avoid instructions, academic content”) to fix Llama/Gemma outputs.
Install huggingface_hub[hf_xet] for faster model downloads:pip install huggingface_hub[hf_xet]




Limitations: Llama’s 0/3 stories and Gemma’s partial outputs indicate CPU limitations. Qwen is the most reliable for creative tasks.

License
MIT License
Author
[Your Name] - Machine Learning Engineer
For feedback or collaboration, open an issue or contact [your-email@example.com].
