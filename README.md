
# Too Consistent to Detect: A Study of Self-Consistent Errors in LLMs

Official code for the **EMNLP 2025 main paper** 📄 ["Too Consistent to Detect: A Study of Self-Consistent Errors in LLMs"](https://arxiv.org/abs/2505.17656). We reveal a critical problem of LLMs, **Self-consistent errors (SCEs)**, where LLMs repeatedly generate the same error.
- Their frequency stays stable or even **increases** as models scale.  
- They are **hard to detect** for current error detection methods.

**This repo provide:**
- 🧩 Curated dataset of **Self-consistent errors**, and matched correct, inconsistent error samples.  
- 🔍 A simple yet effective **cross-model probe** to improve the detection of self-consistent errors.


<p align="center">
  <img src="figure/sce.png" alt="Self-consistent error illustration" width="75%">
  <br>
</p>




---

### 1. Environment, Data, and Models

#### Install
```bash
pip install -r requirements.txt
````

#### Models and Datasets
We use Qwen2.5-14B-Instruct as the verifier in the paper, and you can replace it with any other models.
* **Response model** (produces answers & hidden states):[Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
* **Verifier model** (provides cross signals via hidden states):[Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)

"data" includes SCE splits constructed for `llama3.1-8b-instruct` and `Qwen2.5-8b-instruct`.
* For each subset (CE / IE), the sample count is **matched** for fair comparison.
  * `correct.jsonl` contains the correct responses.
  * `consistent_incorrect.jsonl` contains self-consistent errors
  * `inconsistent_incorrect.jsonl` contains inconsistent errors.

### 2. Quick Start: Cross-Model Probe

```bash
python3 code/cross-model_probe.py \
  --dataset sciq-ce \
  --response_model llama3.1-8b \
  --response_model_path /path/to/the/model \
  --verifier qwen2.5-14b \
  --verifier_path /path/to/the/model \
  --method probe \
  --seed 42 \
  --data_dir /your/data/dir/full-self-consistent-dataset \
  --save_dir /path/you/want/to/save
```

---

### 3. Explanation: What the Cross-Model Probe Does?

1. **Extract hidden states** from for each example from:
   * the **response model** (e.g., Llama-3.1-8B-Instruct), and
   * the **verifier model** (e.g., Qwen2.5-14B-Instruct).
2. **Train a lightweight probe** (including selecting the best layer to use) **separately** on each model’s hidden representation to predict correctness vs. error.
3. **Fuse** the two probe scores with a scalar **λ** chosen on a dev set.

**Implementation note:** The core code lives in `cross-model_probe.py` (around lines **351–379**): it fits two probes (one per model’s hidden states) and **searches λ** to maximize dev performance before reporting test results.


#### Command-Line Arguments

* `--dataset`: Which subset to run on (e.g., `triviaqa-ce`, `sciq-ce`, `triviaqa-ie`, `sciq-ie`).
* `--data_dir`: Root folder of the downloaded SCE dataset.
* `--response_model`: Shorthand name (e.g., `llama3.1-8b`).
* `--response_model_path`: Local path to the response model weights.
* `--verifier`: Shorthand name (e.g., `qwen2.5-14b`).
* `--verifier_path`: Local path to the verifier model weights.
* `--method`: Set to `probe` to enable the cross-model probe pipeline.
* `--seed`: Random seed for reproducibility.
* `--save_dir`: Output directory for logs, metrics, and artifacts.

---

### Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{tan-etal-2025-consistent,
    title = "Too Consistent to Detect: A Study of Self-Consistent Errors in {LLM}s",
    author = "Tan, Hexiang  and
      Sun, Fei  and
      Liu, Sha  and
      Su, Du  and
      Cao, Qi  and
      Chen, Xin  and
      Wang, Jingang  and
      Cai, Xunliang  and
      Wang, Yuanzhuo  and
      Shen, Huawei  and
      Cheng, Xueqi",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.238/",
    doi = "10.18653/v1/2025.emnlp-main.238",
    pages = "4755--4765",
    ISBN = "979-8-89176-332-6",
    abstract = "As large language models (LLMs) often generate plausible but incorrect content, error detection has become increasingly critical to ensure truthfulness.However, existing detection methods often overlook a critical problem we term as **self-consistent error**, where LLMs repeatedly generate the same incorrect response across multiple stochastic samples.This work formally defines self-consistent errors and evaluates mainstream detection methods on them.Our investigation reveals two key findings: (1) Unlike inconsistent errors, whose frequency diminishes significantly as the LLM scale increases, the frequency of self-consistent errors remains stable or even increases.(2) All four types of detection methods significantly struggle to detect self-consistent errors.These findings reveal critical limitations in current detection methods and underscore the need for improvement.Motivated by the observation that self-consistent errors often differ across LLMs, we propose a simple but effective \textit{cross{-}model probe} method that fuses hidden state evidence from an external verifier LLM.Our method significantly enhances performance on self-consistent errors across three LLM families."
}
```
