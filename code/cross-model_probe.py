# coding: utf-8
from __future__ import annotations

import gc
import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
import logging
import json

from joblib import Parallel, delayed

from src.util import compute_auroc, dump_jsonl, init_logger, load_all_jsonl, seed_everything
from src.probe import SupervisedProbe
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

TOTAL_LAYER_DICT: Dict[str, int] = {
    "llama3.1-8b": 32,
    "llama3.1-70b": 80,
    "llama3.1-405b": 126,
    "mistral-12b": 40,
    "mistral-7b": 32,
    "llama3.2-1b": 16,
    "llama3.2-3b": 28,
    "qwen2.5-3b": 36,
    "qwen2.5-7b": 28,
    "qwen2.5-14b": 48,
    "qwen2.5-32b": 64,
    "qwen2.5-72b": 80,
}


def init_llama_from_local(model_name, model_path):
    path = model_path
    tok = AutoTokenizer.from_pretrained(path)
    if any(x in model_name for x in ("70b", "405b")):
        mdl = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype=torch.float16)
    else:
        mdl = AutoModelForCausalLM.from_pretrained(path, device_map="auto")
    mdl.eval()
    return mdl, tok


def shuffle_arrays(hds: np.ndarray, labels: np.ndarray, seed: int | None = None):
    if seed is not None:
        np.random.seed(seed)
    order = np.random.permutation(len(hds))
    return hds[order], labels[order]

def get_generalization_dataset(dataset):
    dataset_mapping = {
        "sciq-ce": "triviaqa-ce",
        "triviaqa-ce": "sciq-ce",
        "sciq-ie": "triviaqa-ie",
        "triviaqa-ie": "sciq-ie",
    }
    return dataset_mapping.get(dataset, dataset)  # 如果映射中没有，返回原始数据集


def _hash_path(p: str) -> str:
    """sha1‑hash a path string and return first 12 chars for brevity."""
    return hashlib.sha1(p.encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------
# Helper for saving a single layer ------------------------------------------------

def _save_single_layer(vecs: List[np.ndarray], file_path: Path) -> None:
    """Save one layer's vectors to ``file_path`` (np.save)."""
    np.save(file_path, np.ascontiguousarray(vecs))

# ----------------------------------------------------------------------------
# Main util -------------------------------------------------------------------

def generate_and_save_hd(
    jsonl_path: str,
    hd_model_name: str,
    total_layer: int,
    target_layer: int,
    seed: int,
    cache_root: Path,
    method: str = "probe",
    *,
    save_parallel: bool = True,
    n_jobs: int = 4,
    model=None,  
    tokenizer=None  
):
    data = load_all_jsonl(jsonl_path)

    # Build unique cache dir ----------------------------------------------
    basename_key = _hash_path(os.path.splitext(jsonl_path)[0])
    hd_dir = Path(cache_root) / f"{basename_key}_{hd_model_name}_{method}"
    hd_dir.mkdir(parents=True, exist_ok=True)
    hd_file = hd_dir / f"{target_layer}.npy"

    # Fast‑path ------------------------------------------------------------
    if hd_file.exists():
        logging.info(f"Load Existing hd: {hd_file}")
        return np.load(hd_file), np.array([1 if d["correctness"] == "A" else 0 for d in data], dtype=int)

    # ---------------------------------------------------------------------
    # Compute HDs ----------------------------------------------------------
    if model is None or tokenizer is None:
        logging.error("Error: Need llm in function generate_and_save_hd")
        exit()
        # model, tokenizer = init_llama_from_local(hd_model_name)
    
    probe = SupervisedProbe()

    all_layers: Dict[int, List[np.ndarray]] = {l: [] for l in range(total_layer)}
    for item in tqdm(data, desc=f"Generating HD"):
        if method == "probe":
            hd_tensor = probe.generate_hd(model, tokenizer, item["query"], item["response"], token=-1)
        for l in range(total_layer):
            all_layers[l].append(hd_tensor[l].squeeze())

    # ---------------------------------------------------------------------
    # Save layer‑wise HDs with progress bar (and optional parallelism) -----
    if save_parallel:
        Parallel(n_jobs=n_jobs)(
            delayed(_save_single_layer)(vecs, hd_dir / f"{l}.npy")
            for l, vecs in tqdm(all_layers.items(), desc="Saving HD layers (parallel)")
        )
    else:
        for l, vecs in tqdm(all_layers.items(), desc="Saving HD layers"):
            _save_single_layer(vecs, hd_dir / f"{l}.npy")

    # ---------------------------------------------------------------------
    # Prepare return values -----------------------------------------------
    labels = np.array([1 if d["correctness"] == "A" else 0 for d in data], dtype=int)
    return np.array(all_layers[target_layer]), labels


# -----------------------------------------------------------------------------
# Data loader ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def get_data(
    response_model_name: str,
    embedding_model_name: str,  
    dataset_name: str,
    split: str,
    subset: str,
    layer: int,
    total_layer: int,
    seed: int,
    data_root: str,
    cache_root: Path,
    method: str,
    model=None,  
    tokenizer=None,  
):
    if subset not in {"error", "correct"}:
        raise ValueError("subset must be error/correct")

    suffix = "consistent_incorrect" if subset == "error" else "correct"
    data_path = f"{data_root}/{response_model_name}_{dataset_name}_{split}_15_0.5/{suffix}.jsonl"
    if subset == "error" and not os.path.exists(data_path):
        data_path = data_path.replace("consistent_incorrect", "inconsistent_incorrect")
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)

    return generate_and_save_hd(
        jsonl_path=data_path,
        hd_model_name=embedding_model_name,
        total_layer=total_layer,
        target_layer=layer,
        seed=seed,
        cache_root=cache_root,
        method=method,
        model=model,  
        tokenizer=tokenizer,  
    )


# -----------------------------------------------------------------------------
# ProbeTrainer -----------------------------------------------------------------
# -----------------------------------------------------------------------------

class ProbeTrainer:
    def __init__(self, response_model_name: str, hd_model_name: str, hd_model_path: str, dataset: str, args: argparse.Namespace, logger):
        self.response_model = response_model_name
        self.hd_model = hd_model_name
        self.hd_model_path = hd_model_path
        self.dataset = dataset
        self.args = args
        self.logger = logger
        self.total_layer = TOTAL_LAYER_DICT[hd_model_name]

        self.save_root = Path(args.save_dir) / args.method / f"{dataset}-{response_model_name}-verifier_{hd_model_name}"
        self.save_root.mkdir(parents=True, exist_ok=True)
        self.cache_root = Path(args.save_dir) / "hd_cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)

        self.probe = None  
        self.best_layer = None
        self.dev_best_auroc = None
        self.dev_scores_best = None
        self.test_scores_best = None

        # init embedding model
        self.embedding_model, self.tokenizer = init_llama_from_local(hd_model_name, hd_model_path)

    # ------------------------------------------------------------------
    def _load(self, split: str, subset: str, layer: int):
        return get_data(
            response_model_name=self.response_model,
            embedding_model_name=self.hd_model,
            dataset_name=self.dataset,
            split=split,
            subset=subset,
            layer=layer,
            total_layer=self.total_layer,
            seed=self.args.seed,
            data_root=self.args.data_dir,
            cache_root=self.cache_root,
            method=self.args.method,
            model=self.embedding_model,  
            tokenizer=self.tokenizer,  
        )

    def train_and_select_layer(self):
        """
        Search the best hidden‑state layer for the current (response_model, hd_model) pair.
        Loads HDs (per layer) to keep the memory footprint small.
        """
        tag = f"{self.response_model}(hd:{self.hd_model})"
        self.logger.info(f"*** Searching best layer for {tag} ***")

        self.best_layer = None
        self.dev_best_auroc = None
        self.dev_scores_best = None
        self.test_scores_best = None
        self.probe = None

        for l in range(self.total_layer):
            # --- Load HDs and labels for *this* layer only -----------------------
            tr_hd, tr_lb = zip(*(self._load("train", s, l) for s in ("correct", "error")))
            dv_hd, dv_lb = zip(*(self._load("dev", s, l)   for s in ("correct", "error")))

            tr_h, tr_l = shuffle_arrays(np.concatenate(tr_hd),
                                        np.concatenate(tr_lb),
                                        seed=self.args.seed)
            dv_h, dv_l = np.concatenate(dv_hd), np.concatenate(dv_lb)

            # --- Train probe on current layer ------------------------------------
            probe = SupervisedProbe()
            probe.train(
                tr_h, tr_l,
                dv_h, dv_l,
                save_dir=str(self.save_root / str(l)),
                epoch=200,
                lr=1e-3,
                batch_size=1024,
                device="cuda",
                weight_decay=0.01,
                writer=None,
            )

            # --- Evaluate on DEV --------------------------------------------------
            dv_scores = probe.estimate(dv_h).cpu().numpy()
            auroc = compute_auroc(y_confs=dv_scores, y_true=dv_l)
            self.logger.info(f"{tag} L{l:02d} DEV‑AUROC = {auroc:.4f}")

            # --- Track best layer -------------------------------------------------
            if self.dev_best_auroc is None or auroc > self.dev_best_auroc:
                self.best_layer = l
                self.dev_best_auroc = auroc
                self.dev_scores_best = dv_scores
                self.probe = probe

                tst_hd, _ = zip(*(self._load("test", s, l) for s in ("correct", "error")))
                self.test_scores_best = self.probe.estimate(np.concatenate(tst_hd)).cpu().numpy()

            # --- Free memory before next iteration -------------------------------
            del tr_hd, tr_lb, dv_hd, dv_lb, tr_h, tr_l, dv_h, dv_l
            gc.collect()
            torch.cuda.empty_cache()

        self.logger.info(f"Best layer for {tag}: L{self.best_layer} (AUROC {self.dev_best_auroc:.4f})")
        dump_jsonl(
            {
                "response_model": self.response_model,
                "hd_model": self.hd_model,
                "best_layer": self.best_layer,
                "dev_auroc": self.dev_best_auroc,
            },
            str(self.save_root / "best_layer.jsonl"),
        )

    def evaluate_generalization(self, gen_dataset: str):
        if self.probe is None:
            raise RuntimeError("No trained probe available. Run train_and_select_layer first")
        
        original_dataset = self.dataset
        self.dataset = gen_dataset

        try:
            tst_h, tst_l = zip(*(self._load("test", s, self.best_layer) for s in ("correct", "error")))
            tst_h_concat = np.concatenate(tst_h)
            tst_l_concat = np.concatenate(tst_l)

            # predict
            gen_scores = self.probe.estimate(tst_h_concat).cpu().numpy()
            gen_auroc = compute_auroc(y_confs=gen_scores, y_true=tst_l_concat)

            self.dataset = original_dataset

            return gen_scores, tst_l_concat, gen_auroc

        except Exception as e:
            self.dataset = original_dataset
            raise e
        
    def get_scores(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.dev_scores_best is None or self.test_scores_best is None:
            raise RuntimeError("train_and_select_layer error")
        return self.dev_scores_best, self.test_scores_best



# Main -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sciq-ce")
    p.add_argument("--response_model", required=True) # only used to get the path of the response data
    p.add_argument("--response_model_path", required=True)
    p.add_argument("--verifier", required=True)
    p.add_argument("--verifier_path", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--method", default="probe")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--save_dir", required=True)
    args = p.parse_args()

    seed_everything(args.seed)
    log_dir = f"{args.save_dir}/{args.method}/{args.dataset}-{args.response_model}-verifier_{args.verifier}"
    os.makedirs(log_dir, exist_ok=True)
    logger = init_logger(log_file=log_dir + "/dual_probe.log")

    # -------------------------------Main Code of Cross-Model Probe----------------------------
    # train probe using hd of verifier model
    trainer_ver = ProbeTrainer(response_model_name=args.response_model, hd_model_name=args.verifier, dataset=args.dataset, args=args, logger=logger, hd_model_path=args.verifier_path)
    trainer_ver.train_and_select_layer()

    # train probe using hd of response model
    trainer_llm = ProbeTrainer(response_model_name=args.response_model, hd_model_name=args.response_model, dataset=args.dataset, args=args, logger=logger, hd_model_path=args.response_model_path)
    trainer_llm.train_and_select_layer()

    # get scores from two probe
    dev_llm, test_llm = trainer_llm.get_scores()
    dev_ver, test_ver = trainer_ver.get_scores()
    dev_labels = np.concatenate([trainer_llm._load("dev", s, trainer_llm.best_layer)[1] for s in  ("correct", "error")])
    test_labels = np.concatenate([trainer_llm._load("test", s, trainer_llm.best_layer)[1] for s in  ("correct", "error")])

    # AUROC using only response model or verifier
    test_llm_auroc = compute_auroc(y_confs=test_llm, y_true=test_labels)
    test_ver_auroc = compute_auroc(y_confs=test_ver, y_true=test_labels)

    # search lambda
    best_lambda, best_dev = 0.0, -1.0
    lambda_auroc_results = []  
    for lam in np.linspace(0, 1, 21):
        auroc = compute_auroc(y_confs=(1 - lam) * dev_llm + lam * dev_ver, y_true=dev_labels)
        logger.info(f"  λ   : {lam:.2f}  auroc: {auroc:.4f}")
        lambda_auroc_results.append({"lambda": float(lam), "auroc": float(auroc)})
        if auroc > best_dev:
            best_lambda, best_dev = lam, auroc
    best_test = compute_auroc(y_confs=(1 - best_lambda) * test_llm + best_lambda * test_ver, y_true=test_labels)
    
    #------------------------------------------------------------------------------------------
    
    # save lambda-AUROCresult
    lambda_auroc_path = str(Path(args.save_dir) / f"{args.dataset}_{args.response_model}_{args.verifier}_lambda_auroc.json")
    with open(lambda_auroc_path, 'w') as f:
        json.dump(lambda_auroc_results, f, indent=2)
    logger.info(f"Lambda-AUROC results saved to {lambda_auroc_path}")

    # test generalization performance
    gen_dataset = get_generalization_dataset(args.dataset)
    logger.info(f"\n===== GENERALIZATION TEST ({gen_dataset}) =====")
    try:
        logger.info(f"Evaluating {args.response_model} on {gen_dataset}...")
        gen_llm_scores, gen_labels, gen_llm_auroc = trainer_llm.evaluate_generalization(gen_dataset)
        logger.info(f"  LLM ({args.response_model}) on {gen_dataset}: {gen_llm_auroc:.4f}")

        # verifier only
        logger.info(f"Evaluating {args.verifier} on {gen_dataset}...")
        gen_ver_scores, _, gen_ver_auroc = trainer_ver.evaluate_generalization(gen_dataset)
        logger.info(f"  Verifier ({args.verifier}) on {gen_dataset}: {gen_ver_auroc:.4f}")

        # cross-model probe
        gen_fused_scores = (1 - best_lambda) * gen_llm_scores + best_lambda * gen_ver_scores
        gen_fused_auroc = compute_auroc(y_confs=gen_fused_scores, y_true=gen_labels)
        logger.info(f"  Fused model on {gen_dataset} (λ={best_lambda:.2f}): {gen_fused_auroc:.4f}")

        # save
        gen_sample_results = []
        for idx in range(len(gen_labels)):
            gen_sample_results.append({
                "index": idx,
                "llm_score": float(gen_llm_scores[idx]),
                "verifier_score": float(gen_ver_scores[idx]),
                "fused_score": float(gen_fused_scores[idx]),
                "label": int(gen_labels[idx])
            })

        gen_output_path = str(Path(args.save_dir) / f"{gen_dataset}_{args.response_model}_{args.verifier}_sample_scores.jsonl")
        with open(gen_output_path, 'w') as f:
            for item in gen_sample_results:
                f.write(json.dumps(item) + '\n')

        logger.info(f"Generalization sample-level scores saved to {gen_output_path}")

        has_gen_results = True
    except Exception as e:
        logger.error(f"Error during generalization testing: {str(e)}")
        has_gen_results = False

    result_str = (
        f"\n===== FINAL RESULTS =====\n"
        f"  Response: {args.response_model} (L{trainer_llm.best_layer})\n"
        f"  Verifier: {args.verifier}, L{trainer_ver.best_layer})\n"
        f"  Best λ   : {best_lambda:.2f}\n"
        f"  Original Probe DEV  : {trainer_llm.dev_best_auroc:.4f}\n"
        f"  Verifier-only Probe DEV  : {trainer_ver.dev_best_auroc:.4f}\n"
        f"  Cross-Model Probe DEV: {best_dev:.4f}\n"
        f"  Original Probe  TEST  : {test_llm_auroc:.4f}\n"
        f"  Verifier-only Probe TEST  : {test_ver_auroc:.4f}\n"
        f"  Cross-Model Probe TEST: {best_test:.4f}\n"
    )

    if has_gen_results:
        result_str += (
            f"  --- Generalization on {gen_dataset} ---\n"
            f"  Original Probe Generalization: {gen_llm_auroc:.4f}\n"
            f"  Verifier-only Probe Generalization: {gen_ver_auroc:.4f}\n"
            f"  Cross-Model Probe Generalization: {gen_fused_auroc:.4f}\n"
        )

    result_str += f"========================"
    logger.info(result_str)

    fused_scores = (1 - best_lambda) * test_llm + best_lambda * test_ver

    sample_results = []
    for idx in range(len(test_labels)):
        sample_results.append({
            "index": idx,
            "llm_score": float(test_llm[idx]),
            "verifier_score": float(test_ver[idx]),
            "fused_score": float(fused_scores[idx]),
            "label": int(test_labels[idx])
        })

    sample_output_path = str(Path(args.save_dir) / f"{args.dataset}_{args.response_model}_{args.verifier}_sample_scores.jsonl")
    with open(sample_output_path, 'w') as f:
        for item in sample_results:
            f.write(json.dumps(item) + '\n')

    logger.info(f"Sample-level scores saved to {sample_output_path}")

    summary_data = {
        "response_model": args.response_model,
        "verifier": args.verifier,
        "dataset": args.dataset,
        "best_lambda": float(best_lambda),
        "llm_test_auroc": float(test_llm_auroc),
        "ver_test_auroc": float(test_ver_auroc),
        "fused_test_auroc": float(best_test),
    }

    if has_gen_results:
        summary_data.update({
            "gen_dataset": gen_dataset,
            "llm_gen_auroc": float(gen_llm_auroc),
            "ver_gen_auroc": float(gen_ver_auroc),
            "fused_gen_auroc": float(gen_fused_auroc)
        })

    dump_jsonl(summary_data, str(Path(args.save_dir) / f"{args.dataset}_{args.response_model}_{args.verifier}_fused_summary.jsonl"))


if __name__ == "__main__":
    main()

# 
# CUDA_VISIBLE_DEVICES=2,3,6,7 python3 code/cross-model_probe.py     --dataset triviaqa-ie     --response_model llama3.1-8b --response_model_path /data/tanhexiang/too_consistent_to_detect/models/llama3.1-8b-instruct  --verifier qwen2.5-14b --verifier_path  /data/tanhexiang/too_consistent_to_detect/models/qwen2.5-14b-instruct  --method probe     --seed 42     --data_dir "/data/tanhexiang/too_consistent_to_detect/self-consistent-dataset"     --save_dir "/data/tanhexiang/too_consistent_to_detect/result"