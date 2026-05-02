#!/usr/bin/env python3
import argparse
import json
import math
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class EvalRecord:
    prompt: str
    response: str


def load_records(path: str, max_samples: int | None) -> list[EvalRecord]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records.append(EvalRecord(prompt=str(row["prompt"]), response=str(row["response"])))
            if max_samples is not None and len(records) >= max_samples:
                break
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def get_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_model_and_tokenizer(model_dir: str, dtype_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if not tokenizer.is_fast:
        raise ValueError("This script requires a fast tokenizer.")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(f"Tokenizer at {model_dir} has no pad_token or eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=get_dtype(dtype_name),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, tokenizer


def chunked(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def collect_rank_probs(
    records: list[EvalRecord],
    model,
    tokenizer,
    top_k: int,
    batch_size: int,
) -> list[list[float]]:
    device = next(model.parameters()).device
    rank_probs = [[] for _ in range(top_k)]
    iterator = tqdm(chunked(records, batch_size), total=math.ceil(len(records) / batch_size), desc="Scoring", dynamic_ncols=True)

    for batch_records in iterator:
        full_texts = [record.prompt + record.response for record in batch_records]
        prompt_char_lens = [len(record.prompt) for record in batch_records]
        tokenized = tokenizer(
            full_texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offset_mapping = tokenized.pop("offset_mapping")
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[:, :-1, :]
            probs = torch.softmax(logits.float(), dim=-1)
            topk_probs = torch.topk(probs, k=top_k, dim=-1).values.cpu()

        attention_mask_cpu = attention_mask.cpu()
        for row_idx, record in enumerate(batch_records):
            valid_len = int(attention_mask_cpu[row_idx].sum().item())
            offsets = offset_mapping[row_idx][:valid_len].tolist()
            response_positions = [
                pos for pos, (start, end) in enumerate(offsets)
                if end > start and start >= prompt_char_lens[row_idx]
            ]
            response_positions = [pos for pos in response_positions if pos > 0]
            for pos in response_positions:
                prob_pos = pos - 1
                for rank_idx in range(top_k):
                    rank_probs[rank_idx].append(float(topk_probs[row_idx, prob_pos, rank_idx].item()))

        del outputs, logits, probs, topk_probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rank_probs


def compute_histograms(rank_probs: list[list[float]], num_bins: int) -> tuple[list[float], list[dict]]:
    bin_edges = torch.linspace(0.0, 1.0, steps=num_bins + 1).tolist()
    results = []
    for rank_idx, values in enumerate(rank_probs, start=1):
        hist = torch.histc(torch.tensor(values, dtype=torch.float32), bins=num_bins, min=0.0, max=1.0)
        proportions = (hist / hist.sum()).tolist()
        results.append(
            {
                "rank": rank_idx,
                "count": len(values),
                "mean": float(sum(values) / len(values)) if values else float("nan"),
                "variance": float(torch.tensor(values, dtype=torch.float32).var(unbiased=False).item()) if values else float("nan"),
                "bin_proportions": proportions,
            }
        )
    return bin_edges, results


def plot_histograms(label: str, bin_edges: list[float], histograms: list[dict], output_png: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), squeeze=False)
    flat_axes = axes.flatten()
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2.0 for i in range(len(bin_edges) - 1)]
    bar_width = bin_edges[1] - bin_edges[0]

    for ax, hist in zip(flat_axes, histograms):
        ax.bar(bin_centers, hist["bin_proportions"], width=bar_width * 0.95, align="center", alpha=0.85)
        ax.set_title(f"{label} rank-{hist['rank']}\nmean={hist['mean']:.4f} var={hist['variance']:.4f}")
        ax.set_xlabel("probability")
        ax.set_ylabel("proportion")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot top-K candidate probability distributions by rank.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_png", required=True)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--num_bins", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--max_samples", type=int)
    args = parser.parse_args()

    records = load_records(args.eval_jsonl, max_samples=args.max_samples)
    model, tokenizer = load_model_and_tokenizer(args.model_dir, args.dtype)
    rank_probs = collect_rank_probs(records, model, tokenizer, args.top_k, args.batch_size)
    bin_edges, histograms = compute_histograms(rank_probs, args.num_bins)

    payload = {
        "label": args.label,
        "model_dir": args.model_dir,
        "eval_jsonl": args.eval_jsonl,
        "top_k": args.top_k,
        "num_bins": args.num_bins,
        "num_records": len(records),
        "bin_edges": bin_edges,
        "histograms": histograms,
    }

    for path in [args.output_json, args.output_png]:
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    with open(args.output_json, "w") as f:
        json.dump(payload, f)
    plot_histograms(args.label, bin_edges, histograms, args.output_png)

    print(f"Wrote json to: {args.output_json}")
    print(f"Wrote plot to: {args.output_png}")


if __name__ == "__main__":
    main()
