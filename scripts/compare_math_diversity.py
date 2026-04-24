#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_SPECS = {
    "llama3b_grpo": "/scratch/prj0000000224/eval_outputs_full/eval-llama3b-grpo-step160-55495/MATH-Llama-3.2-3B-GRPO-step160/amc23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl",
    "simko_rerun": "/scratch/prj0000000224/eval_outputs_full/eval-llama3b_simko_rerun-2g-testqueue-60202/MATH-Llama-3.2-3B-SimKO-Rerun-step160/amc23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl",
    "simko_ts2_rerun": "/scratch/prj0000000224/eval_outputs_full/eval-llama3b_ts2_rerun-2g-testqueue-60203/MATH-Llama-3.2-3B-SimKO-TS2-Rerun-step160/amc23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl",
    "mix001": "/scratch/prj0000000224/eval_outputs_full/eval-llama3b-ts2-nosquare-mix001-step160/MATH-Llama-3.2-3B-SimKO-TS2-nosquare-mix001-step160_rerun/amc23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl",
    "mix005": "/scratch/prj0000000224/eval_outputs_full/eval-llama3b-ts2-nosquare-mix005-step160/MATH-Llama-3.2-3B-SimKO-TS2-nosquare-mix005-step160_rerun/amc23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl",
    "mix01": "/scratch/prj0000000224/eval_outputs_full/eval-llama3b-ts2-nosquare-mix01-step160/MATH-Llama-3.2-3B-SimKO-TS2-nosquare-mix01-step160_rerun/amc23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl",
}

DEFAULT_SBERT_LOCAL_PATH = "/scratch/prj0000000224/hf_cache/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130"

ANSWER_CANDIDATE_FIELDS = ("pred_answer", "answer", "final_answer")
RESPONSE_CANDIDATE_FIELDS = ("response", "vanilla_response", "completion", "output")
CORRECT_LABEL_FIELDS = ("label", "correct", "is_correct")


@dataclass
class Sample:
    question_id: int
    response: str
    correct: bool
    answer: str


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    return re.sub(r"\s+", " ", text)


def normalize_answer(text: str) -> str:
    text = normalize_text(text)
    text = text.strip("$")
    text = re.sub(r"^\\boxed\{(.*)\}$", r"\1", text)
    text = re.sub(r"\s+", "", text)
    return text.lower()


def pick_first(record: dict, candidates: Iterable[str], default: str = "") -> str:
    for key in candidates:
        value = record.get(key)
        if value is not None:
            return str(value)
    return default


def is_correct(record: dict) -> bool:
    for key in CORRECT_LABEL_FIELDS:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        return str(value).strip().lower() in {"1", "true", "yes", "correct"}
    return False


def load_samples(path: str, group_key: str) -> dict[int, list[Sample]]:
    grouped = defaultdict(list)
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if group_key not in record:
                raise KeyError(f"Missing group key '{group_key}' in {path} line {line_no}")
            grouped[int(record[group_key])].append(
                Sample(
                    question_id=int(record[group_key]),
                    response=normalize_text(pick_first(record, RESPONSE_CANDIDATE_FIELDS)),
                    correct=is_correct(record),
                    answer=normalize_answer(pick_first(record, ANSWER_CANDIDATE_FIELDS)),
                )
            )
    return grouped


def tokenize(text: str) -> list[str]:
    return text.split()


def count_ngrams(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    counts = defaultdict(int)
    if len(tokens) < n:
        return counts
    for i in range(len(tokens) - n + 1):
        counts[tuple(tokens[i : i + n])] += 1
    return counts


def sentence_bleu(candidate: str, reference: str, max_order: int = 4) -> float:
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_order + 1):
        cand_ngrams = count_ngrams(cand_tokens, n)
        ref_ngrams = count_ngrams(ref_tokens, n)
        total = max(sum(cand_ngrams.values()), 1)
        match = 0
        for ngram, count in cand_ngrams.items():
            match += min(count, ref_ngrams.get(ngram, 0))
        precisions.append((match + 1.0) / (total + 1.0))

    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)
    if cand_len == 0:
        return 0.0
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - ref_len / cand_len)
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_order)


def sample_pairs(n_items: int, max_pairs: int, seed: int) -> list[tuple[int, int]]:
    all_pairs = list(itertools.combinations(range(n_items), 2))
    if len(all_pairs) <= max_pairs:
        return all_pairs
    rng = random.Random(seed)
    return rng.sample(all_pairs, max_pairs)


def mean_symmetric_pairwise_bleu(texts: list[str], max_pairs: int, seed: int) -> float:
    if len(texts) < 2:
        return float("nan")
    pairs = sample_pairs(len(texts), max_pairs=max_pairs, seed=seed)
    scores = []
    for i, j in pairs:
        score = 0.5 * (sentence_bleu(texts[i], texts[j]) + sentence_bleu(texts[j], texts[i]))
        scores.append(score)
    return float(sum(scores) / len(scores))


def distinct_ngram_ratio(texts: list[str], n: int) -> float:
    total = 0
    unique = set()
    for text in texts:
        tokens = tokenize(text)
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            unique.add(ngram)
            total += 1
    if total == 0:
        return float("nan")
    return len(unique) / total


def upper_triangle_mean(matrix: np.ndarray) -> float:
    n = matrix.shape[0]
    if n < 2:
        return float("nan")
    tri = matrix[np.triu_indices(n, k=1)]
    if tri.size == 0:
        return float("nan")
    return float(tri.mean())


def cosine_distance_mean(embeddings: np.ndarray) -> float:
    if embeddings.shape[0] < 2:
        return float("nan")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normalized = embeddings / norms
    sim = normalized @ normalized.T
    dist = 1.0 - sim
    return upper_triangle_mean(dist)


def embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )


def nanmean(values: list[float]) -> float:
    arr = np.array(values, dtype=float)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def analyze_one(
    label: str,
    path: str,
    model: SentenceTransformer,
    batch_size: int,
    max_bleu_pairs: int,
    bleu_seed: int,
    group_key: str,
) -> dict[str, float | int | str]:
    grouped = load_samples(path, group_key=group_key)
    if not grouped:
        raise ValueError(f"No samples found in {path}")

    all_samples = [sample for rows in grouped.values() for sample in rows]
    all_responses = [sample.response for sample in all_samples]
    response_embeddings = embed_texts(model, all_responses, batch_size=batch_size)

    offset = 0
    group_sbert = []
    group_sbert_correct = []
    group_bleu = []
    group_bleu_correct = []
    correct_fraction_per_group = []

    for group_id in sorted(grouped):
        rows = grouped[group_id]
        n = len(rows)
        group_embeddings = response_embeddings[offset : offset + n]
        offset += n

        responses = [row.response for row in rows]
        correct_rows = [row for row in rows if row.correct]
        correct_responses = [row.response for row in correct_rows]

        correct_fraction_per_group.append(sum(row.correct for row in rows) / n)
        group_sbert.append(cosine_distance_mean(group_embeddings))
        group_bleu.append(mean_symmetric_pairwise_bleu(responses, max_pairs=max_bleu_pairs, seed=bleu_seed + group_id))

        if correct_rows:
            correct_indices = [idx for idx, row in enumerate(rows) if row.correct]
            correct_embeddings = group_embeddings[correct_indices]
            group_sbert_correct.append(cosine_distance_mean(correct_embeddings))
            group_bleu_correct.append(
                mean_symmetric_pairwise_bleu(
                    correct_responses,
                    max_pairs=max_bleu_pairs,
                    seed=bleu_seed + 100000 + group_id,
                )
            )
        else:
            group_sbert_correct.append(float("nan"))
            group_bleu_correct.append(float("nan"))

    result = {
        "label": label,
        "path": path,
        "num_questions": len(grouped),
        "num_samples": len(all_samples),
        "samples_per_question": len(next(iter(grouped.values()))),
        "correct_response_frac": sum(sample.correct for sample in all_samples) / len(all_samples),
        "mean_correct_response_frac_per_question": sum(correct_fraction_per_group) / len(correct_fraction_per_group),
        "bleu_similarity": nanmean(group_bleu),
        "bleu_diversity": 1.0 - nanmean(group_bleu),
        "correct_only_bleu_similarity": nanmean(group_bleu_correct),
        "correct_only_bleu_diversity": 1.0 - nanmean(group_bleu_correct),
        "sbert_diversity": nanmean(group_sbert),
        "correct_only_sbert_diversity": nanmean(group_sbert_correct),
        "distinct_1": distinct_ngram_ratio(all_responses, 1),
        "distinct_2": distinct_ngram_ratio(all_responses, 2),
        "distinct_3": distinct_ngram_ratio(all_responses, 3),
        "distinct_4": distinct_ngram_ratio(all_responses, 4),
        "correct_distinct_1": distinct_ngram_ratio([s.response for s in all_samples if s.correct], 1),
        "correct_distinct_2": distinct_ngram_ratio([s.response for s in all_samples if s.correct], 2),
        "correct_distinct_3": distinct_ngram_ratio([s.response for s in all_samples if s.correct], 3),
        "correct_distinct_4": distinct_ngram_ratio([s.response for s in all_samples if s.correct], 4),
    }
    return result


def parse_specs(spec_args: list[str], preset: str | None) -> list[tuple[str, str]]:
    specs = []
    if preset:
        if preset != "llama3b_amc23":
            raise ValueError(f"Unknown preset: {preset}")
        specs.extend(DEFAULT_SPECS.items())

    for spec in spec_args:
        if "=" not in spec:
            raise ValueError(f"Spec must look like label=/path/to/file.jsonl, got: {spec}")
        label, path = spec.split("=", 1)
        specs.append((label.strip(), path.strip()))

    if not specs:
        raise ValueError("No specs provided. Use --preset llama3b_amc23 or --spec label=path")
    return specs


def format_value(value) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def write_csv(path: str, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_sbert_model_path(model_name_or_path: str) -> str:
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    if model_name_or_path == "sentence-transformers/all-mpnet-base-v2" and os.path.isdir(DEFAULT_SBERT_LOCAL_PATH):
        return DEFAULT_SBERT_LOCAL_PATH
    return model_name_or_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare diversity metrics across math-eval JSONL files.")
    parser.add_argument("--preset", choices=["llama3b_amc23"])
    parser.add_argument("--spec", action="append", default=[], help="label=/path/to/file.jsonl")
    parser.add_argument("--group_key", default="question_id")
    parser.add_argument("--sbert_model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--hf_cache", default="/scratch/prj0000000224/hf_cache")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_bleu_pairs", type=int, default=4096)
    parser.add_argument("--bleu_seed", type=int, default=0)
    parser.add_argument("--output_csv")
    args = parser.parse_args()

    specs = parse_specs(args.spec, args.preset)
    for label, path in specs:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label}: missing file {path}")

    os.environ.setdefault("HF_HOME", args.hf_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(args.hf_cache, "hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    model_name_or_path = resolve_sbert_model_path(args.sbert_model)
    model = SentenceTransformer(model_name_or_path, cache_folder=args.hf_cache, local_files_only=True)

    rows = []
    for label, path in specs:
        rows.append(
            analyze_one(
                label=label,
                path=path,
                model=model,
                batch_size=args.batch_size,
                max_bleu_pairs=args.max_bleu_pairs,
                bleu_seed=args.bleu_seed,
                group_key=args.group_key,
            )
        )

    print(",".join(rows[0].keys()))
    for row in rows:
        print(",".join(format_value(row[key]) for key in row.keys()))

    if args.output_csv:
        write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
