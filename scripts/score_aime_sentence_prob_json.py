#!/usr/bin/env python3
import argparse
import json
import math
import os
from dataclasses import asdict, dataclass

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SPECS = {
    "grpo": (
        "grpo",
        "/scratch/prj0000000224/models/MATH-Llama-3.2-3B-GRPO/global_step_160/actor/huggingface",
        "/scratch/prj0000000224/eval_outputs_full/eval-llama3b-grpo-step160-55495/MATH-Llama-3.2-3B-GRPO-step160/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl",
    ),
    "simko_rerun": (
        "simko_rerun",
        "/scratch/prj0000000224/models/MATH-Llama-3.2-3B-SimKO-Rerun/global_step_160/actor/huggingface",
        "/scratch/prj0000000224/eval_outputs_full/eval-llama3b-simko-rerun-2g-testqueue-20260418-095356/MATH-Llama-3.2-3B-SimKO-Rerun-step160/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl",
    ),
    "ts2_rerun": (
        "ts2_rerun",
        "/scratch/prj0000000224/models/MATH-Llama-3.2-3B-SimKO-TS2-Rerun/global_step_160/actor/huggingface",
        "/scratch/prj0000000224/eval_outputs_full/eval-llama3b-ts2-rerun-2g-testqueue-20260418-095401/MATH-Llama-3.2-3B-SimKO-TS2-Rerun-step160/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl",
    ),
    "mix001": (
        "mix001",
        "/scratch/prj0000000224/models/MATH-Llama-3.2-3B-SimKO-TS2-nosquare-mix001/global_step_160/actor/huggingface",
        "/scratch/prj0000000224/eval_outputs_full/eval-llama3b-ts2-nosquare-mix001-step160/MATH-Llama-3.2-3B-SimKO-TS2-nosquare-mix001-step160_rerun/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl",
    ),
    "mix005": (
        "mix005",
        "/scratch/prj0000000224/models/MATH-Llama-3.2-3B-SimKO-TS2-nosquare-mix005/global_step_160/actor/huggingface",
        "/scratch/prj0000000224/eval_outputs_full/eval-llama3b-ts2-nosquare-mix005-step160/MATH-Llama-3.2-3B-SimKO-TS2-nosquare-mix005-step160_rerun/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl",
    ),
    "mix01": (
        "mix01",
        "/scratch/prj0000000224/models/MATH-Llama-3.2-3B-SimKO-TS2-nosquare-mix01/global_step_160/actor/huggingface",
        "/scratch/prj0000000224/eval_outputs_full/eval-llama3b-ts2-nosquare-mix01-step160/MATH-Llama-3.2-3B-SimKO-TS2-nosquare-mix01-step160_rerun/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl",
    ),
    "kingbaicai_ts2": (
        "kingbaicai_ts2",
        "/scratch/prj0000000224/models/Kingbaicai_TS2-method",
        "/scratch/prj0000000224/eval_outputs_full/eval-kingbaicai-ts2-aime-amc-60561/Kingbaicai_TS2-method/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl",
    ),
}


@dataclass
class EvalRecord:
    sample_index: int
    question_id: int
    generation_id: int
    prompt: str
    response: str


def parse_spec(spec: str) -> tuple[str, str, str]:
    parts = spec.split("::")
    if len(parts) != 3:
        raise ValueError(
            "Each --spec must look like "
            "label::/path/to/model_dir::/path/to/eval.jsonl"
        )
    label, model_dir, eval_file = (part.strip() for part in parts)
    if not label:
        raise ValueError(f"Spec has empty label: {spec}")
    return label, model_dir, eval_file


def resolve_spec(spec_arg: str | None, preset: str | None) -> tuple[str, str, str]:
    if spec_arg and preset:
        raise ValueError("Use either --spec or --preset, not both.")
    if preset:
        if preset not in DEFAULT_SPECS:
            raise ValueError(f"Unknown preset: {preset}")
        return DEFAULT_SPECS[preset]
    if spec_arg:
        return parse_spec(spec_arg)
    raise ValueError("Provide --preset or --spec.")


def load_eval_records(path: str, show_progress: bool) -> list[EvalRecord]:
    records = []
    with open(path) as f:
        for sample_index, line in enumerate(
            tqdm(f, desc=f"Loading {os.path.basename(path)}", dynamic_ncols=True, disable=not show_progress)
        ):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records.append(
                EvalRecord(
                    sample_index=sample_index,
                    question_id=int(row.get("question_id", -1)),
                    generation_id=int(row.get("generation_id", sample_index)),
                    prompt=str(row["prompt"]),
                    response=str(row["response"]),
                )
            )
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
        raise ValueError("This script requires a fast tokenizer to use offset mappings.")
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


def compute_sentence_prob_points(
    records: list[EvalRecord],
    model,
    tokenizer,
    batch_size: int,
    show_progress: bool,
) -> list[dict]:
    device = next(model.parameters()).device
    total_batches = math.ceil(len(records) / batch_size)
    iterator = chunked(records, batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=total_batches, desc="Scoring responses", dynamic_ncols=True)

    points = []
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
            logits = outputs.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            selected_logits = shift_logits.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            normalizer = torch.logsumexp(shift_logits.float(), dim=-1)
            token_probs = torch.exp(selected_logits.float() - normalizer).cpu()

        attention_mask_cpu = attention_mask.cpu()
        for row_idx, record in enumerate(batch_records):
            valid_len = int(attention_mask_cpu[row_idx].sum().item())
            offsets = offset_mapping[row_idx][:valid_len].tolist()
            response_positions = [
                pos for pos, (start, end) in enumerate(offsets)
                if end > start and start >= prompt_char_lens[row_idx]
            ]
            response_positions = [pos for pos in response_positions if pos > 0]

            if response_positions:
                prob_indices = torch.tensor([pos - 1 for pos in response_positions], dtype=torch.long)
                sentence_prob = float(token_probs[row_idx].index_select(0, prob_indices).mean().item())
            else:
                sentence_prob = float("nan")

            points.append(
                {
                    "sample_index": record.sample_index,
                    "question_id": record.question_id,
                    "generation_id": record.generation_id,
                    "sentence_prob": sentence_prob,
                }
            )

        del outputs, logits, shift_logits, shift_labels, selected_logits, normalizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return points


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score one AIME eval file and dump per-response sentence probabilities to JSON."
    )
    parser.add_argument("--preset", choices=sorted(DEFAULT_SPECS))
    parser.add_argument("--spec", help="label::/path/to/model_dir::/path/to/eval.jsonl")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    label, model_dir, eval_file = resolve_spec(spec_arg=args.spec, preset=args.preset)
    show_progress = not args.no_progress

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Missing model dir: {model_dir}")
    if not os.path.isfile(eval_file):
        raise FileNotFoundError(f"Missing eval file: {eval_file}")

    records = load_eval_records(eval_file, show_progress=show_progress)
    model, tokenizer = load_model_and_tokenizer(model_dir=model_dir, dtype_name=args.dtype)
    points = compute_sentence_prob_points(
        records=records,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        show_progress=show_progress,
    )

    payload = {
        "label": label,
        "model_dir": model_dir,
        "eval_file": eval_file,
        "num_responses": len(points),
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "points": points,
        "example_record": asdict(records[0]),
    }

    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(payload, f)

    print(f"Wrote {len(points)} points to: {args.output_json}")


if __name__ == "__main__":
    main()
