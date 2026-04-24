#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import Counter, defaultdict


ANSWER_CANDIDATE_FIELDS = ("pred_answer", "answer", "final_answer")
RESPONSE_CANDIDATE_FIELDS = ("response", "vanilla_response", "completion", "output")
QUESTION_CANDIDATE_FIELDS = ("question", "problem", "prompt")
CORRECT_LABEL_FIELDS = ("label", "correct", "is_correct")


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    return re.sub(r"\s+", " ", text)


def normalize_answer(text: str) -> str:
    text = normalize_text(text)
    text = text.strip("$")
    text = re.sub(r"^\\boxed\{(.*)\}$", r"\1", text)
    text = re.sub(r"\s+", "", text)
    return text.lower()


def pick_first(record: dict, candidates: tuple[str, ...], default: str = "") -> str:
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


def entropy_from_counter(counter: Counter, total: int) -> float:
    probs = [count / total for count in counter.values()]
    return -sum(p * math.log(p + 1e-12) for p in probs)


def summarize_group(rows: list[dict]) -> dict:
    n = len(rows)
    answers = [normalize_answer(pick_first(row, ANSWER_CANDIDATE_FIELDS)) for row in rows]
    responses = [normalize_text(pick_first(row, RESPONSE_CANDIDATE_FIELDS)) for row in rows]
    answer_counter = Counter(answers)
    response_counter = Counter(responses)
    correct_answers = [answer for row, answer in zip(rows, answers) if is_correct(row)]

    return {
        "num_samples": n,
        "distinct_answers": len(answer_counter),
        "distinct_responses": len(response_counter),
        "answer_entropy": entropy_from_counter(answer_counter, n),
        "majority_answer_frac": max(answer_counter.values()) / n,
        "answer_counter": answer_counter,
        "num_correct_samples": len(correct_answers),
        "distinct_correct_answers": len(set(correct_answers)),
        "question_preview": normalize_text(pick_first(rows[0], QUESTION_CANDIDATE_FIELDS))[:180],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze diversity of repeated generations in a JSONL eval file.")
    parser.add_argument("--file_path", required=True, help="Path to the JSONL eval output.")
    parser.add_argument("--group_key", default="question_id", help="Field used to group repeated generations.")
    parser.add_argument("--top_n", type=int, default=8, help="How many groups to show for collapsed/diverse examples.")
    args = parser.parse_args()

    grouped = defaultdict(list)
    with open(args.file_path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if args.group_key not in record:
                raise KeyError(f"Missing group key '{args.group_key}' at line {line_no}")
            grouped[record[args.group_key]].append(record)

    if not grouped:
        raise ValueError("No records found.")

    summaries = {group_id: summarize_group(rows) for group_id, rows in grouped.items()}
    sample_sizes = sorted({summary["num_samples"] for summary in summaries.values()})

    mean_distinct_answers = sum(s["distinct_answers"] for s in summaries.values()) / len(summaries)
    mean_distinct_responses = sum(s["distinct_responses"] for s in summaries.values()) / len(summaries)
    mean_answer_entropy = sum(s["answer_entropy"] for s in summaries.values()) / len(summaries)
    mean_majority_frac = sum(s["majority_answer_frac"] for s in summaries.values()) / len(summaries)
    mean_answer_ratio = sum(s["distinct_answers"] / s["num_samples"] for s in summaries.values()) / len(summaries)
    mean_response_ratio = sum(s["distinct_responses"] / s["num_samples"] for s in summaries.values()) / len(summaries)
    pass_at_k = sum(s["num_correct_samples"] > 0 for s in summaries.values()) / len(summaries)
    mean_correct_samples = sum(s["num_correct_samples"] for s in summaries.values()) / len(summaries)
    mean_distinct_correct_answers = sum(s["distinct_correct_answers"] for s in summaries.values()) / len(summaries)

    global_answer_counter = Counter()
    for summary in summaries.values():
        global_answer_counter.update(summary["answer_counter"])

    collapsed = sorted(
        (
            (
                s["distinct_answers"],
                -s["majority_answer_frac"],
                group_id,
                s["answer_counter"].most_common(5),
                s["question_preview"],
            )
            for group_id, s in summaries.items()
        )
    )
    diverse = sorted(
        (
            (
                -s["distinct_answers"],
                s["majority_answer_frac"],
                group_id,
                s["answer_counter"].most_common(5),
                s["question_preview"],
            )
            for group_id, s in summaries.items()
        )
    )

    print(f"file: {args.file_path}")
    print(f"num_groups: {len(summaries)}")
    print(f"samples_per_group_set: {sample_sizes}")
    print(f"mean_distinct_pred_answer: {mean_distinct_answers:.6f}")
    print(f"mean_distinct_response: {mean_distinct_responses:.6f}")
    print(f"mean_answer_entropy_nats: {mean_answer_entropy:.6f}")
    print(f"mean_majority_answer_frac: {mean_majority_frac:.6f}")
    print(f"mean_answer_distinct_ratio: {mean_answer_ratio:.6f}")
    print(f"mean_response_distinct_ratio: {mean_response_ratio:.6f}")
    print(f"pass_at_k_empirical: {pass_at_k:.6f}")
    print(f"mean_num_correct_samples_per_group: {mean_correct_samples:.6f}")
    print(f"mean_distinct_correct_answers_per_group: {mean_distinct_correct_answers:.6f}")
    print(f"groups_all_same_pred_answer: {sum(s['distinct_answers'] == 1 for s in summaries.values())}")
    print(
        "groups_all_unique_response: "
        f"{sum(s['distinct_responses'] == s['num_samples'] for s in summaries.values())}"
    )
    print(f"top10_pred_answer_global: {global_answer_counter.most_common(10)}")

    print("\nmost_collapsed:")
    for distinct_answers, neg_majority_frac, group_id, top_answers, preview in collapsed[: args.top_n]:
        print(
            f"group={group_id} distinct_answers={distinct_answers} "
            f"majority_answer_frac={-neg_majority_frac:.6f} top_answers={top_answers} "
            f"question={preview}"
        )

    print("\nmost_diverse:")
    for neg_distinct_answers, majority_frac, group_id, top_answers, preview in diverse[: args.top_n]:
        print(
            f"group={group_id} distinct_answers={-neg_distinct_answers} "
            f"majority_answer_frac={majority_frac:.6f} top_answers={top_answers} "
            f"question={preview}"
        )


if __name__ == "__main__":
    main()
