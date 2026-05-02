#!/usr/bin/env python3
import argparse
import json
import math
import os
from math import ceil
from statistics import mean, pvariance

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_payload(path: str) -> dict:
    with open(path) as f:
        payload = json.load(f)
    if "label" not in payload or "points" not in payload:
        raise ValueError(f"Invalid score JSON: {path}")
    return payload


def parse_score_arg(arg: str) -> tuple[str | None, str]:
    if "=" in arg:
        label, path = arg.split("=", 1)
        return label.strip(), path.strip()
    return None, arg.strip()


def get_valid_pairs(payload: dict) -> tuple[list[int], list[float]]:
    valid_pairs = [
        (point["sample_index"], point["sentence_prob"])
        for point in payload["points"]
        if not math.isnan(point["sentence_prob"])
    ]
    x = [pair[0] for pair in valid_pairs]
    y = [pair[1] for pair in valid_pairs]
    return x, y


def summarize_probs(y: list[float]) -> tuple[float, float]:
    if not y:
        return float("nan"), float("nan")
    if len(y) == 1:
        return y[0], 0.0
    return mean(y), pvariance(y)


def plot_payloads(payloads: list[dict], output_path: str, title: str | None) -> None:
    num_models = len(payloads)
    fig, axes = plt.subplots(num_models, 2, figsize=(14, 4 * num_models), squeeze=False)

    for row_idx, payload in enumerate(payloads):
        label = payload["label"]
        x, y = get_valid_pairs(payload)
        y_mean, y_var = summarize_probs(y)

        scatter_ax = axes[row_idx][0]
        hist_ax = axes[row_idx][1]

        scatter_ax.scatter(x, y, s=8, alpha=0.55)
        scatter_ax.set_title(f"{label}: scatter\nmean={y_mean:.4f} var={y_var:.4f}")
        scatter_ax.set_xlabel("response index")
        scatter_ax.set_ylabel("mean token prob")
        scatter_ax.set_ylim(0.0, 1.0)
        scatter_ax.grid(alpha=0.2)

        hist_ax.hist(y, bins=80, alpha=0.8)
        hist_ax.set_title(f"{label}: histogram\nmean={y_mean:.4f} var={y_var:.4f}")
        hist_ax.set_xlabel("mean token prob")
        hist_ax.set_ylabel("count")
        hist_ax.set_xlim(0.0, 1.0)
        hist_ax.grid(alpha=0.2)

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
    else:
        fig.tight_layout()

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_only(payloads: list[dict], output_path: str, title: str | None, ncols: int) -> None:
    num_models = len(payloads)
    ncols = max(1, ncols)
    nrows = ceil(num_models / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.8 * nrows), squeeze=False)
    flat_axes = axes.flatten()

    for ax, payload in zip(flat_axes, payloads):
        label = payload["label"]
        x, y = get_valid_pairs(payload)
        y_mean, y_var = summarize_probs(y)
        ax.scatter(x, y, s=8, alpha=0.55)
        ax.set_title(f"{label}\nmean={y_mean:.4f} var={y_var:.4f}")
        ax.set_xlabel("response index")
        ax.set_ylabel("mean token prob")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)

    for ax in flat_axes[num_models:]:
        ax.axis("off")

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
    else:
        fig.tight_layout()

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot sentence probability scatter/histogram from score JSON files."
    )
    parser.add_argument(
        "--score_json",
        action="append",
        required=True,
        help="Either /path/to/file.json or label=/path/to/file.json",
    )
    parser.add_argument("--output_png", required=True)
    parser.add_argument("--title")
    parser.add_argument("--scatter_only", action="store_true")
    parser.add_argument("--ncols", type=int, default=3, help="Only used with --scatter_only.")
    args = parser.parse_args()

    payloads = []
    for score_arg in args.score_json:
        label_override, path = parse_score_arg(score_arg)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing score json: {path}")
        payload = load_payload(path)
        if label_override:
            payload["label"] = label_override
        payloads.append(payload)

    output_dir = os.path.dirname(args.output_png)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if args.scatter_only:
        plot_scatter_only(payloads, output_path=args.output_png, title=args.title, ncols=args.ncols)
    else:
        plot_payloads(payloads, output_path=args.output_png, title=args.title)
    print(f"Wrote plot to: {args.output_png}")


if __name__ == "__main__":
    main()
