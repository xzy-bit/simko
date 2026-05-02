#!/usr/bin/env python3
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_payload(path: str) -> dict:
    with open(path) as f:
        payload = json.load(f)
    if "label" not in payload or "histograms" not in payload or "bin_edges" not in payload:
        raise ValueError(f"Invalid top-k histogram json: {path}")
    return payload


def parse_input(arg: str) -> tuple[str | None, str]:
    if "=" in arg:
        label, path = arg.split("=", 1)
        return label.strip(), path.strip()
    return None, arg.strip()


def plot_compare(payloads: list[dict], output_png: str, title: str | None) -> None:
    top_k = payloads[0]["top_k"]
    bin_edges = payloads[0]["bin_edges"]
    num_bins = len(bin_edges) - 1
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2.0 for i in range(num_bins)]
    bin_width = bin_edges[1] - bin_edges[0]

    for payload in payloads[1:]:
        if payload["top_k"] != top_k or payload["bin_edges"] != bin_edges:
            raise ValueError("All inputs must use the same top_k and bin edges.")

    fig, axes = plt.subplots(2, 3, figsize=(22, 10), squeeze=False)
    flat_axes = axes.flatten()
    colors = list(plt.get_cmap("tab10").colors)
    bar_width = min(bin_width / max(len(payloads), 1) * 1.35, bin_width * 0.95)

    for rank_idx in range(top_k):
        ax = flat_axes[rank_idx]
        for model_idx, payload in enumerate(payloads):
            label = payload["label"]
            hist = payload["histograms"][rank_idx]
            y = hist["bin_proportions"]
            offset = (model_idx - (len(payloads) - 1) / 2.0) * bar_width
            shifted_centers = [center + offset for center in bin_centers]
            ax.bar(
                shifted_centers,
                y,
                width=bar_width,
                label=label,
                color=colors[model_idx % len(colors)],
                alpha=0.9,
                edgecolor="none",
            )

        ax.set_title(f"Rank-{rank_idx + 1}")
        ax.set_xlabel("probability")
        ax.set_ylabel("proportion")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)

    for ax in flat_axes[top_k:]:
        ax.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0.1, 1, 0.96))
    else:
        fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=min(len(payloads), 4), frameon=False, fontsize=11)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare top-k candidate probability histograms from multiple model json files."
    )
    parser.add_argument(
        "--input_json",
        action="append",
        required=True,
        help="Either /path/to/file.json or label=/path/to/file.json",
    )
    parser.add_argument("--output_png", required=True)
    parser.add_argument("--title")
    args = parser.parse_args()

    payloads = []
    for item in args.input_json:
        label_override, path = parse_input(item)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing input json: {path}")
        payload = load_payload(path)
        if label_override:
            payload["label"] = label_override
        payloads.append(payload)

    output_dir = os.path.dirname(args.output_png)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plot_compare(payloads, args.output_png, args.title)
    print(f"Wrote comparison plot to: {args.output_png}")


if __name__ == "__main__":
    main()
