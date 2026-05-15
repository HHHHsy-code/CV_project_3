from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
ASSETS_DIR = REPORT_DIR / "assets"
GENERATED_DIR = ASSETS_DIR / "generated"
DATA_DIR = REPORT_DIR / "data"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_benchmark_metrics_pdf() -> Path:
    data = load_json(DATA_DIR / "benchmark_mask_metrics.json")["sequences"]
    labels = [entry["name"] for entry in data]
    jm = [entry["jm"] for entry in data]
    jr = [entry["jr"] for entry in data]
    precision = [entry["precision"] for entry in data]

    x = np.arange(len(labels))
    width = 0.22

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    colors = ["#3567a8", "#68a357", "#d28b26"]
    bars = [
        ax.bar(x - width, jm, width, label=r"$J_M$", color=colors[0]),
        ax.bar(x, jr, width, label=r"$J_R$", color=colors[1]),
        ax.bar(x + width, precision, width, label="Precision", color=colors[2]),
    ]

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Mask Metrics on Annotated Benchmark Sequences")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", ncol=3, frameon=False)

    for group in bars:
        for bar in group:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    output = GENERATED_DIR / "benchmark_mask_metrics.pdf"
    ensure_dir(output.parent)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return output


def save_wild_ablation_metrics_pdf() -> Path:
    data = load_json(DATA_DIR / "wild_video2_ablation_metrics.json")["branches"]
    labels = [entry["name"] for entry in data]
    coverage = [entry["temporal_coverage"] for entry in data]
    non_empty = [entry["non_empty_frames"] for entry in data]
    area = [100.0 * entry["mean_area_ratio_all"] for entry in data]

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), constrained_layout=True)
    palette = ["#7a8fb0", "#4575b4", "#6b9e5b", "#b5872e"]
    metrics = [
        ("Temporal coverage", coverage, (0.45, 0.70), "{:.3f}"),
        ("Non-empty frames", non_empty, (90, 130), "{:.0f}"),
        ("Mean area ratio (%)", area, (0.8, 1.9), "{:.2f}"),
    ]

    for ax, (title, values, ylim, fmt) in zip(axes, metrics):
        bars = ax.bar(labels, values, color=palette, width=0.68)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="x", rotation=18)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (ylim[1] - ylim[0]) * 0.03,
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    output = GENERATED_DIR / "wild_video2_ablation_metrics.pdf"
    ensure_dir(output.parent)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return output


@dataclass(frozen=True)
class GridSpec:
    path: Path
    title: str
    row_labels: list[str]
    keep_rows: list[int]


def _extract_grid_rows(path: Path, rows: int = 6, cols: int = 4) -> list[np.ndarray]:
    image = np.asarray(Image.open(path).convert("RGB"))
    height, width = image.shape[:2]
    if width % cols != 0:
        raise RuntimeError(f"Unexpected width for grid image: {path}")
    cell_width = width // cols
    cell_height = height // rows - 80
    if cell_height <= 0:
        raise RuntimeError(f"Failed to infer grid cell height for {path}")

    extracted_rows: list[np.ndarray] = []
    for row_idx in range(rows):
        start_y = row_idx * (cell_height + 80) + 80
        end_y = start_y + cell_height
        row = image[start_y:end_y, :, :]
        extracted_rows.append(row)
    return extracted_rows


def _draw_sequence_block(fig: plt.Figure, slot, spec: GridSpec, column_titles: list[str]) -> None:
    row_images = _extract_grid_rows(spec.path)
    visible_rows = [row_images[idx] for idx in spec.keep_rows]

    block = slot.subgridspec(
        len(visible_rows) + 2,
        len(column_titles),
        height_ratios=[0.22, 0.16] + [1.0] * len(visible_rows),
        hspace=0.05,
        wspace=0.02,
    )

    title_ax = fig.add_subplot(block[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.5, spec.title, fontsize=14, fontweight="bold", ha="left", va="center")

    header_ax = fig.add_subplot(block[1, :])
    header_ax.axis("off")
    for col_idx, title in enumerate(column_titles):
        header_ax.text(
            (col_idx + 0.5) / len(column_titles),
            0.45,
            title,
            fontsize=11,
            fontweight="semibold",
            ha="center",
            va="center",
        )

    for row_idx, row_image in enumerate(visible_rows):
        frame_label = spec.row_labels[row_idx]
        cell_width = row_image.shape[1] // len(column_titles)
        for col_idx in range(len(column_titles)):
            ax = fig.add_subplot(block[row_idx + 2, col_idx])
            start_x = col_idx * cell_width
            end_x = start_x + cell_width
            ax.imshow(row_image[:, start_x:end_x, :])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("#999999")
            if col_idx == 0:
                ax.set_ylabel(frame_label, fontsize=11, rotation=90, labelpad=18, va="center")


def save_qualitative_pair_pdf(output_name: str, specs: list[GridSpec]) -> Path:
    fig = plt.figure(figsize=(12.2, 10.8), constrained_layout=True)
    outer = fig.add_gridspec(len(specs), 1, hspace=0.18)
    titles = ["Original", "Mask", "Part 1", "Part 2"]
    for slot, spec in zip(outer, specs):
        _draw_sequence_block(fig, slot, spec, titles)

    output = GENERATED_DIR / output_name
    ensure_dir(output.parent)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    save_benchmark_metrics_pdf()
    save_wild_ablation_metrics_pdf()
    save_qualitative_pair_pdf(
        "course_sequence_comparisons.pdf",
        [
            GridSpec(
                path=ASSETS_DIR / "bmx_trees_part1_vs_part2.png",
                title="Controlled comparison on bmx-trees",
                row_labels=["Sample 1", "Sample 2", "Sample 3", "Sample 4"],
                keep_rows=[1, 2, 3, 4],
            ),
            GridSpec(
                path=ASSETS_DIR / "tennis_part1_vs_part2.png",
                title="Controlled comparison on tennis",
                row_labels=["Sample 1", "Sample 2", "Sample 3", "Sample 4"],
                keep_rows=[1, 2, 3, 4],
            ),
        ],
    )
    save_qualitative_pair_pdf(
        "wild_sequence_comparisons.pdf",
        [
            GridSpec(
                path=ASSETS_DIR / "wild_video1_part1_vs_part2.png",
                title="Real-scene success case on wild_video1",
                row_labels=["Sample 1", "Sample 2", "Sample 3", "Sample 4"],
                keep_rows=[1, 2, 3, 4],
            ),
            GridSpec(
                path=ASSETS_DIR / "wild_video2_part1_vs_sam2_part2.png",
                title="Failure-to-improvement case on wild_video2",
                row_labels=["Sample 1", "Sample 2", "Sample 3", "Sample 4"],
                keep_rows=[1, 2, 3, 4],
            ),
        ],
    )


if __name__ == "__main__":
    main()
