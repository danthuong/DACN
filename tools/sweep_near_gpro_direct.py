#!/usr/bin/env python3
"""Paired current-vs-direct Near-GPro checkpoint sweep with an SVG report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.datasets.kitti.near_gpro_dataset import build_near_gpro_dataloader
from lib.helpers.config_helper import load_config
from lib.helpers.near_gpro_runtime import NearGProTester
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.utils_helper import create_logger, set_random_seed
from lib.models.monodetr.near_gpro_zmap import build_near_gpro_zmap
from tools.evaluate_near_gpro_depth import evaluate as evaluate_depth


COLORS = {
    "current": "#d95f02",
    "direct": "#1b9e77",
    "gpro": "#7570b3",
    "baseline": "#666666",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--checkpoint-dir",
        help="Default: TRAINER_SAVE_PATH/MODEL_NAME from the resolved config.",
    )
    parser.add_argument(
        "--checkpoint", action="append", default=[],
        help=(
            "Explicit checkpoint path; repeat to select a subset. "
            "Default sweeps best, last, and epoch_N."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help="Summary/report directory (default: CHECKPOINT_DIR/direct_sweep).",
    )
    parser.add_argument(
        "--result-prefix", default="direct_sweep",
        help="Subdirectory below CHECKPOINT_DIR/outputs for KITTI predictions.",
    )
    parser.add_argument("--pseudo-labels")
    parser.add_argument("--ground-truth-dir")
    parser.add_argument("--split-file")
    parser.add_argument("--class-name", default="Car")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--allow-missing-artifacts", action="store_true",
        help="Allow incomplete outputs when producing depth reports.",
    )
    return parser.parse_args()


def checkpoint_epoch(path):
    match = re.search(r"checkpoint_epoch_(\d+)\.pth$", path.name)
    if match:
        return int(match.group(1))
    payload = torch.load(str(path), map_location="cpu")
    return int(payload.get("epoch", -1))


def discover_checkpoints(args, checkpoint_dir):
    if args.checkpoint:
        paths = [Path(value).expanduser().resolve() for value in args.checkpoint]
    else:
        paths = []
        for name in ("checkpoint_best.pth", "checkpoint_last.pth"):
            candidate = checkpoint_dir / name
            if candidate.is_file():
                paths.append(candidate)
        paths.extend(sorted(
            checkpoint_dir.glob("checkpoint_epoch_*.pth"),
            key=lambda item: int(re.search(r"(\d+)$", item.stem).group(1)),
        ))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints: " + ", ".join(missing))
    if not paths:
        raise FileNotFoundError(
            "No best, last, or epoch_N checkpoints in " + str(checkpoint_dir)
        )

    # best/last/periodic files can contain the same epoch. Evaluate it once.
    by_epoch = {}
    for path in paths:
        epoch = checkpoint_epoch(path)
        by_epoch.setdefault(epoch, []).append(path)
    entries = []
    for epoch, epoch_paths in sorted(by_epoch.items()):
        preferred = sorted(
            epoch_paths,
            key=lambda path: (
                0 if path.name == "checkpoint_best.pth" else
                1 if path.name.startswith("checkpoint_epoch_") else 2,
                path.name,
            ),
        )[0]
        entries.append({
            "epoch": epoch,
            "path": preferred,
            "aliases": [path.name for path in epoch_paths],
        })
    return entries


def finite(value):
    return value is not None and math.isfinite(float(value))


def metric(report, branch, name):
    return report.get("branches", {}).get(branch, {}).get(name)


def fmt(value, digits=3):
    if not finite(value):
        return "—"
    return ("{:." + str(digits) + "f}").format(value)


def svg_chart(rows, output_path):
    width, height = 1440, 1040
    left, right = 105, 35
    panel_height, panel_gap = 245, 75
    top = 95
    plot_width = width - left - right
    epochs = [row["epoch"] for row in rows]
    x_min, x_max = min(epochs), max(epochs)
    if x_min == x_max:
        x_min, x_max = x_min - 1, x_max + 1

    panels = (
        (
            "KITTI AP3D R40 Moderate (higher is better)",
            (("Current fusion", "ap_current", COLORS["current"]),
             ("Direct GPro", "ap_direct", COLORS["direct"])),
        ),
        (
            "Matched center-depth MAE [m] (lower is better)",
            (("Current final", "mae_current", COLORS["current"]),
             ("Direct final", "mae_direct", COLORS["direct"]),
             ("GPro valid-only", "mae_gpro", COLORS["gpro"]),
             ("MonoDETR baseline", "mae_baseline", COLORS["baseline"])),
        ),
        (
            "Matched center-depth P90 absolute error [m] (lower is better)",
            (("Current final", "p90_current", COLORS["current"]),
             ("Direct final", "p90_direct", COLORS["direct"]),
             ("GPro valid-only", "p90_gpro", COLORS["gpro"])),
        ),
    )

    chunks = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" '
        'viewBox="0 0 {} {}">'.format(width, height, width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}'
        '.title{font-size:24px;font-weight:700}'
        '.panel{font-size:18px;font-weight:700}'
        '.tick{font-size:13px;fill:#555}.legend{font-size:13px}'
        '.point{stroke:#fff;stroke-width:1.5}</style>',
        '<text class="title" x="{}" y="38">'
        'Near-GPro direct replacement checkpoint sweep</text>'.format(left),
        '<text class="tick" x="{}" y="62">'
        'Current and direct share identical queries, boxes, scores, '
        'dimensions and orientations.</text>'.format(left),
    ]

    for panel_index, (title, series) in enumerate(panels):
        panel_top = top + panel_index * (panel_height + panel_gap)
        panel_bottom = panel_top + panel_height
        all_values = [
            float(row[key]) for _, key, _ in series for row in rows
            if finite(row.get(key))
        ]
        y_max = max(all_values) * 1.12 if all_values else 1.0
        y_max = max(y_max, 1.0)

        def sx(epoch):
            return left + (float(epoch) - x_min) / (x_max - x_min) * plot_width

        def sy(value):
            return panel_bottom - float(value) / y_max * panel_height

        chunks.append(
            '<text class="panel" x="{}" y="{}">{}</text>'.format(
                left, panel_top - 17, html.escape(title)
            )
        )
        for tick in range(6):
            value = y_max * tick / 5.0
            y = sy(value)
            chunks.append(
                '<line x1="{}" y1="{:.2f}" x2="{}" y2="{:.2f}" '
                'stroke="#e5e7eb"/>'.format(left, y, width - right, y)
            )
            chunks.append(
                '<text class="tick" x="{}" y="{:.2f}" '
                'text-anchor="end">{:.2f}</text>'.format(
                    left - 10, y + 5, value
                )
            )
        chunks.append(
            '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#555"/>'.format(
                left, panel_bottom, width - right, panel_bottom
            )
        )
        tick_epochs = sorted(set(epochs))
        if len(tick_epochs) > 12:
            stride = int(math.ceil(len(tick_epochs) / 12.0))
            tick_epochs = tick_epochs[::stride]
            if tick_epochs[-1] != epochs[-1]:
                tick_epochs.append(epochs[-1])
        for epoch in tick_epochs:
            x = sx(epoch)
            chunks.append(
                '<line x1="{:.2f}" y1="{}" x2="{:.2f}" y2="{}" '
                'stroke="#777"/>'.format(
                    x, panel_bottom, x, panel_bottom + 5
                )
            )
            chunks.append(
                '<text class="tick" x="{:.2f}" y="{}" '
                'text-anchor="middle">{}</text>'.format(
                    x, panel_bottom + 22, epoch
                )
            )

        legend_x = left
        for label, _, color in series:
            chunks.append(
                '<line x1="{}" y1="{}" x2="{}" y2="{}" '
                'stroke="{}" stroke-width="3"/>'.format(
                    legend_x, panel_top + 10, legend_x + 25,
                    panel_top + 10, color
                )
            )
            chunks.append(
                '<text class="legend" x="{}" y="{}">{}</text>'.format(
                    legend_x + 32, panel_top + 15, html.escape(label)
                )
            )
            legend_x += 70 + len(label) * 7

        for label, key, color in series:
            points = [
                (sx(row["epoch"]), sy(row[key]), row)
                for row in rows if finite(row.get(key))
            ]
            if len(points) > 1:
                chunks.append(
                    '<polyline fill="none" stroke="{}" stroke-width="3" '
                    'points="{}"/>'.format(
                        color,
                        " ".join(
                            "{:.2f},{:.2f}".format(x, y)
                            for x, y, _ in points
                        ),
                    )
                )
            for x, y, row in points:
                tooltip = "{} | epoch {} | {:.4f}".format(
                    label, row["epoch"], row[key]
                )
                chunks.append(
                    '<circle class="point" cx="{:.2f}" cy="{:.2f}" r="5" '
                    'fill="{}"><title>{}</title></circle>'.format(
                        x, y, color, html.escape(tooltip)
                    )
                )
        chunks.append(
            '<text class="tick" x="{}" y="{}" text-anchor="middle">'
            'Epoch</text>'.format(
                left + plot_width / 2, panel_bottom + 45
            )
        )

    chunks.append("</svg>")
    output_path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def write_reports(rows, output_dir, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"protocol": metadata, "checkpoints": rows}
    (output_dir / "near_gpro_direct_sweep.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    if rows:
        with (output_dir / "near_gpro_direct_sweep.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    svg_chart(rows, output_dir / "near_gpro_direct_sweep.svg")

    best_current = max(
        rows,
        key=lambda row: (
            row["ap_current"] if finite(row["ap_current"]) else -math.inf
        ),
    )
    best_direct = max(
        rows,
        key=lambda row: (
            row["ap_direct"] if finite(row["ap_direct"]) else -math.inf
        ),
    )
    best_gpro = min(
        rows,
        key=lambda row: (
            row["mae_gpro"] if finite(row["mae_gpro"]) else math.inf
        ),
    )
    lines = [
        "# Near-GPro direct replacement sweep",
        "",
        "![AP and depth-error curves](near_gpro_direct_sweep.svg)",
        "",
        (
            "Both variants use the same checkpoint forward pass, selected "
            "queries, 2D/3D heads, class scores, dimensions, orientations, "
            "and uncertainty score. Only scalar object depth differs."
        ),
        "",
        "- Best current AP: **{}** at epoch {}.".format(
            fmt(best_current["ap_current"], 4), best_current["epoch"]
        ),
        "- Best direct AP: **{}** at epoch {}.".format(
            fmt(best_direct["ap_direct"], 4), best_direct["epoch"]
        ),
        (
            "- Lowest full-validation GPro valid-only MAE: "
            "**{} m** at epoch {}."
        ).format(fmt(best_gpro["mae_gpro"], 4), best_gpro["epoch"]),
        "",
        (
            "| Epoch | Checkpoint aliases | AP current | AP direct | ΔAP | "
            "MAE current | MAE direct | ΔMAE | GPro MAE* | "
            "P90 current | P90 direct |"
        ),
        (
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: |"
        ),
    ]
    for row in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["epoch"], row["aliases"], fmt(row["ap_current"]),
                fmt(row["ap_direct"]), fmt(row["delta_ap"]),
                fmt(row["mae_current"]), fmt(row["mae_direct"]),
                fmt(row["delta_mae"]), fmt(row["mae_gpro"]),
                fmt(row["p90_current"]), fmt(row["p90_direct"]),
            )
        )
    lines.extend([
        "",
        (
            "* GPro MAE is valid-only; direct-final MAE includes the "
            "MonoDETR fallback and has full matched coverage."
        ),
        "",
        (
            "Positive ΔAP means direct replacement improves AP over the "
            "current averaged formula on the same weights."
        ),
    ])
    (output_dir / "near_gpro_direct_sweep.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    args = parse_args()
    cfg = load_config(args.config, args.env_file)
    set_random_seed(cfg.get("random_seed", 444))
    model_name = cfg["model_name"]
    default_checkpoint_dir = (
        Path(cfg["trainer"]["save_path"]) / model_name
    )
    checkpoint_dir = Path(
        args.checkpoint_dir or default_checkpoint_dir
    ).resolve()
    output_dir = Path(
        args.output_dir or checkpoint_dir / "direct_sweep"
    ).resolve()
    entries = discover_checkpoints(args, checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(str(output_dir / "sweep.log"))

    dataset_cfg = dict(cfg["dataset"])
    dataset_cfg["near_gpro"] = cfg["near_gpro"]
    _, test_loader = build_near_gpro_dataloader(
        dataset_cfg,
        workers=int(cfg.get("runtime", {}).get("num_workers", 4)),
    )
    model_cfg = dict(cfg["model"])
    model_cfg["near_gpro"] = cfg["near_gpro"]
    model, _ = build_near_gpro_zmap(model_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = [
        int(value) for value in cfg["trainer"]["gpu_ids"].split(",")
    ]
    model = (
        model.to(device)
        if len(gpu_ids) == 1
        else torch.nn.DataParallel(
            model, device_ids=gpu_ids
        ).to(device)
    )
    tester = NearGProTester(
        cfg=cfg["tester"], model=model, dataloader=test_loader,
        logger=logger, train_cfg=cfg["trainer"], model_name=model_name,
    )
    # Honor --checkpoint-dir for both checkpoint discovery and generated
    # predictions instead of silently falling back to the config save path.
    tester.output_dir = str(checkpoint_dir)

    dataset_root = Path(
        cfg["dataset"].get("root_dir", "data/KITTIDataset")
    )
    split_name = cfg["dataset"].get("test_split", "val")
    gt_dir = Path(
        args.ground_truth_dir
        or dataset_root / "training" / "label_2"
    )
    split_file = Path(
        args.split_file
        or dataset_root / "ImageSets" / (split_name + ".txt")
    )
    if args.pseudo_labels:
        pseudo_labels = Path(args.pseudo_labels)
    else:
        pseudo_root = Path(
            cfg["near_gpro"]["pseudo_labels"]["root"]
        )
        pseudo_labels = pseudo_root / (split_name + ".jsonl")

    rows = []
    metadata = {
        "config": str(Path(args.config).resolve()),
        "checkpoint_dir": str(checkpoint_dir),
        "class": args.class_name,
        "iou_threshold": args.iou_threshold,
        "split_file": str(split_file),
        "current_formula": (
            "(Zquery + Zsize + valid_or_Zmap(Zgpro)) / 3"
        ),
        "direct_formula": (
            "valid ? Zgpro_center : Zbaseline_original"
        ),
    }
    for index, entry in enumerate(entries, start=1):
        epoch = entry["epoch"]
        logger.info(
            "[%d/%d] epoch %d: %s",
            index, len(entries), epoch, entry["path"],
        )
        load_checkpoint(
            model=model, optimizer=None, filename=str(entry["path"]),
            map_location=device, logger=logger,
        )
        tag = "epoch_{:04d}".format(max(epoch, 0))
        current_name = "{}/{}/current".format(
            args.result_prefix, tag
        )
        direct_name = "{}/{}/direct".format(
            args.result_prefix, tag
        )
        tester.inference((
            ("current", current_name),
            ("direct", direct_name),
        ))

        reports = {}
        for variant, output_name in (
                ("current", current_name), ("direct", direct_name)):
            result_root = Path(tester._result_root(output_name))
            report_dir = (
                output_dir / "per_checkpoint" / tag / variant
            )
            reports[variant] = evaluate_depth(SimpleNamespace(
                artifact_dir=str(result_root / "artifacts"),
                pseudo_labels=str(pseudo_labels),
                ground_truth_dir=str(gt_dir),
                split_file=str(split_file),
                output_dir=str(report_dir),
                prediction_dir=str(result_root / "data"),
                class_name=args.class_name,
                iou_threshold=args.iou_threshold,
                allow_missing_artifacts=args.allow_missing_artifacts,
                skip_official_ap=False,
                final_formula=variant,
                quiet=True,
            ))

        current, direct = reports["current"], reports["direct"]
        row = {
            "epoch": epoch,
            "checkpoint": str(entry["path"]),
            "aliases": ",".join(entry["aliases"]),
            "ap_current": current.get("ap3d_r40_moderate"),
            "ap_direct": direct.get("ap3d_r40_moderate"),
            "delta_ap": None,
            "mae_current": metric(current, "final", "mae"),
            "mae_direct": metric(direct, "final", "mae"),
            "delta_mae": None,
            "mae_gpro": metric(current, "gpro_center", "mae"),
            "mae_baseline": metric(
                current, "baseline_original_formula", "mae"
            ),
            "rmse_current": metric(current, "final", "rmse"),
            "rmse_direct": metric(direct, "final", "rmse"),
            "median_current": metric(
                current, "final", "median_ae"
            ),
            "median_direct": metric(
                direct, "final", "median_ae"
            ),
            "p90_current": metric(current, "final", "p90_ae"),
            "p90_direct": metric(direct, "final", "p90_ae"),
            "p90_gpro": metric(
                current, "gpro_center", "p90_ae"
            ),
            "bias_current": metric(current, "final", "signed_bias"),
            "bias_direct": metric(direct, "final", "signed_bias"),
            "bias_gpro": metric(
                current, "gpro_center", "signed_bias"
            ),
            "gpro_coverage_matched": metric(
                current, "gpro_center", "coverage_of_matched"
            ),
            "detector_recall": current.get(
                "protocol", {}
            ).get("detector_recall"),
        }
        if finite(row["ap_current"]) and finite(row["ap_direct"]):
            row["delta_ap"] = (
                row["ap_direct"] - row["ap_current"]
            )
        if finite(row["mae_current"]) and finite(row["mae_direct"]):
            row["delta_mae"] = (
                row["mae_direct"] - row["mae_current"]
            )
        rows.append(row)
        logger.info(
            (
                "epoch %d | AP current %s direct %s (delta %s) | "
                "MAE current %s direct %s GPro %s"
            ),
            epoch, fmt(row["ap_current"]), fmt(row["ap_direct"]),
            fmt(row["delta_ap"]), fmt(row["mae_current"]),
            fmt(row["mae_direct"]), fmt(row["mae_gpro"]),
        )
        # Refresh partial reports after every checkpoint so a long sweep is
        # still useful if it is interrupted.
        write_reports(rows, output_dir, metadata)

    print("Sweep complete: {}".format(output_dir))
    print(
        "Graph: {}".format(
            output_dir / "near_gpro_direct_sweep.svg"
        )
    )
    print(
        "Report: {}".format(
            output_dir / "near_gpro_direct_sweep.md"
        )
    )


if __name__ == "__main__":
    main()
