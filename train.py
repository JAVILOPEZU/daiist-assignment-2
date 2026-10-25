"""
Assignment 2: training pipeline. Run with `uv run python main.py train`.

Decision supported: can a forager eat this mushroom? Positive class = Poisonous.

Start to finish, with no manual steps:

1. Download jamurdetection/jamur-detection (version 2) from Roboflow Universe,
   using the public dataset directly (no fork).
2. Describe it: instances per class per split, and object size relative to
   the image.
3. Finetune yolo11n from the COCO-pretrained checkpoint and save the best
   weights to weights/best.pt.
4. Evaluate on the test split: AP50, mAP50-95, per-class precision/recall/AP
   and the confusion matrix. Compare against COCO yolo11n zero-shot and two
   trivial baselines.
5. Give every ground-truth mushroom one poison score, sweep the threshold
   on the VALIDATION split with the euro figures in costs.json, pick the
   cheapest single threshold and review band there, then report what those
   choices cost on the TEST split. Choosing on test would make the test cost
   optimistic.
6. Write everything app.py needs to artifacts/, so the app never trains.
"""

import json
import os
import shutil
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

import costing
from api_key import load_api_key

ROOT = Path(__file__).resolve().parent

WORKSPACE = "jamurdetection"
PROJECT = "jamur-detection"
VERSION = 2

BASE_MODEL = "yolo11n.pt"
EPOCHS = int(os.environ.get("EPOCHS", "50"))
PATIENCE = int(os.environ.get("PATIENCE", "15"))
IMGSZ = 640
BATCH = int(os.environ.get("BATCH", "16"))
SEED = 0

DATA_DIR = ROOT / "datasets" / f"{PROJECT}-v{VERSION}"
RUNS_DIR = ROOT / "runs"
WEIGHTS_DIR = ROOT / "weights"
BEST_WEIGHTS = WEIGHTS_DIR / "best.pt"
ARTIFACTS = ROOT / "artifacts"

SPLITS = {"train": "train", "val": "valid", "test": "test"}
MATCH_IOU = 0.5        # a prediction "covers" a mushroom at IoU >= 0.5
SCORE_CONF = 0.001     # keep almost every box so the sweep sees the full range
ZERO_SHOT_CONF = 0.25  # Ultralytics' default confidence for the COCO baseline
THRESHOLDS = np.round(np.linspace(0, 1, 101), 2)
BAND_GRID = np.round(np.linspace(0, 1, 51), 2)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def download_dataset() -> Path:
    """Fetch the dataset once and point data.yaml at absolute split paths."""
    data_yaml = DATA_DIR / "data.yaml"
    if not data_yaml.exists():
        rf = Roboflow(api_key=load_api_key())
        version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
        version.download("yolov11", location=str(DATA_DIR), overwrite=True)

    # Roboflow's exported paths are relative to a folder layout we don't use.
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    config.update(path=str(DATA_DIR), train="train/images",
                  val="valid/images", test="test/images")
    data_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return data_yaml


def class_names(data_yaml: Path) -> list[str]:
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def find_class(names: list[str], keyword: str) -> int:
    matches = [i for i, n in enumerate(names) if keyword in n.lower()]
    if len(matches) != 1:
        raise ValueError(f"Expected one class containing '{keyword}', got {names}")
    return matches[0]


def read_labels(label_file: Path) -> list[tuple[int, np.ndarray]]:
    """YOLO label file -> [(class, normalised xyxy)]. Polygons become their box."""
    boxes = []
    if not label_file.exists():
        return boxes
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, coords = int(parts[0]), np.array(parts[1:], dtype=float)
        if len(coords) == 4:
            x, y, w, h = coords
            box = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        else:
            xs, ys = coords[0::2], coords[1::2]
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
        boxes.append((cls, box))
    return boxes


def split_images(split: str) -> list[Path]:
    folder = DATA_DIR / SPLITS[split] / "images"
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def label_path(image: Path) -> Path:
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def describe_dataset(names: list[str]) -> dict:
    """Instances per class per split and object size relative to the image."""
    stats = {}
    for split in SPLITS:
        images = split_images(split)
        counts, areas = Counter(), []
        empty = 0
        for image in images:
            labels = read_labels(label_path(image))
            empty += not labels
            for cls, box in labels:
                counts[names[cls]] += 1
                areas.append((box[2] - box[0]) * (box[3] - box[1]))
        areas = np.array(areas) if areas else np.zeros(1)
        stats[split] = {
            "images": len(images),
            "images_without_labels": empty,
            "instances": {n: counts.get(n, 0) for n in names},
            "box_area_share": {
                "median": float(np.median(areas)),
                "p10": float(np.percentile(areas, 10)),
                "p90": float(np.percentile(areas, 90)),
                "share_small_under_1pct": float((areas < 0.01).mean()),
            },
        }
    return stats


# --------------------------------------------------------------------------- #
# Training and standard evaluation
# --------------------------------------------------------------------------- #

def finetune(data_yaml: Path) -> Path:
    model = YOLO(BASE_MODEL)
    model.train(data=str(data_yaml), epochs=EPOCHS, patience=PATIENCE, imgsz=IMGSZ,
                batch=BATCH, seed=SEED, deterministic=True,
                project=str(RUNS_DIR), name="finetune", exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copy2(model.trainer.best, BEST_WEIGHTS)
    return BEST_WEIGHTS


def evaluate_on_test(weights: Path, data_yaml: Path, names: list[str]) -> dict:
    """Ultralytics' own test-split metrics: overall, per class, confusion matrix."""
    result = YOLO(str(weights)).val(data=str(data_yaml), split="test", imgsz=IMGSZ,
                                    project=str(RUNS_DIR), name="test", exist_ok=True)
    box = result.box
    per_class = {}
    for i, cls in enumerate(box.ap_class_index):
        per_class[names[int(cls)]] = {
            "precision": float(box.p[i]), "recall": float(box.r[i]),
            "ap50": float(box.ap50[i]), "ap50_95": float(box.ap[i]),
        }
    for plot in ("confusion_matrix.png", "confusion_matrix_normalized.png",
                 "BoxPR_curve.png"):
        source = Path(result.save_dir) / plot
        if source.exists():
            shutil.copy2(source, ARTIFACTS / plot)
    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": per_class,
        # Rows = predicted, columns = true; last row/column is background.
        "confusion_matrix": result.confusion_matrix.matrix.astype(int).tolist(),
        "confusion_labels": names + ["background"],
    }


# --------------------------------------------------------------------------- #
# One poison score per ground-truth mushroom
# --------------------------------------------------------------------------- #

def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against many. Normalised coords give the same IoU."""
    if len(boxes) == 0:
        return np.zeros(0)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-9)


def poison_score(p_poison: float, p_edible: float) -> float:
    """High = treat as poisonous.

    A confident "poisonous" box pushes the score up; only a confident
    "edible" box can pull it down. A mushroom the model does not find at all
    scores 1.0, so "I don't know" is never read as "safe to eat".
    """
    return max(p_poison, 1.0 - p_edible)


def score_objects(model: YOLO, split: str, names: list[str],
                  poison_idx: int, edible_idx: int) -> pd.DataFrame:
    rows = []
    for image in split_images(split):
        labels = read_labels(label_path(image))
        if not labels:
            continue
        pred = model.predict(str(image), conf=SCORE_CONF, imgsz=IMGSZ,
                             verbose=False)[0].boxes
        pred_xyxy = pred.xyxyn.cpu().numpy()
        pred_cls = pred.cls.cpu().numpy().astype(int)
        pred_conf = pred.conf.cpu().numpy()
        for cls, box in labels:
            covers = iou(box, pred_xyxy) >= MATCH_IOU
            p_poison = float(pred_conf[covers & (pred_cls == poison_idx)].max(initial=0))
            p_edible = float(pred_conf[covers & (pred_cls == edible_idx)].max(initial=0))
            rows.append({
                "image": image.name, "true_class": names[cls],
                "poisonous": cls == poison_idx,
                "p_poison": p_poison, "p_edible": p_edible,
                "score": poison_score(p_poison, p_edible),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #

def coco_zero_shot(split: str) -> dict:
    """Pretrained COCO yolo11n, no finetuning.

    COCO has no mushroom class, let alone edible vs poisonous, so its AP on
    our two classes is zero by construction and it cannot produce a poison
    score: under poison_score every mushroom scores 1.0, exactly like the
    predict-nothing baseline. What it can tell us is whether it even finds
    the mushrooms (class-agnostic recall) and what it thinks they are.
    """
    model = YOLO(BASE_MODEL)
    found, total, seen_as = 0, 0, Counter()
    for image in split_images(split):
        labels = read_labels(label_path(image))
        if not labels:
            continue
        result = model.predict(str(image), conf=ZERO_SHOT_CONF, imgsz=IMGSZ,
                               verbose=False)[0]
        pred_xyxy = result.boxes.xyxyn.cpu().numpy()
        pred_cls = result.boxes.cls.cpu().numpy().astype(int)
        for _, box in labels:
            total += 1
            overlaps = iou(box, pred_xyxy)
            if len(overlaps) and overlaps.max() >= MATCH_IOU:
                found += 1
                seen_as[result.names[int(pred_cls[overlaps.argmax()])]] += 1
    return {
        "map50": 0.0, "map50_95": 0.0,
        "class_agnostic_recall": found / total if total else 0.0,
        "most_common_coco_labels": dict(seen_as.most_common(5)),
    }


def majority_class(names: list[str], stats: dict) -> str:
    counts = stats["train"]["instances"]
    return max(names, key=lambda n: counts[n])


# --------------------------------------------------------------------------- #
# Costing
# --------------------------------------------------------------------------- #

def plot_cost_curve(val_sweep: list[dict], test_sweep: list[dict],
                    chosen: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["threshold"] for r in val_sweep],
            [r["cost_per_1000"] for r in val_sweep], label="validation (chosen here)")
    ax.plot([r["threshold"] for r in test_sweep],
            [r["cost_per_1000"] for r in test_sweep], label="test", alpha=0.6)
    ax.axvline(chosen, color="black", linestyle="--", label=f"chosen t = {chosen:.2f}")
    ax.set_xlabel("Poison-score threshold (fail at or above)")
    ax.set_ylabel("Expected cost per 1,000 mushrooms (€)")
    ax.set_title("Cost against threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    data_yaml = download_dataset()
    names = class_names(data_yaml)
    poison_idx = find_class(names, "poison")
    edible_idx = find_class(names, "edible")

    stats = describe_dataset(names)
    print(json.dumps(stats, indent=2))

    weights = finetune(data_yaml)
    finetuned = evaluate_on_test(weights, data_yaml, names)

    model = YOLO(str(weights))
    scores = {split: score_objects(model, split, names, poison_idx, edible_idx)
              for split in ("val", "test")}
    for split, frame in scores.items():
        frame.to_csv(ARTIFACTS / f"scores_{split}.csv", index=False)

    costs = costing.load_costs()
    val, test = scores["val"], scores["test"]
    val_sweep = costing.sweep(val.score, val.poisonous, costs, THRESHOLDS)
    test_sweep = costing.sweep(test.score, test.poisonous, costs, THRESHOLDS)
    chosen = min(val_sweep, key=lambda r: r["cost_per_1000"])["threshold"]
    band = costing.best_band(val.score, val.poisonous, costs, BAND_GRID)
    pd.DataFrame(val_sweep).to_csv(ARTIFACTS / "cost_sweep_val.csv", index=False)
    pd.DataFrame(test_sweep).to_csv(ARTIFACTS / "cost_sweep_test.csv", index=False)
    plot_cost_curve(val_sweep, test_sweep, chosen, ARTIFACTS / "cost_vs_threshold.png")

    # Baselines on the same test mushrooms. Predict-nothing (and COCO, which
    # has no poison class) scores every mushroom 1.0: all thrown away.
    majority = majority_class(names, stats)
    trivial_scores = {
        "predict_nothing": np.ones(len(test)),
        f"always_{majority}": np.full(len(test), 1.0 if names.index(majority) == poison_idx
                                      else 0.0),
    }
    baselines = {
        "coco_zero_shot": coco_zero_shot("test"),
        **{name: {"map50": 0.0, "map50_95": 0.0} for name in trivial_scores},
    }
    baselines["coco_zero_shot"]["cost_at_chosen_t"] = costing.expected_cost(
        np.ones(len(test)), test.poisonous, chosen, chosen, costs)
    for name, s in trivial_scores.items():
        baselines[name]["cost_at_chosen_t"] = costing.expected_cost(
            s, test.poisonous, chosen, chosen, costs)

    operating_point = {
        "positive_class": names[poison_idx],
        "costs": costs,
        "chosen_on": "val",
        "single_threshold": chosen,
        "band": {"t_low": band["t_low"], "t_high": band["t_high"]},
        "test_at_single_threshold": costing.expected_cost(
            test.score, test.poisonous, chosen, chosen, costs),
        "test_at_band": costing.expected_cost(
            test.score, test.poisonous, band["t_low"], band["t_high"], costs),
        "test_policies": costing.policy_costs(
            test.score, test.poisonous, costs, chosen, band["t_low"], band["t_high"]),
    }

    metrics = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION,
                    "classes": names, "stats": stats},
        "training": {"base_model": BASE_MODEL, "epochs": EPOCHS, "patience": PATIENCE,
                     "imgsz": IMGSZ, "batch": BATCH, "seed": SEED},
        "finetuned": finetuned,
        "baselines": baselines,
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ARTIFACTS / "operating_point.json").write_text(json.dumps(operating_point, indent=2))
    print(json.dumps({"finetuned_map50": finetuned["map50"],
                      "operating_point": operating_point}, indent=2))


if __name__ == "__main__":
    main()
