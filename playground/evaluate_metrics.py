"""Evaluate FakeShield test outputs.

`scripts/test.sh` only *produces* outputs; it does not score them. This script
fills that gap. It has two independent parts:

  1. DETECTION  (--dte-fdm-output + --labels)
     Parses each DTE-FDM answer for "has not been tampered with" -> authentic,
     else tampered, and compares to the ground-truth labels written by
     `eval_jsonl.py --labels-output`. Reports accuracy / precision / recall / F1
     and a confusion matrix.

  2. LOCALIZATION  (--pred-dir + --gt-dir)
     Compares each predicted mask in --pred-dir against the matching GT mask in
     --gt-dir (matched by basename stem, so a `.jpg` prediction matches a `.png`
     GT). Reports per-image IoU / F1 / precision / recall and the two aggregate
     IoU numbers used by the repo's training validation:
        * gIoU  = mean of per-image IoU  (repo: trackers["gIoU"].avg[1])
        * cIoU  = sum(intersect) / sum(union)  (repo: iou_per_class[1])
     Optional --pixel-auc sweeps thresholds for a pixel-level ROC AUC (note:
     test.py saves already-binarized 0/255 masks, so this is coarse).

Examples
--------
# Detection only
python playground/evaluate_metrics.py \
    --dte-fdm-output ./playground/DTE-FDM_output.jsonl \
    --labels ./playground/test_labels.jsonl

# Localization only
python playground/evaluate_metrics.py \
    --pred-dir ./playground/MFLM_output \
    --gt-dir   /scratch/dataset/photoshop/CASIAv1+_Tp/mask

# Both
python playground/evaluate_metrics.py \
    --dte-fdm-output ./playground/DTE-FDM_output.jsonl \
    --labels ./playground/test_labels.jsonl \
    --pred-dir ./playground/MFLM_output \
    --gt-dir   /scratch/dataset/photoshop/CASIAv1+_Tp/mask \
    --per-image
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def predict_tampered(text):
    """DTE-FDM says 'has not been tampered with' for authentic images."""
    return 0 if "has not been tampered with" in text else 1


def evaluate_detection(dte_jsonl, labels_jsonl):
    labels = {os.path.basename(r["image"]): int(r["tampered"]) for r in labels_jsonl}
    tp = fp = tn = fn = 0
    skipped = 0
    for rec in dte_jsonl:
        stem = Path(rec["image"]).stem
        # match by stem (labels keyed by basename incl. extension first)
        gt = None
        base = os.path.basename(rec["image"])
        if base in labels:
            gt = labels[base]
        elif stem in labels:
            gt = labels[stem]
        else:
            # try matching against label stems
            stem_labels = {Path(k).stem: v for k, v in labels.items()}
            gt = stem_labels.get(stem)
        if gt is None:
            skipped += 1
            continue
        pred = predict_tampered(rec.get("outputs", ""))
        if gt == 1 and pred == 1:
            tp += 1
        elif gt == 0 and pred == 1:
            fp += 1
        elif gt == 0 and pred == 0:
            tn += 1
        elif gt == 1 and pred == 0:
            fn += 1

    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    print("\n===== DETECTION =====")
    print(f"evaluated: {n}   (skipped {skipped} without GT labels)")
    print(f"accuracy : {acc:.4f}")
    print(f"precision: {prec:.4f}")
    print(f"recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    print("confusion matrix (rows=GT, cols=Pred):")
    print("              pred_auth  pred_tamp")
    print(f"  gt_auth         {tn:>7}     {fp:>7}")
    print(f"  gt_tamp         {fn:>7}     {tp:>7}")


# --------------------------------------------------------------------------- #
# Localization
# --------------------------------------------------------------------------- #
def _load_gray(path):
    try:
        g = np.array(Image.open(str(path)).convert("L"))
    except Exception:
        return None
    return g


def _metrics(pred, gt):
    p = pred > 127
    g = gt > 127
    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    ps = int(p.sum())
    gs = int(g.sum())
    iou = inter / union if union else 1.0  # repo convention: empty|empty -> 1
    dice = 2 * inter / (ps + gs) if (ps + gs) else 1.0
    prec = inter / ps if ps else (1.0 if gs == 0 else 0.0)
    rec = inter / gs if gs else 1.0
    return inter, union, iou, dice, prec, rec


def pixel_auc(preds, gts):
    """Coarse pixel-level ROC AUC from binarized (0/255) masks."""
    scores = np.concatenate([p.ravel() / 255.0 for p in preds])
    labels = np.concatenate([g.ravel() > 127 for g in gts])
    if len(np.unique(labels)) < 2:
        return float("nan")
    thr = np.unique(scores)
    tpr, fpr = [], []
    for t in thr:
        pred_pos = scores > t
        tp = np.logical_and(pred_pos, labels).sum()
        fp = np.logical_and(pred_pos, ~labels).sum()
        p = labels.sum()
        n = len(labels) - p
        tpr.append(tp / p if p else 0.0)
        fpr.append(fp / n if n else 0.0)
    # add the two ROC corners (all-negative / all-positive) and keep the upper
    # envelope (max TPR at each FPR) so the trapezoid is correct.
    fx = [0.0] + fpr + [1.0]
    tx = [0.0] + tpr + [1.0]
    envelope = {}
    for f_, t_ in zip(fx, tx):
        envelope[f_] = max(envelope.get(f_, -1.0), t_)
    xs = sorted(envelope.keys())
    ys = [envelope[x] for x in xs]
    return float(np.trapz(ys, xs))


def evaluate_localization(pred_dir, gt_dirs, per_image, do_auc):
    # Build GT index by stem -> path.
    gt_index = {}
    for d in gt_dirs:
        for p in Path(d).iterdir():
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                if p.stem in gt_index:
                    print(f"[warn] duplicate GT stem {p.stem!r}; keeping first")
                else:
                    gt_index[p.stem] = p

    rows = []
    sum_inter = sum_union = 0
    ious, dices, precs, recs = [], [], [], []
    pred_no_gt = 0

    for pred_path in sorted(Path(pred_dir).iterdir()):
        if not pred_path.is_file():
            continue
        stem = pred_path.stem
        gt_path = gt_index.get(stem)
        if gt_path is None:
            pred_no_gt += 1
            continue
        pred = _load_gray(pred_path)
        gt = _load_gray(gt_path)
        if pred is None or gt is None:
            continue
        if pred.shape != gt.shape:
            gt = np.array(Image.fromarray(gt).resize(
                (pred.shape[1], pred.shape[0]), Image.NEAREST))
        inter, union, iou, dice, prec, rec = _metrics(pred, gt)
        sum_inter += inter
        sum_union += union
        ious.append(iou)
        dices.append(dice)
        precs.append(prec)
        recs.append(rec)
        rows.append((stem, iou, dice, prec, rec))

    if not rows:
        print("\n===== LOCALIZATION =====\nNo matched pred/GT mask pairs found.")
        print(f"(predicted masks without GT: {pred_no_gt})")
        return

    giou = float(np.mean(ious))
    ciou = sum_inter / sum_union if sum_union else 0.0

    print("\n===== LOCALIZATION =====")
    print(f"matched images: {len(rows)}   (predicted masks without GT: {pred_no_gt})")
    print(f"gIoU (mean per-image IoU)        : {giou:.4f}")
    print(f"cIoU (sum inter / sum union)     : {ciou:.4f}")
    print(f"mean F1 (Dice)                   : {np.mean(dices):.4f}")
    print(f"mean precision                   : {np.mean(precs):.4f}")
    print(f"mean recall                      : {np.mean(recs):.4f}")
    if do_auc:
        preds, gts = [], []
        for pred_path in sorted(Path(pred_dir).iterdir()):
            gt_path = gt_index.get(pred_path.stem)
            if gt_path is None:
                continue
            p = _load_gray(pred_path)
            g = _load_gray(gt_path)
            if p is None or g is None:
                continue
            if p.shape != g.shape:
                g = np.array(Image.fromarray(g).resize(
                    (p.shape[1], p.shape[0]), Image.NEAREST))
            preds.append(p)
            gts.append(g)
        print(f"pixel AUC (coarse, binarized)    : {pixel_auc(preds, gts):.4f}")

    if per_image:
        print("\nper-image:")
        print(f"  {'image':<40} {'IoU':>7} {'F1':>7} {'Prec':>7} {'Rec':>7}")
        for stem, iou, dice, prec, rec in rows:
            print(f"  {stem:<40} {iou:>7.4f} {dice:>7.4f} {prec:>7.4f} {rec:>7.4f}")


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Score FakeShield test outputs.")
    parser.add_argument("--dte-fdm-output", help="DTE-FDM JSONL (detection).")
    parser.add_argument("--labels", help="Labels JSONL from eval_jsonl.py (detection).")
    parser.add_argument("--pred-dir", help="MFLM predicted mask directory (localization).")
    parser.add_argument("--gt-dir", nargs="+", help="GT mask directory/ies (localization).")
    parser.add_argument("--per-image", action="store_true", help="Print per-image localization metrics.")
    parser.add_argument("--pixel-auc", action="store_true", help="Also compute (coarse) pixel-level AUC.")
    args = parser.parse_args()

    if args.dte_fdm_output and args.labels:
        evaluate_detection(load_jsonl(args.dte_fdm_output), load_jsonl(args.labels))
    elif args.dte_fdm_output or args.labels:
        print("[warn] detection needs BOTH --dte-fdm-output and --labels; skipping detection.")

    if args.pred_dir and args.gt_dir:
        evaluate_localization(args.pred_dir, args.gt_dir, args.per_image, args.pixel_auc)
    elif args.pred_dir or args.gt_dir:
        print("[warn] localization needs BOTH --pred-dir and --gt-dir; skipping localization.")


if __name__ == "__main__":
    main()
