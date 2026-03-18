#!/usr/bin/env python3
"""
ProkBERT Finetuning with Length-Based Groups
=============================================

Uses the same pipeline as finetuning.py (BertForBinaryClassificationWithPooling +
ProkBERT internal tokenizer + ProkBERTTrainingDatasetPT) but reads a local
FASTA file + labels.csv instead of the HuggingFace dataset.

Preprocessing:
  1. Load genomes from FASTA
  2. Cut each genome into contigs using a sliding window
  3. Assign each contig to a length group:
       Group A: 100-400 bp
       Group B: 400-800 bp
       Group C: 800-1200 bp
       Group D: 1200-1800 bp
  4. Train/eval split per genome (no data leakage)
  5. Tokenize with ProkBERT internal tokenizer
  6. Finetune with HuggingFace Trainer + BertForBinaryClassificationWithPooling

Usage:
    python finetuning_groups.py \\
        --fasta /path/to/genomes.fasta \\
        --labels_csv /path/to/labels.csv \\
        --output_dir ./finetune_results \\
        --model_name neuralbioinfo/prokbert-mini \\
        --num_epochs 5 \\
        --batch_size 128 \\
        --bf16
"""

import argparse
import logging
import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer

from prokbert import helper
from prokbert.training_utils import (
    check_nvidia_gpu,
    get_default_pretrained_model_parameters,
    get_torch_data_from_segmentdb_classification,
    compute_metrics_eval_prediction,
)
from prokbert.models import BertForBinaryClassificationWithPooling
from prokbert.prok_datasets import ProkBERTTrainingDatasetPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Length group definitions
# ---------------------------------------------------------------------------

GROUPS = {
    "A": (100, 400),
    "B": (400, 800),
    "C": (800, 1200),
    "D": (1200, 1800),
}


# ---------------------------------------------------------------------------
# Data loading & sliding window segmentation
# ---------------------------------------------------------------------------

def load_labels_csv(path: str) -> Dict[str, int]:
    """Load labels CSV -> dict mapping genome_id -> label (int)."""
    df = pd.read_csv(path)
    if "genome_id" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Labels CSV must have 'genome_id' and 'label' columns. Found: {list(df.columns)}"
        )
    label_map = dict(zip(df["genome_id"].astype(str), df["label"].astype(int)))
    logger.info("Loaded %d labels from %s", len(label_map), path)
    return label_map


def sliding_window_segments(
    sequence: str,
    window_size: int,
    step_size: int,
    min_length: int = 0,
) -> List[str]:
    """Cut a sequence into overlapping segments using a sliding window."""
    segments = []
    seq_len = len(sequence)
    if seq_len < min_length:
        return segments

    for start in range(0, seq_len - min_length + 1, step_size):
        end = min(start + window_size, seq_len)
        seg = sequence[start:end]
        if len(seg) >= min_length:
            segments.append(seg)
        if end == seq_len:
            break

    return segments


def build_segment_dataframe(
    fasta_path: str,
    label_map: Dict[str, int],
    groups: Dict[str, Tuple[int, int]],
    step_fraction: float = 0.5,
) -> pd.DataFrame:
    """Load FASTA, cut genomes into contigs via sliding window, assign to groups.

    For each length group (min_bp, max_bp):
      - window_size = max_bp
      - step_size = int(max_bp * step_fraction)
      - Only keep segments with length in [min_bp, max_bp]

    Returns DataFrame with columns: segment_id, segment, sequence_id, y, label, group
    (compatible with get_torch_data_from_segmentdb_classification)
    """
    records = list(SeqIO.parse(fasta_path, "fasta"))
    logger.info("Loaded %d records from %s", len(records), fasta_path)

    all_rows = []
    missing = []
    segment_id = 0

    for record in records:
        genome_id = record.id
        if genome_id not in label_map:
            missing.append(genome_id)
            continue
        label_int = label_map[genome_id]
        seq = str(record.seq).upper()

        for group_name, (min_bp, max_bp) in groups.items():
            window_size = max_bp
            step_size = max(1, int(max_bp * step_fraction))

            segments = sliding_window_segments(
                seq,
                window_size=window_size,
                step_size=step_size,
                min_length=min_bp,
            )
            for seg in segments:
                all_rows.append({
                    "segment_id": segment_id,
                    "segment": seg,
                    "sequence_id": genome_id,
                    "y": label_int,
                    "label": f"class_{label_int}",
                    "group": group_name,
                })
                segment_id += 1

    if missing:
        logger.warning(
            "%d genomes not found in labels CSV (first 5: %s)",
            len(missing), missing[:5],
        )

    df = pd.DataFrame(all_rows)
    logger.info("Total segments: %d", len(df))
    for g in sorted(groups.keys()):
        n = len(df[df["group"] == g])
        logger.info("  Group %s: %d segments", g, n)

    return df


# ---------------------------------------------------------------------------
# Train/val split by genome (no data leakage)
# ---------------------------------------------------------------------------

def split_by_genome(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ensuring all segments from the same genome stay in the same split."""
    genome_ids = df["sequence_id"].unique()
    labels_per_genome = df.groupby("sequence_id")["y"].first()

    train_ids, val_ids = train_test_split(
        genome_ids,
        test_size=test_size,
        random_state=seed,
        stratify=labels_per_genome.loc[genome_ids],
    )
    train_df = df[df["sequence_id"].isin(set(train_ids))].reset_index(drop=True)
    val_df = df[df["sequence_id"].isin(set(val_ids))].reset_index(drop=True)
    logger.info(
        "Split: %d train segments (%d genomes), %d val segments (%d genomes)",
        len(train_df), len(train_ids), len(val_df), len(val_ids),
    )
    return train_df, val_df


# ---------------------------------------------------------------------------
# Training (same as finetuning.py)
# ---------------------------------------------------------------------------

def train_group(
    group_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str,
    output_dir: str,
    args: argparse.Namespace,
) -> Dict:
    """Train one group using the same pipeline as finetuning.py."""
    logger.info("=" * 60)
    logger.info("GROUP %s — train: %d segments, val: %d segments",
                group_name, len(train_df), len(val_df))
    logger.info("=" * 60)

    group_output = os.path.join(output_dir, f"group_{group_name}")
    os.makedirs(group_output, exist_ok=True)

    # ---- Model & tokenizer (same as finetuning.py) ----
    pretrained_model, tokenizer = get_default_pretrained_model_parameters(
        model_name=model_name,
        model_class='MegatronBertModel',
        output_hidden_states=False,
        output_attentions=False,
        move_to_gpu=False,
    )
    model = BertForBinaryClassificationWithPooling(pretrained_model)

    # ---- Tokenize with ProkBERT tokenizer (same as finetuning.py) ----
    print(f'Processing train data for group {group_name}!')
    [X_train, y_train, torchdb_train] = get_torch_data_from_segmentdb_classification(
        tokenizer, train_df
    )
    print(f'Processing val data for group {group_name}!')
    [X_val, y_val, torchdb_val] = get_torch_data_from_segmentdb_classification(
        tokenizer, val_df
    )

    # ---- Create datasets (same as finetuning.py) ----
    train_ds = ProkBERTTrainingDatasetPT(X_train, y_train, AddAttentionMask=True)
    val_ds = ProkBERTTrainingDatasetPT(X_val, y_val, AddAttentionMask=True)

    # ---- Training args ----
    training_args = TrainingArguments(
        output_dir=group_output,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        greater_is_better=True,
        logging_strategy="epoch",
        report_to="none",
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    # ---- Trainer (same as finetuning.py) ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_eval_prediction,
    )

    trainer.train()

    # Evaluate
    metrics = trainer.evaluate()
    logger.info("Group %s metrics: %s", group_name, metrics)

    # Save model (same as finetuning.py)
    best_path = os.path.join(group_output, "best_model")
    model.save_pretrained(best_path)
    logger.info("Saved model to %s", best_path)

    return {"group": group_name, **metrics}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ProkBERT Finetuning with Length-Based Groups"
    )
    parser.add_argument("--fasta", type=str, required=True, help="Path to input FASTA file")
    parser.add_argument("--labels_csv", type=str, required=True, help="Path to labels CSV")
    parser.add_argument(
        "--model_name", type=str, default="neuralbioinfo/prokbert-mini",
        help="Pretrained model name or path",
    )
    parser.add_argument("--output_dir", type=str, default="./finetune_results")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction for validation")
    parser.add_argument("--step_fraction", type=float, default=0.5,
                        help="Sliding window step as fraction of window size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 (A100/H100)")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument(
        "--groups", type=str, nargs="+", default=None,
        help="Specific groups to train (e.g. A B). Default: all (A B C D)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    helper.set_seed(args.seed)
    check_nvidia_gpu()

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Select groups
    selected_groups = {k: v for k, v in GROUPS.items()}
    if args.groups:
        selected_groups = {k: GROUPS[k] for k in args.groups if k in GROUPS}
    logger.info("Groups to train: %s", list(selected_groups.keys()))

    # Load data
    label_map = load_labels_csv(args.labels_csv)

    logger.info("Building segments with sliding window (step_fraction=%.2f) ...", args.step_fraction)
    seg_df = build_segment_dataframe(
        args.fasta, label_map, selected_groups, step_fraction=args.step_fraction,
    )

    if len(seg_df) == 0:
        logger.error("No segments created. Check your FASTA and labels.")
        return

    # Split by genome
    train_df, val_df = split_by_genome(seg_df, test_size=args.test_size, seed=args.seed)

    # Train each group
    all_metrics = []
    for group_name in sorted(selected_groups.keys()):
        g_train = train_df[train_df["group"] == group_name].reset_index(drop=True)
        g_val = val_df[val_df["group"] == group_name].reset_index(drop=True)

        if len(g_train) == 0:
            logger.warning("Group %s has no training data, skipping.", group_name)
            continue
        if len(g_val) == 0:
            logger.warning("Group %s has no validation data, skipping.", group_name)
            continue

        # Re-assign segment_id for this group subset
        g_train = g_train.copy()
        g_train["segment_id"] = range(len(g_train))
        g_val = g_val.copy()
        g_val["segment_id"] = range(len(g_val))

        metrics = train_group(
            group_name=group_name,
            train_df=g_train,
            val_df=g_val,
            model_name=args.model_name,
            output_dir=args.output_dir,
            args=args,
        )
        all_metrics.append(metrics)

    # Summary
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        summary_path = os.path.join(args.output_dir, "group_metrics.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info("Summary saved to %s", summary_path)
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
