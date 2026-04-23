#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


LABEL_COLS = [
    "fracture_visible",
    "deplacement_ou_incongruence",
    "consolidation_ou_reaction_periostee",
    "materiel_implant",
    "osteotomie_ou_arthrodese",
]

KEEP_METADATA_COLS = [
    "timestamp", "annotator", "subset_mode", "seed", "split", "report_id",
    "comment", "skipped"
]


def _normalize_yes_no_unknown(x):
    """Normalize values to {'oui','non','inconnu'}; keep NA if truly missing."""
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().lower()
    # tolerate variants just in case
    mapping = {
        "oui": "oui",
        "non": "non",
        "inconnu": "inconnu",
        "unknown": "inconnu",
        "na": pd.NA,
        "nan": pd.NA,
        "": pd.NA,
    }
    return mapping.get(s, s)


def load_and_clean_manual(manual_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(manual_csv)

    # basic schema checks
    missing = set(["report_id", "split", "timestamp"] + LABEL_COLS + ["skipped"]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in manual CSV: {sorted(missing)}")

    # keep only useful cols (if you want to keep all, remove this block)
    cols_present = [c for c in KEEP_METADATA_COLS if c in df.columns] + [c for c in LABEL_COLS if c in df.columns]
    df = df[cols_present].copy()

    # normalize
    for c in LABEL_COLS:
        df[c] = df[c].map(_normalize_yes_no_unknown)

    # ensure skipped is int-ish
    df["skipped"] = df["skipped"].fillna(0).astype(int)

    # parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # drop rows with missing critical ids
    df = df.dropna(subset=["report_id", "split"])

    # standardize split strings
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df = df[df["split"].isin(["val", "test"])].copy()

    # remove skipped
    df = df[df["skipped"] == 0].copy()

    return df


def deduplicate_take_latest(df: pd.DataFrame) -> pd.DataFrame:
    """
    If a report_id appears multiple times (re-annotation), keep the most recent timestamp.
    Ties are broken by keeping the last row after sorting (stable).
    """
    df = df.sort_values(["report_id", "timestamp"], ascending=[True, True])
    df_latest = df.groupby("report_id", as_index=False).tail(1).copy()
    return df_latest


def check_consistency(df_latest: pd.DataFrame) -> pd.DataFrame:
    """
    Optional: detect report_id appearing in both val and test after deduplication.
    Ideally should not happen; if it happens, keep the latest timestamp overall
    (already done) but we can still warn.
    """
    dup_split = (
        df_latest.groupby("report_id")["split"]
        .nunique()
        .reset_index(name="n_splits")
        .query("n_splits > 1")
    )
    if len(dup_split) > 0:
        # Keep going but warn in console
        print(f"[WARN] {len(dup_split)} report_id appear in >1 split after cleaning. "
              f"This should not happen. Example:\n{dup_split.head(5)}")
    return df_latest


def export_gold(df_latest: pd.DataFrame, out_dir: str | Path, keep_metadata: bool = False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # "Gold clean" columns
    base_cols = ["report_id"] + LABEL_COLS
    if keep_metadata:
        meta_cols = [c for c in ["split", "timestamp", "annotator"] if c in df_latest.columns]
        cols = meta_cols + base_cols
    else:
        cols = base_cols

    df_all = df_latest.copy()

    # Exports
    df_val = df_all[df_all["split"] == "val"][cols].sort_values("report_id")
    df_test = df_all[df_all["split"] == "test"][cols].sort_values("report_id")

    df_all_export = df_all[["split"] + cols].sort_values(["split", "report_id"])

    df_val.to_csv(out_dir / "gold_val.csv", index=False)
    df_test.to_csv(out_dir / "gold_test.csv", index=False)
    df_all_export.to_csv(out_dir / "gold_all.csv", index=False)

    print(f"[OK] Exported: {out_dir/'gold_val.csv'} ({len(df_val)})")
    print(f"[OK] Exported: {out_dir/'gold_test.csv'} ({len(df_test)})")
    print(f"[OK] Exported: {out_dir/'gold_all.csv'} ({len(df_all_export)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", type=str, required=True,
                        help="Path to annotations/manual_val_test.csv")
    parser.add_argument("--out_dir", type=str, default="annotations/gold_clean",
                        help="Output directory")
    parser.add_argument("--keep_metadata", action="store_true",
                        help="If set, keep split/timestamp/annotator in outputs.")
    args = parser.parse_args()

    df = load_and_clean_manual(args.manual)
    df_latest = deduplicate_take_latest(df)
    df_latest = check_consistency(df_latest)

    export_gold(df_latest, args.out_dir, keep_metadata=args.keep_metadata)


if __name__ == "__main__":
    main()
