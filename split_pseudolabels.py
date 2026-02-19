from __future__ import annotations

from pathlib import Path
import csv


# Dossiers
SPLITS_DIR = Path("./splits")               # contient train_exams.csv / val_exams.csv / test_exams.csv
PSEUDO_DIR = Path("./pseudolabels")         # contient tes output(s).csv (ministral / qwen)
OUT_DIR = Path("./pseudo_splits")           # sortie
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fichiers pseudolabels à splitter (tu peux en mettre plusieurs)
PSEUDO_FILES = [
    PSEUDO_DIR / "output_ministral3_3B.csv",
    PSEUDO_DIR / "output_qwen3_4B.csv",
    # PSEUDO_DIR / "output.csv",
]


def normalize_report_id(x: str) -> str:
    """Normalise un id (enlève extension .txt éventuelle, strip)."""
    x = (x or "").strip()
    if x.lower().endswith(".txt"):
        return Path(x).stem
    return x


def load_split_report_ids(split_csv: Path) -> set[str]:
    """Charge les report_id depuis un CSV de split (train/val/test)."""
    report_ids: set[str] = set()
    with split_csv.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError(f"{split_csv} est vide ou sans header.")

        if "report_id" not in r.fieldnames:
            raise ValueError(f"Colonne 'report_id' introuvable dans {split_csv}. Colonnes: {r.fieldnames}")

        for row in r:
            rid = normalize_report_id(row.get("report_id", ""))
            if rid:
                report_ids.add(rid)

    return report_ids


def split_pseudolabel_file(pseudo_csv: Path, split_name: str, keep_ids: set[str], out_csv: Path) -> None:
    with pseudo_csv.open("r", newline="", encoding="utf-8") as fin:
        r = csv.DictReader(fin)
        if r.fieldnames is None:
            raise ValueError(f"{pseudo_csv} est vide ou sans header.")

        if "report_id" not in r.fieldnames:
            raise ValueError(f"Colonne 'report_id' introuvable dans {pseudo_csv}. Colonnes: {r.fieldnames}")

        with out_csv.open("w", newline="", encoding="utf-8") as fout:
            w = csv.DictWriter(fout, fieldnames=r.fieldnames)
            w.writeheader()

            total = 0
            kept = 0
            missing_in_split = 0

            for row in r:
                total += 1
                rid = normalize_report_id(row.get("report_id", ""))

                if rid in keep_ids:
                    w.writerow(row)
                    kept += 1
                else:
                    missing_in_split += 1

    print(f"[{pseudo_csv.name}] -> {split_name}: wrote {kept} rows (from {total}) to {out_csv.name}")


def main():
    train_ids = load_split_report_ids(SPLITS_DIR / "train_exams.csv")
    val_ids   = load_split_report_ids(SPLITS_DIR / "val_exams.csv")
    test_ids  = load_split_report_ids(SPLITS_DIR / "test_exams.csv")

    # Sanity: pas de mélange
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    for pseudo_csv in PSEUDO_FILES:
        if not pseudo_csv.exists():
            print(f"Skip (not found): {pseudo_csv}")
            continue

        stem = pseudo_csv.stem
        split_pseudolabel_file(pseudo_csv, "train", train_ids, OUT_DIR / f"{stem}_train.csv")
        split_pseudolabel_file(pseudo_csv, "val",   val_ids,   OUT_DIR / f"{stem}_val.csv")
        split_pseudolabel_file(pseudo_csv, "test",  test_ids,  OUT_DIR / f"{stem}_test.csv")


if __name__ == "__main__":
    main()
