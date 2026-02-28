from pathlib import Path
import csv
import random
from collections import defaultdict
from dataclasses import dataclass


# ----------------------------
# Config
# ----------------------------
DATA_DIR = Path("./fa")         # dossier contenant les rapports (.txt)
PATTERN = "*.txt"              # ou "*.txt"; si sous-dossiers => RECURSIVE = True
RECURSIVE = False              # True => rglob
SEED = 12345

# Cibles "souples" (on peut dépasser car patient-wise sans tronquage)
TARGET_VAL_EXAMS = 250
TARGET_TEST_EXAMS = 250

# Optionnel: plafonds "durs" pour éviter qu’un patient énorme fasse exploser la taille
# Mets None si tu veux désactiver
MAX_VAL_EXAMS = TARGET_VAL_EXAMS + 15
MAX_TEST_EXAMS = TARGET_TEST_EXAMS + 15

OUT_DIR = Path("./splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# underscore séparateur à la position 17 (1-based) => index 16 (0-based)
SEP_IDX = 16


# ----------------------------
# Structures
# ----------------------------
@dataclass(frozen=True)
class Exam:
    patient_id: str
    exam_id: str
    report_id: str   # filename stem
    path: Path


# ----------------------------
# IO / parsing
# ----------------------------
def discover_reports(data_dir: Path, pattern: str = "*.txt", recursive: bool = False) -> list[Path]:
    if recursive:
        return sorted(data_dir.rglob(pattern))
    return sorted(data_dir.glob(pattern))

def parse_ids_from_filename(p: Path, sep_idx: int = SEP_IDX) -> tuple[str, str]:
    stem = p.stem

    if len(stem) <= sep_idx:
        raise ValueError(f"Nom trop court pour utiliser sep_idx={sep_idx}: {p.name}")

    if stem[sep_idx] != "_":
        raise ValueError(
            f"Attendu '_' à la position {sep_idx} (0-based) / {sep_idx+1} (1-based) "
            f"mais trouvé '{stem[sep_idx]}' dans: {p.name}"
        )

    patient_id = stem[:sep_idx]      # tout avant l’underscore fixe
    exam_id = stem[sep_idx + 1:]     # tout après
    if not exam_id:
        raise ValueError(f"Exam_id vide après le séparateur dans: {p.name}")

    return patient_id, exam_id

def build_exam_index(paths: list[Path]) -> list[Exam]:
    exams: list[Exam] = []
    for p in paths:
        patient_id, exam_id = parse_ids_from_filename(p)
        exams.append(Exam(patient_id=patient_id, exam_id=exam_id, report_id=p.stem, path=p))
    return exams

def group_by_patient(exams: list[Exam]) -> dict[str, list[Exam]]:
    d = defaultdict(list)
    for e in exams:
        d[e.patient_id].append(e)
    for pid in d:
        d[pid] = sorted(d[pid], key=lambda x: x.report_id)
    return dict(d)

def write_exam_list_csv(exams: list[Exam], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["patient_id", "exam_id", "report_id"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for e in exams:
            w.writerow({
                "patient_id": e.patient_id,
                "exam_id": e.exam_id,
                "report_id": e.report_id
            })


# ----------------------------
# Patient-wise selection (sans tronquage)
# ----------------------------
def select_patients_patientwise(
    patient_to_exams: dict[str, list[Exam]],
    target_exams: int,
    rng: random.Random,
    max_exams: int | None = None,
) -> tuple[set[str], list[Exam]]:
    """
    Ajoute des patients entiers jusqu'à atteindre >= target_exams.
    - Sans tronquer le dernier patient.
    - Si max_exams est défini: on évite d'ajouter un patient qui ferait dépasser max_exams,
      sauf si on n'a encore rien sélectionné (pour ne pas bloquer).
    """
    patients = list(patient_to_exams.keys())
    rng.shuffle(patients)

    selected_patients: set[str] = set()
    selected_exams: list[Exam] = []

    for pid in patients:
        if len(selected_exams) >= target_exams:
            break

        exs = patient_to_exams[pid]
        would_be = len(selected_exams) + len(exs)

        if max_exams is not None and would_be > max_exams and len(selected_exams) > 0:
            # skip ce patient pour rester raisonnable en taille
            continue

        selected_patients.add(pid)
        selected_exams.extend(exs)

    return selected_patients, selected_exams


def main_split():
    rng = random.Random(SEED)

    paths = discover_reports(DATA_DIR, PATTERN, RECURSIVE)
    if not paths:
        raise RuntimeError(f"Aucun fichier trouvé dans {DATA_DIR} avec pattern={PATTERN} recursive={RECURSIVE}")

    exams = build_exam_index(paths)
    patient_to_exams = group_by_patient(exams)

    # --- TEST
    test_patients, test_exams = select_patients_patientwise(
        patient_to_exams, TARGET_TEST_EXAMS, rng, max_exams=MAX_TEST_EXAMS
    )

    # --- remove test patients
    remaining = {pid: exs for pid, exs in patient_to_exams.items() if pid not in test_patients}

    # --- VAL
    val_patients, val_exams = select_patients_patientwise(
        remaining, TARGET_VAL_EXAMS, rng, max_exams=MAX_VAL_EXAMS
    )

    # --- TRAIN = reste
    train_patients = set(remaining.keys()) - val_patients
    train_exams: list[Exam] = []
    for pid in sorted(train_patients):
        train_exams.extend(remaining[pid])

    # sanity checks
    assert test_patients.isdisjoint(val_patients)
    assert test_patients.isdisjoint(train_patients)
    assert val_patients.isdisjoint(train_patients)

    # write lists
    write_exam_list_csv(train_exams, OUT_DIR / "train_exams.csv")
    write_exam_list_csv(val_exams, OUT_DIR / "val_exams.csv")
    write_exam_list_csv(test_exams, OUT_DIR / "test_exams.csv")

    print("---- Split summary (patient-wise, no truncation) ----")
    print(f"Total exams: {len(exams)} | Patients: {len(patient_to_exams)}")
    print(f"TRAIN: exams={len(train_exams)} patients={len(train_patients)}")
    print(f"VAL:   exams={len(val_exams)} patients={len(val_patients)} target={TARGET_VAL_EXAMS} max={MAX_VAL_EXAMS}")
    print(f"TEST:  exams={len(test_exams)} patients={len(test_patients)} target={TARGET_TEST_EXAMS} max={MAX_TEST_EXAMS}")
    print(f"CSVs written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main_split()
