import os
import re
import csv
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import gradio as gr

# Optionnel mais recommandé pour sécuriser l'écriture
try:
    from filelock import FileLock
except Exception:
    FileLock = None


# -----------------------------
# CONFIG
# -----------------------------
FA_DIR = Path("./fa")
SPLITS_DIR = Path("./splits")
OUT_DIR = Path("./annotations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANN_CSV = OUT_DIR / "manual_val_test.csv"
LOCK_PATH = OUT_DIR / "manual_val_test.lock"

VAL_SPLIT_CSV = SPLITS_DIR / "val_exams.csv"
TEST_SPLIT_CSV = SPLITS_DIR / "test_exams.csv"

LABEL_CHOICES = ["oui", "non", "inconnu"]

LABEL_COLUMNS = [
    "fracture_visible",
    "deplacement_ou_incongruence",
    "consolidation_ou_reaction_periostee",
    "materiel_implant",
    "osteotomie_ou_arthrodese",
]

# Colonnes de sortie (append-only)
OUT_COLUMNS = [
    "timestamp",
    "annotator",
    "subset_mode",
    "seed",
    "split",
    "report_id",
    *LABEL_COLUMNS,
    "comment",
    "skipped",
]


# -----------------------------
# UTILITAIRES DATASET
# -----------------------------
def read_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    # Essai UTF-8 puis fallback
    for enc in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="strict")
        except Exception:
            pass
    return path.read_text(encoding="utf-8", errors="replace")


def list_exam_images(report_id: str) -> List[Path]:
    """
    Retourne les images fa/{report_id}.{k}.jpg triées par k (k entier).
    Ignore les fichiers qui matchent mais où k n'est pas un entier.
    """
    candidates = list(FA_DIR.glob(f"{report_id}.*.jpg"))
    items: List[Tuple[int, Path]] = []

    # Exemple filename: <rid>.<k>.jpg
    pattern = re.compile(re.escape(report_id) + r"\.(\d+)\.jpg$")

    for p in candidates:
        m = pattern.search(p.name)
        if not m:
            continue
        k = int(m.group(1))
        items.append((k, p))

    items.sort(key=lambda x: x[0])
    return [p for _, p in items]


def load_splits() -> Tuple[List[str], List[str]]:
    if not VAL_SPLIT_CSV.exists():
        raise FileNotFoundError(f"Fichier manquant: {VAL_SPLIT_CSV}")
    if not TEST_SPLIT_CSV.exists():
        raise FileNotFoundError(f"Fichier manquant: {TEST_SPLIT_CSV}")

    df_val = pd.read_csv(VAL_SPLIT_CSV)
    df_test = pd.read_csv(TEST_SPLIT_CSV)

    if "report_id" not in df_val.columns:
        raise ValueError(f"Colonne 'report_id' absente dans {VAL_SPLIT_CSV}")
    if "report_id" not in df_test.columns:
        raise ValueError(f"Colonne 'report_id' absente dans {TEST_SPLIT_CSV}")

    val_ids = df_val["report_id"].astype(str).tolist()
    test_ids = df_test["report_id"].astype(str).tolist()
    return val_ids, test_ids


def build_split_map(val_ids: List[str], test_ids: List[str]) -> Dict[str, str]:
    split_map = {rid: "val" for rid in val_ids}
    split_map.update({rid: "test" for rid in test_ids})
    return split_map


# -----------------------------
# UTILITAIRES ANNOTATIONS
# -----------------------------
def read_annotations_df() -> pd.DataFrame:
    if not ANN_CSV.exists():
        return pd.DataFrame(columns=OUT_COLUMNS)
    try:
        df = pd.read_csv(ANN_CSV)
    except Exception:
        # Si le CSV est corrompu, on force une lecture robuste minimale
        df = pd.read_csv(ANN_CSV, on_bad_lines="skip")
    # Normalisation: report_id en str
    if "report_id" in df.columns:
        df["report_id"] = df["report_id"].astype(str)
    return df


def annotated_id_set(df_ann: pd.DataFrame) -> set:
    if df_ann.empty or "report_id" not in df_ann.columns:
        return set()
    return set(df_ann["report_id"].astype(str).tolist())


def latest_annotation_for(report_id: str, df_ann: pd.DataFrame) -> Optional[dict]:
    if df_ann.empty:
        return None
    if "report_id" not in df_ann.columns:
        return None
    sub = df_ann[df_ann["report_id"].astype(str) == str(report_id)]
    if sub.empty:
        return None
    row = sub.iloc[-1].to_dict()
    return row


def append_annotation_row(row: dict):
    """
    Ecriture append-only. Si filelock dispo, on lock pendant l'écriture.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _write():
        file_exists = ANN_CSV.exists()
        with open(ANN_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
            if not file_exists:
                w.writeheader()
            # Remplir toutes les colonnes manquantes
            safe_row = {k: row.get(k, "") for k in OUT_COLUMNS}
            w.writerow(safe_row)

    if FileLock is None:
        _write()
    else:
        lock = FileLock(str(LOCK_PATH))
        with lock:
            _write()


# -----------------------------
# LOGIQUE "QUEUE"
# -----------------------------
def build_queue(
    subset_mode: str,
    seed: int,
    skip_annotated: bool,
    df_ann: pd.DataFrame,
) -> Tuple[List[str], Dict[str, str]]:
    val_ids, test_ids = load_splits()
    split_map = build_split_map(val_ids, test_ids)

    if subset_mode == "val":
        order = val_ids
    elif subset_mode == "test":
        order = test_ids
    elif subset_mode == "val+test (shuffle)":
        order = val_ids + test_ids
        rnd = random.Random(int(seed))
        rnd.shuffle(order)
    else:
        raise ValueError(f"subset_mode inconnu: {subset_mode}")

    if skip_annotated:
        done = annotated_id_set(df_ann)
        order = [rid for rid in order if rid not in done]

    return order, split_map


def format_progress(i: int, n: int) -> str:
    if n == 0:
        return "0 / 0"
    return f"{min(i+1, n)} / {n}"


# -----------------------------
# CALLBACKS GRADIO
# -----------------------------
def cb_start(subset_mode: str, seed: int, skip_annotated: bool, annotator: str, state: dict):
    df_ann = read_annotations_df()
    order, split_map = build_queue(subset_mode, seed, skip_annotated, df_ann)

    state = {
        "subset_mode": subset_mode,
        "seed": int(seed),
        "skip_annotated": bool(skip_annotated),
        "annotator": annotator.strip() if annotator else "",
        "order": order,
        "split_map": split_map,
        "i": 0,
        "df_ann": df_ann,  # cache
    }

    return cb_load_current(state)


def cb_load_current(state: dict):
    order = state.get("order", [])
    i = state.get("i", 0)
    df_ann = state.get("df_ann", pd.DataFrame(columns=OUT_COLUMNS))
    split_map = state.get("split_map", {})

    if not order:
        header = "✅ Rien à annoter (liste vide ou tout est déjà annoté)."
        return (
            state,
            header,
            "",
            [],
            *[None for _ in LABEL_COLUMNS],
            "",
            "Status: queue vide.",
        )

    if i < 0:
        i = 0
    if i >= len(order):
        i = len(order) - 1
    state["i"] = i

    report_id = order[i]
    split = split_map.get(report_id, "unknown")

    # Charge images + texte
    imgs = list_exam_images(report_id)
    gallery_value = [(str(p), p.name) for p in imgs]  # (filepath, caption)

    txt_path = FA_DIR / f"{report_id}.txt"
    report_text = read_text_file(txt_path)
    if not report_text:
        report_text = "(rapport .txt manquant ou vide)"

    # Pré-remplissage (si déjà annoté et si skip_annotated=False, ou pour review)
    last = latest_annotation_for(report_id, df_ann)
    prefills = {k: None for k in LABEL_COLUMNS}
    comment = ""

    if last is not None:
        for k in LABEL_COLUMNS:
            v = last.get(k, None)
            v = None if pd.isna(v) else v
            if v in LABEL_CHOICES:
                prefills[k] = v
        c = last.get("comment", "")
        comment = "" if pd.isna(c) else str(c)

    header = (
        f"### Examen: `{report_id}`  \n"
        f"**Split:** {split}  |  **Progress:** {format_progress(i, len(order))}  \n"
        f"**Images:** {len(imgs)}"
    )

    status = "Status: prêt."
    if len(imgs) == 0:
        status = "⚠️ Status: aucune image trouvée pour cet examen (tu peux Skip ou annoter quand même)."

    return (
        state,
        header,
        report_text,
        gallery_value,
        *[prefills[k] for k in LABEL_COLUMNS],
        comment,
        status,
    )


def _validate_labels(values: List[Optional[str]]) -> Optional[str]:
    for v in values:
        if v not in LABEL_CHOICES:
            return "Merci de sélectionner une valeur pour tous les labels (oui/non/inconnu)."
    return None


def cb_save_next(
    fracture_visible: str,
    deplacement_ou_incongruence: str,
    consolidation_ou_reaction_periostee: str,
    materiel_implant: str,
    osteotomie_ou_arthrodese: str,
    comment: str,
    state: dict,
):
    order = state.get("order", [])
    i = state.get("i", 0)
    split_map = state.get("split_map", {})
    df_ann = state.get("df_ann", pd.DataFrame(columns=OUT_COLUMNS))

    if not order:
        return cb_load_current(state)

    report_id = order[i]
    split = split_map.get(report_id, "unknown")

    label_values = [
        fracture_visible,
        deplacement_ou_incongruence,
        consolidation_ou_reaction_periostee,
        materiel_implant,
        osteotomie_ou_arthrodese,
    ]
    err = _validate_labels(label_values)
    if err is not None:
        # On renvoie juste un status d'erreur (sans bouger)
        out = list(cb_load_current(state))
        out[-1] = f"❌ Status: {err}"
        return tuple(out)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "annotator": state.get("annotator", ""),
        "subset_mode": state.get("subset_mode", ""),
        "seed": state.get("seed", ""),
        "split": split,
        "report_id": report_id,
        "fracture_visible": fracture_visible,
        "deplacement_ou_incongruence": deplacement_ou_incongruence,
        "consolidation_ou_reaction_periostee": consolidation_ou_reaction_periostee,
        "materiel_implant": materiel_implant,
        "osteotomie_ou_arthrodese": osteotomie_ou_arthrodese,
        "comment": comment or "",
        "skipped": 0,
    }
    append_annotation_row(row)

    # MAJ cache df_ann (append en mémoire)
    df_ann = pd.concat([df_ann, pd.DataFrame([row])], ignore_index=True)
    state["df_ann"] = df_ann

    # Next
    state["i"] = min(i + 1, max(len(order) - 1, 0))

    # Si on est à la fin, on laisse l'index sur le dernier et on affiche un status
    if i == len(order) - 1:
        out = list(cb_load_current(state))
        out[-1] = "✅ Status: dernier examen annoté. (Queue terminée pour ce mode)"
        return tuple(out)

    out = list(cb_load_current(state))
    out[-1] = "✅ Status: sauvegardé, passage à l’examen suivant."
    return tuple(out)


def cb_skip(state: dict):
    order = state.get("order", [])
    i = state.get("i", 0)
    split_map = state.get("split_map", {})
    df_ann = state.get("df_ann", pd.DataFrame(columns=OUT_COLUMNS))

    if not order:
        return cb_load_current(state)

    report_id = order[i]
    split = split_map.get(report_id, "unknown")

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "annotator": state.get("annotator", ""),
        "subset_mode": state.get("subset_mode", ""),
        "seed": state.get("seed", ""),
        "split": split,
        "report_id": report_id,
        # Labels vides pour "skip"
        "fracture_visible": "",
        "deplacement_ou_incongruence": "",
        "consolidation_ou_reaction_periostee": "",
        "materiel_implant": "",
        "osteotomie_ou_arthrodese": "",
        "comment": "SKIPPED",
        "skipped": 1,
    }
    append_annotation_row(row)

    df_ann = pd.concat([df_ann, pd.DataFrame([row])], ignore_index=True)
    state["df_ann"] = df_ann

    # Next
    state["i"] = min(i + 1, max(len(order) - 1, 0))

    if i == len(order) - 1:
        out = list(cb_load_current(state))
        out[-1] = "✅ Status: dernier examen (skipped). Queue terminée pour ce mode."
        return tuple(out)

    out = list(cb_load_current(state))
    out[-1] = "⚠️ Status: examen skipped, passage au suivant."
    return tuple(out)


def cb_prev(state: dict):
    order = state.get("order", [])
    if not order:
        return cb_load_current(state)
    state["i"] = max(int(state.get("i", 0)) - 1, 0)
    out = list(cb_load_current(state))
    out[-1] = "Status: examen précédent."
    return tuple(out)


# -----------------------------
# UI GRADIO
# -----------------------------
with gr.Blocks(title="FA Manual Annotation Tool") as demo:
    gr.Markdown("## Outil d’annotation manuelle (val / test / val+test)")

    with gr.Row():
        subset_mode = gr.Radio(
            choices=["val", "test", "val+test (shuffle)"],
            value="val",
            label="Subset à annoter",
        )
        seed = gr.Number(value=42, precision=0, label="Seed (utilisé si val+test shuffle)")
        skip_annotated = gr.Checkbox(value=True, label="Ignorer les examens déjà annotés (reprise automatique)")
        annotator = gr.Textbox(value=os.getenv("USER", ""), label="Annotateur (optionnel)")

    with gr.Row():
        btn_start = gr.Button("Start / Reload", variant="primary")
        btn_prev = gr.Button("Prev")
        btn_skip = gr.Button("Skip")
        # Save&Next sera plus bas près des labels pour être ergonomique

    state = gr.State({})

    header_md = gr.Markdown("### (non démarré)")
    status_md = gr.Markdown("Status: clique sur **Start / Reload**.")

    with gr.Row():
        report_text = gr.Textbox(label="Rapport (.txt)", lines=16, interactive=False)
        gallery = gr.Gallery(
            label="Images de l’examen",
            columns=2,
            height=520,
            preview=True,
            show_label=True,
        )

    gr.Markdown("---")
    gr.Markdown("### Labels (oui / non / inconnu)")

    with gr.Row():
        fracture_visible = gr.Radio(LABEL_CHOICES, value=None, label="fracture_visible")
        deplacement_ou_incongruence = gr.Radio(LABEL_CHOICES, value=None, label="deplacement_ou_incongruence")
        consolidation_ou_reaction_periostee = gr.Radio(LABEL_CHOICES, value=None, label="consolidation_ou_reaction_periostee")
        materiel_implant = gr.Radio(LABEL_CHOICES, value=None, label="materiel_implant")
        osteotomie_ou_arthrodese = gr.Radio(LABEL_CHOICES, value=None, label="osteotomie_ou_arthrodese")

    comment = gr.Textbox(label="Commentaire (optionnel)", lines=1)

    btn_save_next = gr.Button("Save & Next", variant="primary")

    # Wiring callbacks
    btn_start.click(
        cb_start,
        inputs=[subset_mode, seed, skip_annotated, annotator, state],
        outputs=[
            state,
            header_md,
            report_text,
            gallery,
            fracture_visible,
            deplacement_ou_incongruence,
            consolidation_ou_reaction_periostee,
            materiel_implant,
            osteotomie_ou_arthrodese,
            comment,
            status_md,
        ],
    )

    btn_prev.click(
        cb_prev,
        inputs=[state],
        outputs=[
            state,
            header_md,
            report_text,
            gallery,
            fracture_visible,
            deplacement_ou_incongruence,
            consolidation_ou_reaction_periostee,
            materiel_implant,
            osteotomie_ou_arthrodese,
            comment,
            status_md,
        ],
    )

    btn_skip.click(
        cb_skip,
        inputs=[state],
        outputs=[
            state,
            header_md,
            report_text,
            gallery,
            fracture_visible,
            deplacement_ou_incongruence,
            consolidation_ou_reaction_periostee,
            materiel_implant,
            osteotomie_ou_arthrodese,
            comment,
            status_md,
        ],
    )

    btn_save_next.click(
        cb_save_next,
        inputs=[
            fracture_visible,
            deplacement_ou_incongruence,
            consolidation_ou_reaction_periostee,
            materiel_implant,
            osteotomie_ou_arthrodese,
            comment,
            state,
        ],
        outputs=[
            state,
            header_md,
            report_text,
            gallery,
            fracture_visible,
            deplacement_ou_incongruence,
            consolidation_ou_reaction_periostee,
            materiel_implant,
            osteotomie_ou_arthrodese,
            comment,
            status_md,
        ],
    )

if __name__ == "__main__":
    # server_name="0.0.0.0" si tu veux accéder depuis une autre machine du réseau
    demo.launch(server_name="127.0.0.1", server_port=7860)
