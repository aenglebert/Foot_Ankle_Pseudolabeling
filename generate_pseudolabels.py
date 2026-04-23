#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import re
import difflib
from typing import Dict, List, Optional, Sequence, Set, Tuple, Any

from vllm import LLM, SamplingParams

# Optionnel: llm_json tolère parfois des JSON "sales"
try:
    from llm_json import json as llmjson  # type: ignore
except Exception:
    llmjson = None

import json as stdjson


# =========================
# LABEL SCHEMA
# =========================

EXPECTED_KEYS = [
    "fracture_visible",
    "deplacement_ou_incongruence",
    "consolidation_ou_reaction_periostee",
    "materiel_implant",
    "osteotomie_ou_arthrodese",
]
ALLOWED_VALUES = {"oui", "non", "inconnu"}


# =========================
# MODEL PROFILES
# =========================
# Ajout d’un modèle = ajouter une entrée ici.
MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
    "ministral3_3b": {
        "model_name": "unsloth/Ministral-3-3B-Instruct-2512-FP8",
        "llm_kwargs": {
            "max_model_len": 4096,
        },
        "sampling_defaults": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 512,
        },
        "batch_size": 64,
    },
    "qwen3_4b_fp8": {
        "model_name": "Qwen/Qwen3-4B-Instruct-2507-FP8",
        "llm_kwargs": {
            "max_model_len": 4096,
            "max_num_seqs": 64,  # spécifique à ton script qwen
        },
        "sampling_defaults": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 512,
        },
        "batch_size": 64,
    },
}


# =========================
# NORMALIZATION + JSON PARSING
# =========================

def normalize_keys(data: dict) -> Dict[str, str]:
    corrected: Dict[str, str] = {}

    # 1) corriger/aligner les clés existantes
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        match = difflib.get_close_matches(key, EXPECTED_KEYS, n=1, cutoff=0.7)
        if match:
            corrected_key = match[0]
            corrected[corrected_key] = value

    # 2) ajouter clés manquantes
    for key in EXPECTED_KEYS:
        if key not in corrected:
            corrected[key] = "inconnu"

    # 3) vérifier valeurs
    for key in EXPECTED_KEYS:
        v = corrected.get(key)
        if isinstance(v, str):
            v_norm = v.strip().lower()
        else:
            v_norm = "inconnu"

        if v_norm not in ALLOWED_VALUES:
            v_norm = "inconnu"
        corrected[key] = v_norm

    return corrected


def extract_json_object(text: str) -> Optional[str]:
    """
    Extrait le premier bloc JSON { ... } trouvé dans le texte.
    Non-gourmand (.*?), pour éviter d'englober trop large.
    """
    if not text:
        return None
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    return m.group(0) if m else None


def parse_json_best_effort(json_str: str) -> Optional[dict]:
    if not json_str:
        return None

    if llmjson is not None:
        try:
            obj = llmjson.loads(json_str)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    try:
        obj = stdjson.loads(json_str)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# =========================
# PROMPT
# =========================

def build_prompt(report_text: str) -> str:
    return f"""
Tu es un système d'extraction structurée.

Règles OBLIGATOIRES :

1. Retourne un JSON valide.
2. Ne retourne QUE ce JSON.
3. N'ajoute aucun commentaire.
4. Chaque champ doit contenir UNE SEULE valeur.
5. AUCUNE liste.
6. AUCUNE clé supplémentaire.
7. Utilise EXACTEMENT les clés suivantes :

- fracture_visible
- deplacement_ou_incongruence
- consolidation_ou_reaction_periostee
- materiel_implant
- osteotomie_ou_arthrodese

Valeurs autorisées UNIQUEMENT :
- "oui"
- "non"
- "inconnu"

Si plusieurs éléments sont mentionnés :
→ Si au moins un est "oui", alors mets "oui"
→ Sinon si explicitement absent, mets "non"
→ Sinon mets "inconnu"

DÉFINITIONS / RÈGLES DE CODAGE :
- fracture_visible = "oui" si fracture décrite ou trait/fragment compatible fracture ; "non" si absence ; "inconnu" si non mentionné ou ambigu.
- deplacement_ou_incongruence = "oui" si déplacement, diastasis, subluxation/luxation, incongruence, step-off ; "non" si alignement conservé ; "inconnu" si non mentionné ou ambigu.
- consolidation_ou_reaction_periostee = "oui" si cal, consolidation, pontage, réparation, réaction périostée ; "non" si absence ; "inconnu" si non mentionné ou ambigu.
- materiel_implant = "oui" si vis/plaque/broche/agrafes/clou/prothèse/ancres/fixateur externe ; sinon "non" ; "inconnu" si non mentionné ou ambigu.
- osteotomie_ou_arthrodese = "oui" si ostéotomie ou arthrodèse mentionnée/compatible ; "non" sinon ; "inconnu" si non mentionné ou ambigu.


Format EXACT attendu :

{{
  "fracture_visible": "oui | non | inconnu",
  "deplacement_ou_incongruence": "oui | non | inconnu",
  "consolidation_ou_reaction_periostee": "oui | non | inconnu",
  "materiel_implant": "oui | non | inconnu",
  "osteotomie_ou_arthrodese": "oui | non | inconnu"
}}

TEXTE :
\"\"\"
{report_text}
\"\"\"
"""


# =========================
# SPLITS / SELECTION
# =========================

def load_report_ids_from_csv(path: Path) -> Set[str]:
    if not path.exists():
        raise FileNotFoundError(f"CSV introuvable: {path}")

    report_ids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "report_id" not in reader.fieldnames:
            raise ValueError(f"Le CSV {path} ne contient pas de colonne 'report_id'.")
        for row in reader:
            rid = (row.get("report_id") or "").strip()
            if rid:
                report_ids.add(rid)
    return report_ids


def load_report_ids_from_splits_dir(splits_dir: Path, splits: Sequence[str]) -> Set[str]:
    all_ids: Set[str] = set()
    for s in splits:
        s = s.strip().lower()
        if s not in {"train", "val", "test"}:
            raise ValueError(f"Split invalide: {s} (attendu: train/val/test)")
        p = splits_dir / f"{s}_exams.csv"
        all_ids |= load_report_ids_from_csv(p)
    return all_ids


def select_txt_files(input_dir: Path, report_ids: Optional[Set[str]] = None) -> Tuple[List[Path], List[str]]:
    txt_files = sorted(input_dir.glob("*.txt"))

    if report_ids is None:
        return txt_files, [p.stem for p in txt_files]

    filtered_files: List[Path] = []
    filtered_ids: List[str] = []
    for p in txt_files:
        if p.stem in report_ids:
            filtered_files.append(p)
            filtered_ids.append(p.stem)
    return filtered_files, filtered_ids


# =========================
# GENERATION
# =========================

def run_generation(
    llm: LLM,
    sampling_params: SamplingParams,
    prompts: List[str],
    file_ids: List[str],
    batch_size: int = 64,
) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []

    n = len(prompts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        prompts_b = prompts[start:end]
        ids_b = file_ids[start:end]

        outputs = llm.generate(prompts_b, sampling_params)

        for i, out in enumerate(outputs):
            rid = ids_b[i]
            generated_text = out.outputs[0].text if out.outputs else ""

            json_str = extract_json_object(generated_text)
            if json_str is None:
                print(f"⚠ Aucun JSON trouvé pour {rid}")
                continue

            data = parse_json_best_effort(json_str)
            if data is None:
                print(f"⚠ JSON invalide pour {rid}")
                continue

            data_norm = normalize_keys(data)
            data_norm["report_id"] = rid
            results.append(data_norm)

    return results


def export_csv(results: List[Dict[str, str]], output_csv: Path) -> None:
    fieldnames = ["report_id"] + EXPECTED_KEYS
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Export terminé → {output_csv}")


# =========================
# CLI
# =========================

def parse_args():
    p = argparse.ArgumentParser("Generate pseudo-labels from radiology reports")

    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--output_csv", type=Path, required=True)

    p.add_argument(
        "--profile",
        type=str,
        required=True,
        choices=sorted(MODEL_PROFILES.keys()),
        help="Profil modèle (définit model_name + llm_kwargs + sampling defaults).",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optionnel: override du model_name du profil.",
    )

    # overrides sampling
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--max_tokens", type=int, default=None)

    # overrides runtime
    p.add_argument("--batch_size", type=int, default=None)

    # selection mode
    p.add_argument("--splits_dir", type=Path, default=None)
    p.add_argument("--use_splits", nargs="*", default=None, help="ex: val test")
    p.add_argument("--report_ids_csv", type=Path, default=None, help="CSV avec colonne report_id")

    return p.parse_args()


def main():
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"input_dir introuvable: {args.input_dir}")

    prof = MODEL_PROFILES[args.profile]
    model_name = args.model_name or prof["model_name"]
    llm_kwargs = dict(prof.get("llm_kwargs", {}))

    sampling_defaults = dict(prof.get("sampling_defaults", {}))
    temperature = sampling_defaults["temperature"] if args.temperature is None else args.temperature
    top_p = sampling_defaults["top_p"] if args.top_p is None else args.top_p
    max_tokens = sampling_defaults["max_tokens"] if args.max_tokens is None else args.max_tokens

    batch_size = prof.get("batch_size", 64) if args.batch_size is None else args.batch_size

    # 1) selection IDs
    report_ids: Optional[Set[str]] = None

    if args.report_ids_csv is not None:
        report_ids = load_report_ids_from_csv(args.report_ids_csv)

    if args.use_splits is not None and len(args.use_splits) > 0:
        if args.splits_dir is None:
            raise ValueError("--use_splits requiert --splits_dir")
        ids_from_splits = load_report_ids_from_splits_dir(args.splits_dir, args.use_splits)
        report_ids = ids_from_splits if report_ids is None else (report_ids | ids_from_splits)

    # 2) select files
    txt_files, file_ids = select_txt_files(args.input_dir, report_ids=report_ids)

    n_all = len(list(args.input_dir.glob("*.txt")))
    n_sel = len(txt_files)

    print("======================================")
    print(f"📁 input_dir:  {args.input_dir}")
    print(f"🧾 .txt total: {n_all}")
    if report_ids is None:
        print("🎯 mode: FULL DATASET (tous les .txt)")
    else:
        print(f"🎯 mode: SUBSET | report_ids demandés: {len(report_ids)} | .txt matchés: {n_sel}")
        existing = {p.stem for p in args.input_dir.glob("*.txt")}
        missing = sorted(list(report_ids - existing))
        if missing:
            print(f"⚠ report_id demandés mais sans .txt: {len(missing)} (ex: {', '.join(missing[:10])})")

    if n_sel == 0:
        print("❌ Aucun fichier à traiter après filtrage.")
        return

    # 3) prompts
    prompts: List[str] = []
    for pth in txt_files:
        text = pth.read_text(encoding="utf-8")
        prompts.append(build_prompt(text))

    # 4) vLLM init
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    llm = LLM(model=model_name, **llm_kwargs)

    print("======================================")
    print(f"🤖 profile:    {args.profile}")
    print(f"🤖 model_name:  {model_name}")
    print(f"⚙️ llm_kwargs:  {llm_kwargs}")
    print(f"⚙️ sampling:    temp={temperature} top_p={top_p} max_tokens={max_tokens}")
    print(f"🚀 n={len(prompts)} | batch_size={batch_size}")
    print("======================================")

    # 5) generation
    results = run_generation(
        llm=llm,
        sampling_params=sampling_params,
        prompts=prompts,
        file_ids=file_ids,
        batch_size=batch_size,
    )

    # 6) export
    if results:
        export_csv(results, args.output_csv)
        print(f"✅ lignes exportées: {len(results)} / {len(prompts)}")
    else:
        print("❌ Aucun résultat exporté (JSON invalides/absents).")


if __name__ == "__main__":
    main()
