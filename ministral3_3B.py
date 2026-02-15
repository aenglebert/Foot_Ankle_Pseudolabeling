from pathlib import Path
from llm_json import json
import csv
from vllm import LLM, SamplingParams
import re
import difflib

EXPECTED_KEYS = [
    "fracture_visible",
    "deplacement_ou_incongruence",
    "consolidation_ou_reaction_periostee",
    "materiel_implant",
    "osteotomie_ou_arthrodese",
]

ALLOWED_VALUES = {"oui", "non", "inconnu"}

def normalize_keys(data: dict) -> dict:
    corrected = {}

    # 1️⃣ Corriger les clés existantes
    for key, value in data.items():
        match = difflib.get_close_matches(key, EXPECTED_KEYS, n=1, cutoff=0.7)
        if match:
            corrected_key = match[0]
            corrected[corrected_key] = value

    # 2️⃣ Ajouter les clés manquantes
    for key in EXPECTED_KEYS:
        if key not in corrected:
            corrected[key] = "inconnu"

    # 3️⃣ Vérifier valeurs
    for key in EXPECTED_KEYS:
        if corrected[key] not in ALLOWED_VALUES:
            corrected[key] = "inconnu"

    return corrected


def extract_json_object(text: str) -> str:
    """
    Extrait le premier bloc JSON { ... } trouvé dans le texte.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return None

# =========================
# CONFIG
# =========================

INPUT_DIR = Path("./fa")
OUTPUT_CSV = Path("ministral3_3B_output.csv")

MODEL_NAME = "mistralai/Ministral-3-3B-Instruct-2512"

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=512,
)

llm = LLM(
    model=MODEL_NAME,
    max_model_len=4096,
)

# =========================
# PROMPT TEMPLATE
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
# LOAD FILES
# =========================

txt_files = list(INPUT_DIR.glob("*.txt"))
print(f"{len(txt_files)} fichiers trouvés.")

prompts = []
file_ids = []

for path in txt_files:
    report_id = path.stem
    text = path.read_text(encoding="utf-8")

    prompts.append(build_prompt(text))
    file_ids.append(report_id)

# =========================
# INFERENCE
# =========================

outputs = llm.generate(prompts, sampling_params)

results = []

for i, output in enumerate(outputs):
    generated_text = output.outputs[0].text

    json_str = extract_json_object(generated_text)

    if json_str is None:
        print(f"⚠ Aucun JSON trouvé pour {file_ids[i]}")
        continue

    try:
        data = json.loads(extract_json_object(json_str))
    except Exception:
        print(f"⚠ JSON invalide pour {file_ids[i]}")
        continue

    data = normalize_keys(data)
    data["report_id"] = file_ids[i]

    results.append(data)

# =========================
# EXPORT CSV
# =========================

if results:
    fieldnames = [
        "report_id",
        "fracture_visible",
        "deplacement_ou_incongruence",
        "consolidation_ou_reaction_periostee",
        "materiel_implant",
        "osteotomie_ou_arthrodese",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Export terminé → {OUTPUT_CSV}")
else:
    print("❌ Aucun résultat exporté.")
