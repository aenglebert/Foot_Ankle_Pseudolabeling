PSEUDO-LABELING & EVALUATION PIPELINE (FOOT/ANKLE RADIOLOGY REPORTS)
===================================================================

This repository implements a small pipeline to:
  1) build dataset splits (train / val / test) based on report_id,
  2) perform manual annotation (val/test),
  3) clean/standardize manual labels into a "gold" reference,
  4) generate LLM pseudo-labels from raw text reports,
  5) evaluate pseudo-labels vs gold (metrics + error listings),
  6) analyze abstention behavior (value "inconnu").


KEY ID / JOIN STRATEGY
----------------------

Everything is joined using report_id (report_id = patient_id + "_" + study_id)

  - fa/{report_id}.txt                      raw source text
  - splits/*_exams.csv                      contains report_id for each splits (train, val, test)
  - annotations/*                           manual labels + cleaned gold (contains report_id)
  - pseudolabels/*.csv                      LLM outputs (contains report_id)


LABEL SCHEMA (5 TASKS)
----------------------

Expected columns (both gold and pseudo-label files):

  - fracture_visible
  - deplacement_ou_incongruence
  - consolidation_ou_reaction_periostee
  - materiel_implant
  - osteotomie_ou_arthrodese

Allowed values ONLY:

  - oui
  - non
  - inconnu

Meaning:
  - "inconnu" is abstention / not mentioned / ambiguous.
  - Do NOT treat "inconnu" as "non".


SPLITS
------

Script: split_sets.py

Goal:
  - build splits/train_exams.csv, splits/val_exams.csv, splits/test_exams.csv

Run:
  python split_sets.py


MANUAL ANNOTATION (GUI)
-----------------------

Script: annotations.py

Goal:
  - create or extend a manual annotation file using a gradio gui:
      annotations/manual_val_test.csv
  - typical use: annotate validation and/or test set.

Run:
  python annotations.py


CLEAN / STANDARDIZE MANUAL LABELS (GOLD)
----------------------------------------

Script: prepare_labels.py

Inputs:
  - annotations/manual_val_test.csv
  - splits/val_exams.csv and splits/test_exams.csv (for filtering/consistency)

Outputs (cleaned gold):
  - annotations/gold_clean/gold_val.csv
  - annotations/gold_clean/gold_test.csv
  - annotations/gold_clean/gold_all.csv

Run:
  python prepare_labels.py

Notes:
  - this step enforces the strict value set (oui/non/inconnu)
  - keeps a consistent schema across all downstream steps


GENERATE LLM PSEUDO-LABELS
--------------------------

Script: generate_pseudolabels.py

Goal:
  - read fa/{report_id}.txt
  - query an LLM with a strict JSON-only prompt
  - write predictions to pseudolabels/*.csv (must include report_id + 5 label columns)

exemple:
python generate_pseudolabels.py \
  --profile qwen3_4b_fp8 \
  --input_dir ./fa \
  --output_csv pseudolabels/qwen_valtest.csv \
  --splits_dir splits \
  --use_splits val test

      
OPTIONAL: SPLIT / FILTER PSEUDO-LABEL FILES
-------------------------------------------

Script: split_pseudolabels.py

Goal:
  - from a single pseudo-label file, generate separate subsets aligned with splits
  - ensure only report_id present in the split is kept

Run:
  python split_pseudolabels.py


EVALUATION (PSEUDO-LABELS vs GOLD)
----------------------------------

Script: evaluate_pseudolabels.py

Inputs (typical):
  - gold:
      annotations/gold_clean/gold_val.csv
      annotations/gold_clean/gold_test.csv
  - predictions:
      pseudolabels/qwen_valtest.csv

Outputs:
  - a results directory with:
      * summary metrics (val/test/all)
      * per-label metrics
      * error listings (cases where the model answered but was wrong)
      * abstention listings (cases containing at least one "inconnu")

Example:
  python evaluate_pseudolabels.py \
    --gold_val  annotations/gold_clean/gold_val.csv \
    --gold_test annotations/gold_clean/gold_test.csv \
    --pred pseudolabels/qwen_valtest.csv \
    --out_dir results/qwen/prompt_XX


WORKFLOW
--------

1) Create splits
     python split_sets.py

2) Manual annotation (val/test)
     python annotations.py
   -> annotations/manual_val_test.csv

3) Build cleaned gold files
     python prepare_labels.py
   -> annotations/gold_clean/gold_{val,test,all}.csv

4) Iterate on validation (prompt/model selection)
     python generate_pseudolabels.py --use_splits val
     python evaluate_pseudolabels.py ...

5) Final evaluation on test (hold-out)
     python generate_pseudolabels.py --use_splits test
     python evaluate_pseudolabels.py ...
