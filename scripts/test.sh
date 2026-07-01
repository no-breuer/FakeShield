#!/usr/bin/env bash
# Run the full FakeShield evaluation pipeline (DTE-FDM then MFLM).
#
# All paths are configurable via environment variables:
#   WEIGHT_PATH      dir with DTE-FDM/, MFLM/, DTG.pth   (default ./weight/fakeshield-v1-22b)
#   QUESTION_PATH    question JSONL from eval_jsonl.py  (default ./playground/test.jsonl)
#   DTE_FDM_OUTPUT   where DTE-FDM writes its answers    (default ./playground/DTE-FDM_output.jsonl)
#   MFLM_OUTPUT      where MFLM writes predicted masks   (default ./playground/MFLM_output)
#   CUDA_VISIBLE_DEVICES                              (default 0)
#
# NOTE on transformers versions:
#   DTE-FDM (LLaVA) needs transformers==4.37.2; MFLM (GLaMM) needs 4.28.0.
#   The original repo "solved" this by re-pip-installing between stages. We keep
#   that behavior so a single environment works. If you maintain two envs
#   instead, split this script at the MFLM stage and drop the re-pins.
set -euo pipefail

WEIGHT_PATH="${WEIGHT_PATH:-./weight/fakeshield-v1-22b}"
QUESTION_PATH="${QUESTION_PATH:-./playground/test.jsonl}"
DTE_FDM_OUTPUT="${DTE_FDM_OUTPUT:-./playground/DTE-FDM_output.jsonl}"
MFLM_OUTPUT="${MFLM_OUTPUT:-./playground/MFLM_output}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export CUDA_VISIBLE_DEVICES

echo "======== Stage 1/2: DTE-FDM (detection + explanation) ========"
pip install -q transformers==4.37.2
python ./DTE-FDM/llava/eval/model_vqa.py \
    --model-path "${WEIGHT_PATH}/DTE-FDM" \
    --DTG-path "${WEIGHT_PATH}/DTG.pth" \
    --question-file "${QUESTION_PATH}" \
    --image-folder / \
    --answers-file "${DTE_FDM_OUTPUT}"

echo "======== Stage 2/2: MFLM (localization) ========"
pip install -q transformers==4.28.0
python ./MFLM/test.py \
    --version "${WEIGHT_PATH}/MFLM" \
    --DTE-FDM-output "${DTE_FDM_OUTPUT}" \
    --MFLM-output "${MFLM_OUTPUT}"

echo "======== Done. Score outputs with: ========"
echo "  python playground/evaluate_metrics.py \\"
echo "      --dte-fdm-output ${DTE_FDM_OUTPUT} --labels ./playground/test_labels.jsonl \\"
echo "      --pred-dir ${MFLM_OUTPUT} --gt-dir <GT_MASK_DIR> --per-image"
