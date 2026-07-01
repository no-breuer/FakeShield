#!/usr/bin/env bash
# Download FakeShield weights and the evaluation datasets that are scriptable.
# Run this ON THE GPU CLUSTER (large files, fast network, plenty of disk).
#
# Datasets that require manual login/registration cannot be downloaded here; see
# the MANUAL DOWNLOADS section at the bottom for links + instructions.
set -euo pipefail

WEIGHT_DIR="${WEIGHT_DIR:-./weight}"
DATA_DIR="${DATA_DIR:-./dataset}"

# --------------------------------------------------------------------------- #
# 0. Prereqs
# --------------------------------------------------------------------------- #
pip install -q huggingface_hub
mkdir -p "${WEIGHT_DIR}" "${DATA_DIR}"

# --------------------------------------------------------------------------- #
# 1. Model weights (~40 GB total)
# --------------------------------------------------------------------------- #
echo "==== Downloading FakeShield weights (~40 GB) ===="
hf download --resume zhipeixu/fakeshield-v1-22b --local-dir "${WEIGHT_DIR}/fakeshield-v1-22b"

echo "==== Downloading SAM ViT-H weight (~2.5 GB) ===="
hf download --resume ybelkada/segment-anything \
    checkpoints/sam_vit_h_4b8939.pth --local-dir "${WEIGHT_DIR}/_sam_tmp"
mv "${WEIGHT_DIR}/_sam_tmp/checkpoints/sam_vit_h_4b8939.pth" "${WEIGHT_DIR}/sam_vit_h_4b8939.pth"
rm -rf "${WEIGHT_DIR}/_sam_tmp"

# --------------------------------------------------------------------------- #
# 2. Scriptable datasets
# --------------------------------------------------------------------------- #
echo "==== Downloading SD_inpaint dataset (AIGC editing) ===="
hf download --resume zhipeixu/SD_inpaint_dataset --repo-type dataset \
    --local-dir "${DATA_DIR}/aigc/SD_inpaint"

echo "==== Downloading MMTD-Set-34k ===="
hf download --resume zhipeixu/MMTD-Set-34k --repo-type dataset \
    --local-dir "${DATA_DIR}/MMTD_Set"

echo "==== Downloading coverage dataset (Photoshop forgery) ===="
git clone --depth 1 https://github.com/wenbihan/coverage "${DATA_DIR}/photoshop/coverage_raw"
# coverage ships images + masks together; arrange into image/ and mask/.
DATA_DIR="${DATA_DIR}" python - <<'PY'
import os, shutil, glob
src = os.path.join(os.environ["DATA_DIR"], "photoshop", "coverage_raw")
dst_img = os.path.join(os.environ["DATA_DIR"], "photoshop", "coverage", "image")
dst_mask = os.path.join(os.environ["DATA_DIR"], "photoshop", "coverage", "mask")
os.makedirs(dst_img, exist_ok=True); os.makedirs(dst_mask, exist_ok=True)
for f in glob.glob(os.path.join(src, "*")):
    b = os.path.basename(f).lower()
    if "forge" in b or "tamper" in b or (b.endswith(".png") and "mask" in b):
        shutil.copy(f, os.path.join(dst_mask, os.path.basename(f)))
    else:
        shutil.copy(f, os.path.join(dst_img, os.path.basename(f)))
print(f"coverage -> {dst_img} / {dst_mask}")
PY

echo "==== Downloading CASIA1+ via PSCC-Net repo (Photoshop forgery) ===="
# PSCC-Net README lists CASIA1+ under its testing section.
git clone --depth 1 https://github.com/proteus1991/PSCC-Net "${DATA_DIR}/photoshop/PSCC-Net_raw"
echo "  -> manually move CASIA1+ tampered/auth images from"
echo "     ${DATA_DIR}/photoshop/PSCC-Net_raw into ${DATA_DIR}/photoshop/CASIAv1+_{Tp,Au}/{image,mask}"

# --------------------------------------------------------------------------- #
# 3. MANUAL DOWNLOADS (need a browser / account)
# --------------------------------------------------------------------------- #
cat <<'EOF'

================ MANUAL DOWNLOADS REQUIRED ================
The following evaluation datasets need a browser and/or registration and
cannot be fetched by this script. Download each, then arrange under dataset/
following the layout in the README (dataset/<category>/<Name>_{Tp,Au}/{image,mask}).

  CASIAv2 (training, but also commonly used)  -> https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset  (Kaggle account)
  Fantastic Reality                         -> http://zefirus.org/MAG                                 (account login)
  IMD2020                                   -> http://zefirus.org/MAG                                 (account login)
  Columbia                                  -> https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp  (403 to scripts; use a browser)
  NIST16 (NIST MFC)                         -> https://mfc.nist.gov/                                  (registration required)
  DSO                                       -> https://recodbr.wordpress.com/code-n-data/#dso1_dsi1   (account)
  Korus                                     -> https://pkorus.pl/downloads/dataset-realistic-tampering
  DFFD: FFHQ + FaceAPP (DeepFake)            -> https://cvlab.cse.msu.edu/dffd-dataset.html            (data-use agreement + login)

After downloading, regenerate the question/labels files with:
  python playground/eval_jsonl.py --folders <tampered image dirs> --authentic-folders <auth dirs> \\
      --output ./playground/test.jsonl --labels-output ./playground/test_labels.jsonl
EOF

echo
echo "==== download_data.sh finished ===="
echo "Weights downloaded to : ${WEIGHT_DIR}"
echo "  -> FakeShield weights : ${WEIGHT_DIR}/fakeshield-v1-22b/{DTE-FDM,MFLM,DTG.pth}"
echo "  -> SAM weight         : ${WEIGHT_DIR}/sam_vit_h_4b8939.pth"
echo "Datasets downloaded to : ${DATA_DIR}"
echo
echo "When running the pipeline, point test.sh at these locations, e.g.:"
echo "  WEIGHT_PATH=${WEIGHT_DIR}/fakeshield-v1-22b bash scripts/test.sh"
echo "  (pass GT mask dirs under ${DATA_DIR} to evaluate_metrics.py --gt-dir)"
