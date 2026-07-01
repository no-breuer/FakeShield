"""Generate the JSONL question file consumed by `scripts/test.sh`.

The original version of this script had hardcoded paths to the authors' server
(`/data03/xzp/dataset/...`). This version takes everything from the command line
so it can be run anywhere (e.g. on a GPU cluster).

Two files are produced:
  * <output>          : one `{image, text}` record per line, the QUESTION_PATH
                        consumed by DTE-FDM (`model_vqa.py`).
  * <labels-output>   : one `{image, tampered}` record per line, used by
                        `evaluate_metrics.py` to score DETECTION accuracy.
                        `tampered` is 1 for images from `--folders` and 0 for
                        images from `--authentic-folders`.

Example
-------
python playground/eval_jsonl.py \
    --folders /scratch/dataset/photoshop/CASIAv1+_Tp/image \
              /scratch/dataset/deepfake/FaceAPP_Val/image \
    --authentic-folders /scratch/dataset/photoshop/CASIAv1+_Au/image \
    --output ./playground/test.jsonl \
    --labels-output ./playground/test_labels.jsonl

NOTE: `model_vqa.py` is invoked with `--image-folder /`, so the `image` field
MUST be an absolute path. This script therefore resolves every image to an
absolute path regardless of how the folders are given.
"""
import argparse
import json
import os

# Same detection prompt the original script used.
DEFAULT_QUESTION = (
    "Was this photo taken directly from the camera without any processing? "
    "Has it been tampered with by any artificial photo modification techniques "
    "such as ps? Please zoom in on any details in the image, paying special "
    "attention to the edges of the objects, capturing some unnatural edges and "
    "perspective relationships, some incorrect semantics, unnatural lighting and "
    "darkness etc."
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def list_images(folder):
    out = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(IMAGE_EXTS):
            out.append(os.path.abspath(os.path.join(folder, name)))
    return out


def generate_json(tampered_folders, authentic_folders, output_file, labels_file,
                  question, num_images_per_folder):
    records = []
    labels = []

    def add(folder, tampered):
        imgs = list_images(folder)
        if num_images_per_folder is not None:
            imgs = imgs[:num_images_per_folder]
        for path in imgs:
            records.append({"image": path, "text": question})
            labels.append({"image": path, "tampered": 1 if tampered else 0})

    for folder in tampered_folders:
        add(folder, tampered=True)
    for folder in authentic_folders:
        add(folder, tampered=False)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        for item in records:
            f.write(json.dumps(item) + "\n")

    if labels_file:
        os.makedirs(os.path.dirname(os.path.abspath(labels_file)) or ".", exist_ok=True)
        with open(labels_file, "w") as f:
            for item in labels:
                f.write(json.dumps(item) + "\n")

    n_t = sum(1 for x in labels if x["tampered"])
    n_a = len(labels) - n_t
    print(f"Wrote {len(records)} questions -> {output_file}")
    print(f"  tampered: {n_t}   authentic: {n_a}")
    if labels_file:
        print(f"Wrote {len(labels)} labels   -> {labels_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the DTE-FDM question JSONL (and optional labels JSONL)."
    )
    parser.add_argument("--folders", nargs="+", required=True,
                        help="Folders of TAMPERED test images.")
    parser.add_argument("--authentic-folders", nargs="+", default=[],
                        help="Folders of AUTHENTIC test images (for detection eval).")
    parser.add_argument("--output", default="./playground/test.jsonl",
                        help="Output question JSONL path (QUESTION_PATH).")
    parser.add_argument("--labels-output", default="./playground/test_labels.jsonl",
                        help="Output labels JSONL path (for evaluate_metrics.py). "
                             "Pass an empty string to disable.")
    parser.add_argument("--num-images", type=int, default=None,
                        help="Cap images per folder (default: all).")
    parser.add_argument("--question", default=DEFAULT_QUESTION,
                        help="Detection question text.")
    args = parser.parse_args()

    generate_json(
        tampered_folders=args.folders,
        authentic_folders=args.authentic_folders,
        output_file=args.output,
        labels_file=args.labels_output if args.labels_output else None,
        question=args.question,
        num_images_per_folder=args.num_images,
    )
