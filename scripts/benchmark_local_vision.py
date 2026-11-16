"""Compare the local FabriCare scanner against a local vision-language model."""

import argparse
import base64
import csv
import io
import json
from pathlib import Path
from time import perf_counter

import requests
from PIL import Image, ImageOps


CLASSES = ("cotton", "denim", "linen", "polyester", "silk", "wool")
OLLAMA = "http://127.0.0.1:11434"
SCANNER = "http://127.0.0.1:5000"
PROMPT = (
    "Identify the visible fabric in this garment photo. Choose one of cotton, denim, "
    "linen, polyester, silk, wool, or uncertain. Choose uncertain when visual evidence "
    "is insufficient or the fabric appears to be outside these classes or a blend. "
    'Keep reasoning brief and return a JSON object with a single field, such as {"fabric":"uncertain"}.'
)


def image_bytes(path):
    with Image.open(path) as original:
        photo = ImageOps.exif_transpose(original).convert("RGB")
        photo.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        with io.BytesIO() as output:
            photo.save(output, format="JPEG", quality=92)
            return output.getvalue()


def benchmark_photo(session, model, path):
    photo = image_bytes(path)
    start = perf_counter()
    scanner = session.post(
        SCANNER + "/predict/hybrid",
        files={"image": ("garment.jpg", photo, "image/jpeg")},
        data={"model_type": "cnn", "ai_assist": "true", "allow_low_quality": "true"},
        timeout=120,
    )
    scanner.raise_for_status()
    scanner = scanner.json()
    scanner_seconds = perf_counter() - start
    start = perf_counter()
    vision = session.post(
        OLLAMA + "/api/chat",
        json={
            "model": model,
            "messages": [{
                "role": "user", "content": PROMPT,
                "images": [base64.b64encode(photo).decode("ascii")],
            }],
            "format": {
                "type": "object",
                "properties": {"fabric": {"type": "string", "enum": [*CLASSES, "uncertain"]}},
                "required": ["fabric"],
            },
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 1536},
        },
        timeout=180,
    )
    vision.raise_for_status()
    answer = vision.json()
    content = answer["message"]["content"]
    if not content.strip():
        raise ValueError(f"Vision model returned no final answer ({answer.get('done_reason')}).")
    label = json.loads(content)["fabric"]
    if label not in (*CLASSES, "uncertain"):
        raise ValueError(f"Unexpected vision model label: {label}")
    return {
        "scanner": scanner["fabric_type"],
        "scanner_needs_review": scanner["fabric_needs_confirmation"],
        "scanner_quality_passed": not scanner["quality_override"],
        "scanner_seconds": round(scanner_seconds, 2),
        "vision": label,
        "vision_seconds": round(perf_counter() - start, 2),
    }


def manifest_cases(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"garment_id", "photo", "label_photo", "fiber_label", "expected"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Manifest needs garment_id, photo, label_photo, fiber_label, expected.")
        rows = list(reader)
    if not rows:
        raise ValueError("Add at least one verified garment to the manifest.")
    ids, images, labels = set(), set(), set()
    for row in rows:
        garment = row["garment_id"].strip()
        photo = (path.parent / row["photo"]).resolve()
        label = (path.parent / row["label_photo"]).resolve()
        expected = row["expected"].strip().lower()
        if (not garment or garment in ids or photo in images or label in labels
                or not photo.is_file() or not label.is_file() or photo == label
                or not row["fiber_label"].strip() or expected not in CLASSES):
            raise ValueError(f"Invalid or duplicate label evidence for garment {garment!r}.")
        ids.add(garment)
        images.add(photo)
        labels.add(label)
        yield garment, photo, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="CSV of manually verified garment/label pairs")
    source.add_argument("--smoke-photo", type=Path, help="Check local inference without scoring accuracy")
    parser.add_argument("--model", default="qwen3-vl:2b")
    args = parser.parse_args()
    with requests.Session() as session:
        session.trust_env = False
        if args.smoke_photo:
            print(json.dumps(benchmark_photo(session, args.model, args.smoke_photo), indent=2))
            return
        cases = list(manifest_cases(args.manifest))
        results = []
        for garment, photo, expected in cases:
            result = {"garment": garment, "expected": expected,
                      **benchmark_photo(session, args.model, photo)}
            results.append(result)
            print(json.dumps(result), flush=True)
    scanner_correct = [r["scanner"] == r["expected"] for r in results]
    vision_correct = [r["vision"] == r["expected"] for r in results]
    summary = {
        "vision_model": args.model,
        "garments": len(results),
        "scanner_correct": sum(scanner_correct),
        "vision_correct": sum(vision_correct),
        "vision_uncertain": sum(r["vision"] == "uncertain" for r in results),
        "scanner_marked_for_review": sum(r["scanner_needs_review"] for r in results),
        "vision_corrected_scanner_errors": sum(v and not s for s, v in zip(scanner_correct, vision_correct)),
        "vision_missed_scanner_correct": sum(s and not v for s, v in zip(scanner_correct, vision_correct)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
