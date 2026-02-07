#!/usr/bin/env python3
"""Download QuickDraw simplified drawings (.ndjson) for selected categories.

Streams large .ndjson files line-by-line from Google Cloud Storage,
filters to recognized drawings only, and saves a limited number per category.
"""

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlopen, Request

DEFAULT_CATEGORIES = [
    "cat", "car", "house", "tree", "flower",
    "fish", "bicycle", "guitar", "airplane", "face",
]
URL_TEMPLATE = (
    "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{category}.ndjson"
)
DEFAULT_MAX_PER_CATEGORY = 5000
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "quickdraw"


def download_category(category: str, max_samples: int, output_dir: Path) -> int:
    """Stream-download a single category, keeping up to *max_samples* recognized drawings."""
    url = URL_TEMPLATE.format(category=category)
    out_path = output_dir / f"{category}.ndjson"

    print(f"[{category}] Downloading from {url} ...")
    req = Request(url, headers={"User-Agent": "DriftSketch/0.1"})

    kept = 0
    seen = 0
    try:
        with urlopen(req) as resp, open(out_path, "w", encoding="utf-8") as fout:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                seen += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not record.get("recognized", False):
                    continue
                fout.write(json.dumps(record, separators=(",", ":")) + "\n")
                kept += 1
                if kept % 500 == 0:
                    print(f"  [{category}] kept {kept}/{max_samples}  (scanned {seen} lines)")
                if kept >= max_samples:
                    break
    except Exception as exc:
        print(f"  [{category}] ERROR: {exc}", file=sys.stderr)
        return kept

    print(f"  [{category}] Done -- kept {kept} drawings (scanned {seen} lines)")
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Download QuickDraw simplified drawings.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="Categories to download (default: %(default)s)",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=DEFAULT_MAX_PER_CATEGORY,
        help="Max recognized drawings per category (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: %(default)s)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for cat in args.categories:
        n = download_category(cat, args.max_per_category, args.output_dir)
        total += n

    print(f"\nAll done. {total} drawings saved to {args.output_dir}")


if __name__ == "__main__":
    main()
