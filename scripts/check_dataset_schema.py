#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


REQUIRED_ITEM_KEYS = ["file_path", "video_caption_path"]
REQUIRED_CAPTION_KEYS = ["audiovisual_caption", "audio_content", "max_start_end_time"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def short(v, n=160):
    s = str(v).replace("\n", " ")
    return s[:n] + ("..." if len(s) > n else "")


def check_one(meta_path, sample_size=100, seed=42):
    meta_path = Path(meta_path)
    data = load_json(meta_path)

    if isinstance(data, dict):
        items = list(data.values())
    elif isinstance(data, list):
        items = data
    else:
        print(f"[BAD] {meta_path}: top-level must be list/dict, got {type(data).__name__}")
        return

    total = len(items)
    rng = random.Random(seed)
    indices = list(range(total)) if total <= sample_size else rng.sample(range(total), sample_size)

    item_missing = {k: 0 for k in REQUIRED_ITEM_KEYS}
    caption_missing = {k: 0 for k in REQUIRED_CAPTION_KEYS}
    caption_open_fail = 0
    valid = 0
    example_bad = None
    example_good = None

    for idx in indices:
        item = items[idx]
        if not isinstance(item, dict):
            if example_bad is None:
                example_bad = (idx, "item is not dict", item)
            continue

        miss_item = [k for k in REQUIRED_ITEM_KEYS if k not in item]
        for k in miss_item:
            item_missing[k] += 1

        if miss_item:
            if example_bad is None:
                example_bad = (idx, f"missing item keys {miss_item}", item)
            continue

        cap_path = Path(item["video_caption_path"])
        if not cap_path.exists():
            caption_open_fail += 1
            if example_bad is None:
                example_bad = (idx, f"caption file not found: {cap_path}", item)
            continue

        try:
            cap = load_json(cap_path)
        except Exception as e:
            caption_open_fail += 1
            if example_bad is None:
                example_bad = (idx, f"caption load error: {type(e).__name__}: {e}", item)
            continue

        miss_cap = [k for k in REQUIRED_CAPTION_KEYS if k not in cap]
        for k in miss_cap:
            caption_missing[k] += 1

        if miss_cap:
            if example_bad is None:
                example_bad = (idx, f"missing caption keys {miss_cap}", {
                    "item": item,
                    "caption_keys": sorted(cap.keys()),
                })
            continue

        valid += 1
        if example_good is None:
            example_good = (idx, item, cap)

    print(f"\n=== {meta_path} ===")
    print(f"total items: {total}")
    print(f"sampled: {len(indices)}")
    print(f"valid for current dataloader: {valid}/{len(indices)} ({valid / max(1, len(indices)):.1%})")
    print(f"missing item keys: {item_missing}")
    print(f"caption open/load failed: {caption_open_fail}")
    print(f"missing caption keys: {caption_missing}")

    if example_good:
        idx, item, cap = example_good
        print(f"\n[GOOD example idx={idx}]")
        print("file_path:", item.get("file_path"))
        print("video_caption_path:", item.get("video_caption_path"))
        print("audiovisual_caption:", short(cap.get("audiovisual_caption")))
        print("max_start_end_time:", cap.get("max_start_end_time"))

    if example_bad:
        idx, reason, obj = example_bad
        print(f"\n[BAD example idx={idx}] {reason}")
        print(short(obj, 800))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="+", help="meta json files to check")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for path in args.json_files:
        check_one(path, args.sample_size, args.seed)


if __name__ == "__main__":
    main()