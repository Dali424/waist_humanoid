#!/usr/bin/env python3
import argparse
import json
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count success/fail results for episode_* folders in a directory.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Directory containing episode_* subfolders (default: current directory)",
    )
    return parser.parse_args()


def iter_episode_dirs(root: str):
    for name in sorted(os.listdir(root)):
        if name.startswith("episode_"):
            full = os.path.join(root, name)
            if os.path.isdir(full):
                yield full


def read_result(data_path: str):
    try:
        with open(data_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return "missing", "missing data.json"
    except json.JSONDecodeError:
        return "invalid", "invalid JSON"

    raw = data.get("result")
    normalized = ""
    if isinstance(raw, str):
        normalized = raw.strip().lower()
    elif raw is not None:
        normalized = str(raw).strip().lower()

    if normalized in ("success", "fail"):
        return normalized, None
    return "unknown", None


def main() -> int:
    args = parse_args()
    target = os.path.abspath(args.path)

    if not os.path.isdir(target):
        print(f"Target is not a directory: {target}", file=sys.stderr)
        return 1

    counts = {"success": 0, "fail": 0, "unknown": 0, "missing": 0, "invalid": 0}
    issues = []

    episodes = list(iter_episode_dirs(target))
    for ep in episodes:
        result, msg = read_result(os.path.join(ep, "data.json"))
        if result in counts:
            counts[result] += 1
        else:
            counts["unknown"] += 1
        if msg:
            issues.append((os.path.basename(ep), msg))

    print(f"Folder: {target}")
    print(f"Episodes found: {len(episodes)}")
    print(f"success: {counts['success']}")
    print(f"fail   : {counts['fail']}")
    print(f"unknown: {counts['unknown']}")
    if counts["missing"] or counts["invalid"]:
        print(f"missing data.json: {counts['missing']}")
        print(f"invalid data.json: {counts['invalid']}")

    if issues:
        print("\nIssues:")
        for ep, msg in issues:
            print(f" - {ep}: {msg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
