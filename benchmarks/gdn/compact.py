"""Export small auditable metrics while keeping individual event traces remotely."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="docs/gdn_data")
    args = parser.parse_args()
    for path in Path(args.root).glob("*/*.jsonl"):
        rows = []
        for line in path.open():
            row = json.loads(line)
            for key in ("profile", "cache_profile"):
                profile = row.get(key)
                if not profile:
                    continue
                grouped = defaultdict(lambda: {"ms": 0.0, "count": 0})
                for event in profile.pop("events"):
                    value = grouped[
                        (event["module"], event["phase"], event["update_cache"])
                    ]
                    value["ms"] += event["ms"]
                    value["count"] += 1
                profile["events"] = [
                    dict(module=k[0], phase=k[1], update_cache=k[2], **v)
                    for k, v in grouped.items()
                ]
            rows.append(row)
        path.with_suffix(".compact.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
