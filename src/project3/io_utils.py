from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def list_images(path: str | Path) -> List[Path]:
    directory = Path(path)
    return sorted(
        item
        for item in directory.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def write_json(data: dict, path: str | Path) -> None:
    output = Path(path)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def write_lines(lines: Iterable[str], path: str | Path) -> None:
    output = Path(path)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")
