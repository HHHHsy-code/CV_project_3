from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _parse_scalar(value: str) -> Any:
    if value in {"null", "Null", "~"}:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def _normalize_yaml_lines(text: str) -> List[Tuple[int, str]]:
    normalized: List[Tuple[int, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        if raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        normalized.append((indent, raw_line.strip()))
    return normalized


def _parse_yaml_block(lines: List[Tuple[int, str]], start: int, indent: int) -> Tuple[Any, int]:
    if start >= len(lines):
        return {}, start

    is_list = lines[start][1].startswith("- ")
    if is_list:
        items = []
        idx = start
        while idx < len(lines):
            current_indent, content = lines[idx]
            if current_indent < indent or not content.startswith("- "):
                break
            if current_indent != indent:
                raise ValueError(f"Unexpected indentation in YAML list item: {content}")

            value_text = content[2:].strip()
            idx += 1
            if value_text:
                items.append(_parse_scalar(value_text))
            else:
                value, idx = _parse_yaml_block(lines, idx, indent + 2)
                items.append(value)
        return items, idx

    mapping: Dict[str, Any] = {}
    idx = start
    while idx < len(lines):
        current_indent, content = lines[idx]
        if current_indent < indent:
            break
        if current_indent != indent:
            raise ValueError(f"Unexpected indentation in YAML mapping: {content}")

        key, sep, raw_value = content.partition(":")
        if not sep:
            raise ValueError(f"Invalid YAML line: {content}")

        key = key.strip()
        value_text = raw_value.strip()
        idx += 1
        if value_text:
            mapping[key] = _parse_scalar(value_text)
        else:
            value, idx = _parse_yaml_block(lines, idx, indent + 2)
            mapping[key] = value
    return mapping, idx


def _load_simple_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lines = _normalize_yaml_lines(text)
    if not lines:
        return {}
    parsed, _ = _parse_yaml_block(lines, 0, lines[0][0])
    if not isinstance(parsed, dict):
        raise ValueError("Top-level YAML content must be a mapping.")
    return parsed


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if config_path.suffix.lower() == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))

    try:
        import yaml
    except ModuleNotFoundError:
        return _load_simple_yaml(config_path)

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
