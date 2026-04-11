from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .io_utils import ensure_dir, write_json


def prepare_failure_case_workspace(
    experiment_dir: str | Path,
    failure_case_name: str,
    keyframes: List[int],
    notes: str,
) -> Dict[str, str]:
    root = ensure_dir(Path(experiment_dir) / failure_case_name)
    paths = {
        "root": str(root),
        "keyframes": str(ensure_dir(root / "keyframes")),
        "masks": str(ensure_dir(root / "masks")),
        "edited": str(ensure_dir(root / "edited")),
        "comparisons": str(ensure_dir(root / "comparisons")),
    }
    write_json(
        {
            "failure_case": failure_case_name,
            "keyframes": keyframes,
            "notes": notes,
            "goal": "Repair frames where ProPainter cannot borrow missing background content.",
        },
        root / "part3_plan.json",
    )
    return paths
