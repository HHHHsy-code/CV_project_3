from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .io_utils import ensure_dir, write_json


class Part2Adapters:
    """Store external-tool metadata and standardize experiment directories."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config["part2"]

    def prepare_experiment(self, experiment_dir: str | Path, prompt_boxes: List[dict] | None = None) -> Dict[str, str]:
        root = ensure_dir(experiment_dir)
        paths = {
            "root": str(root),
            "masks": str(ensure_dir(root / "masks")),
            "frames": str(ensure_dir(root / "frames")),
            "propainter_input": str(ensure_dir(root / "propainter_input")),
            "propainter_output": str(ensure_dir(root / "propainter_output")),
        }
        metadata = {
            "sam2_repo": self.config["sam2"]["repo_dir"],
            "sam2_checkpoint": self.config["sam2"]["checkpoint"],
            "sam2_config": self.config["sam2"]["config"],
            "propainter_repo": self.config["propainter"]["repo_dir"],
            "propainter_checkpoint_dir": self.config["propainter"]["checkpoint_dir"],
            "prompt_boxes": prompt_boxes or [],
        }
        write_json(metadata, root / "part2_metadata.json")
        return paths

    def recommended_commands(self, video_path: str | Path, experiment_dir: str | Path) -> Dict[str, str]:
        video_path = str(video_path)
        root = Path(experiment_dir)
        frames_dir = root / "frames"
        masks_dir = root / "masks"
        return {
            "sam2": (
                f"cd {self.config['sam2']['repo_dir']} && "
                f"python demo_video.py --video_path {video_path} "
                f"--output_masks {masks_dir} --output_frames {frames_dir}"
            ),
            "propainter": (
                f"cd {self.config['propainter']['repo_dir']} && "
                f"python inference_propainter.py --video {video_path} "
                f"--mask_dir {masks_dir} --save_dir {root / 'propainter_output'}"
            ),
        }
