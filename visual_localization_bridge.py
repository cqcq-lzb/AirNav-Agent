"""Bridge between the navigation UI and the patient-specific PoseMap model."""

from __future__ import annotations

import sys
from pathlib import Path


class VisualLocalizationBridge:
    def __init__(self, model_path: Path, case_id: str) -> None:
        expected = model_path.parent.name.split("_mapclassifier", 1)[0]
        if expected and expected != case_id:
            raise ValueError(f"模型属于 {expected}，当前病例是 {case_id}")
        code_dir = model_path.parents[2] / "code"
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        from posemap_inference import PoseMapTracker
        self.tracker = PoseMapTracker(
            model_path, device="auto", max_jump_mm=15.0,
            top_k=10, route_gate_mm=12.0,
        )

    def set_route(self, points_xyz_mm) -> None:
        self.tracker.set_route(points_xyz_mm)
        if len(points_xyz_mm) and self.tracker.route_start_class_id is not None:
            self.tracker.reset(
                self.tracker.class_positions[self.tracker.route_start_class_id]
            )
        else:
            self.tracker.reset(None)

    @property
    def route_start_class_id(self) -> int | None:
        return self.tracker.route_start_class_id

    @property
    def route_start_offset_mm(self) -> float | None:
        return self.tracker.route_start_offset_mm

    def reset(self, position_xyz_mm=None) -> None:
        if self.tracker.route_start_class_id is not None:
            position_xyz_mm = self.tracker.class_positions[self.tracker.route_start_class_id]
        self.tracker.reset(position_xyz_mm)

    def update(self, rgb_frame):
        return self.tracker.update(rgb_frame)
