"""Episode writer for LeRobot v3 datasets (the format v1 recorded and SmolVLA trained on), with a plain
fallback (npz + json per episode) wherever lerobot is not importable, so the recording pipeline is tested
in the rig without it. ROS-free.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")      # local datasets only: never ask the Hub (v1 run.sh did the same)
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from .schema import ACTION_DIM, CAMERA_KEYS, STATE_DIM, features


def lerobot_available() -> bool:
    try:
        import lerobot  # noqa: F401
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


class EpisodeWriter:
    """add_frame(obs images, state, action, task) ... save_episode(); the dataset grows under root/repo_id."""

    def __init__(self, root: Path | str, repo_id: str, fps: int = 12, width: int = 960, height: int = 540,
                 cameras=CAMERA_KEYS, use_videos: bool = True, backend: str = "auto"):
        self.root, self.repo_id, self.fps = Path(root), repo_id, int(fps)
        self.width, self.height, self.cameras = int(width), int(height), tuple(cameras)
        self.use_videos = use_videos
        self.backend = backend if backend != "auto" else ("lerobot" if lerobot_available() else "npz")
        self.frames = 0
        self.episodes = 0
        self._buf: list[dict] = []
        self.dataset = None
        if self.backend == "lerobot":
            self._open_lerobot()

    @property
    def path(self) -> Path:
        return self.root / self.repo_id

    # ---- lerobot
    def _open_lerobot(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        if (self.path / "meta").is_dir():
            self.dataset = LeRobotDataset(repo_id=self.repo_id, root=str(self.path))
            self.fps = self.dataset.fps
            img = self.dataset.meta.features.get(f"observation.images.{self.cameras[0]}")
            if img and len(img["shape"]) == 3:
                self.height, self.width = int(img["shape"][0]), int(img["shape"][1])
        else:
            self.dataset = LeRobotDataset.create(repo_id=self.repo_id, root=str(self.path), fps=self.fps,
                                                 features=features(self.width, self.height, self.cameras, self.use_videos),
                                                 robot_type="ur5e_amazinghand", use_videos=self.use_videos)
        self.episodes = int(self.dataset.meta.total_episodes)

    def _fit(self, img):
        import cv2
        if img.shape[0] != self.height or img.shape[1] != self.width:
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(img, dtype=np.uint8)

    def add_frame(self, images: dict, state, action, task: str) -> None:
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if state.shape != (STATE_DIM,) or action.shape != (ACTION_DIM,):
            raise ValueError(f"state {state.shape} / action {action.shape}")
        frame = {"observation.state": state, "action": action, "task": task}
        for cam in self.cameras:
            if cam not in images:
                raise KeyError(f"missing camera {cam}")
            frame[f"observation.images.{cam}"] = self._fit(images[cam])
        if self.backend == "lerobot":
            self.dataset.add_frame(frame)
        else:
            self._buf.append(frame)
        self.frames += 1

    def save_episode(self) -> dict:
        if self.backend == "lerobot":
            n = self.dataset.meta.total_episodes
            self.dataset.save_episode()
            self.episodes = int(self.dataset.meta.total_episodes)
            return {"episode": n, "frames": self.frames, "backend": "lerobot", "path": str(self.path)}
        ep = self.episodes
        d = self.path / "episodes"
        d.mkdir(parents=True, exist_ok=True)
        if self._buf:
            arrays = {"observation.state": np.stack([f["observation.state"] for f in self._buf]),
                      "action": np.stack([f["action"] for f in self._buf])}
            for cam in self.cameras:
                arrays[f"observation.images.{cam}"] = np.stack([f[f"observation.images.{cam}"] for f in self._buf])
            np.savez_compressed(d / f"episode_{ep:06d}.npz", **arrays)
            (d / f"episode_{ep:06d}.json").write_text(json.dumps({"task": self._buf[0]["task"], "frames": len(self._buf),
                                                                   "fps": self.fps, "saved": time.time()}))
        meta = {"repo_id": self.repo_id, "fps": self.fps, "episodes": ep + 1, "backend": "npz",
                "features": features(self.width, self.height, self.cameras, self.use_videos)}
        (self.path / "meta.json").write_text(json.dumps(meta, indent=1))
        self._buf.clear()
        self.episodes = ep + 1
        return {"episode": ep, "frames": self.frames, "backend": "npz", "path": str(self.path)}

    def discard_episode(self) -> None:
        if self.backend == "lerobot":
            self.dataset.clear_episode_buffer()
        else:
            self._buf.clear()
        self.frames = 0


def load_episode_npz(path: Path | str) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
