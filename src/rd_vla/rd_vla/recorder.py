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
    """add_frame(obs images, state, action, task) ... save_episode(); the dataset grows under root/repo_id.

    Image sizes: `width` x `height` given -> every camera is fitted to it (the rig's tests). Left None -> each camera
    keeps the size of its first frame, as v1's sets do (scene and side 540x540 squares, the wrist 540x960), so a
    square camera is never stretched into a landscape box. A dataset that already exists dictates its own shapes.
    """

    def __init__(self, root: Path | str, repo_id: str, fps: int = 12, width: int | None = None, height: int | None = None,
                 cameras=CAMERA_KEYS, use_videos: bool = True, backend: str = "auto"):
        self.root, self.repo_id, self.fps = Path(root), repo_id, int(fps)
        self.cameras = tuple(cameras)
        self.shapes: dict[str, tuple[int, int]] = {}      # camera -> (height, width), fixed once known
        if width and height:
            self.shapes = {cam: (int(height), int(width)) for cam in self.cameras}
        self.use_videos = use_videos
        self.backend = backend if backend != "auto" else ("lerobot" if lerobot_available() else "npz")
        self.frames = 0
        self.episodes = 0
        self._buf: list[dict] = []
        self.dataset = None
        if self.backend == "lerobot" and self.path.exists() and not (self.path / "meta").is_dir():
            # something else sits at this name (an npz recording from a python without lerobot): LeRobot will not
            # create over it, so record beside it
            n = 2
            while (self.root / f"{self.repo_id}_{n}").exists():
                n += 1
            print(f"[recorder] {self.path} exists and is not a LeRobot dataset: recording to {self.repo_id}_{n}", flush=True)
            self.repo_id = f"{self.repo_id}_{n}"
        if self.backend == "lerobot" and ((self.path / "meta").is_dir() or len(self.shapes) == len(self.cameras)):
            self._open_lerobot()       # else created at the first frame, once every camera's size is known

    @property
    def path(self) -> Path:
        return self.root / self.repo_id

    # ---- lerobot
    def _features(self) -> dict:
        return features(0, 0, self.cameras, self.use_videos, shapes=self.shapes)

    def _open_lerobot(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        if (self.path / "meta").is_dir():
            self.dataset = LeRobotDataset(repo_id=self.repo_id, root=str(self.path))
            self.fps = self.dataset.fps
            for cam in self.cameras:
                img = self.dataset.meta.features.get(f"observation.images.{cam}")
                if img and len(img["shape"]) == 3:
                    self.shapes[cam] = (int(img["shape"][0]), int(img["shape"][1]))
        else:
            self.dataset = LeRobotDataset.create(repo_id=self.repo_id, root=str(self.path), fps=self.fps,
                                                 features=self._features(), robot_type="ur5e_amazinghand", use_videos=self.use_videos)
        self.episodes = int(self.dataset.meta.total_episodes)

    def _fit(self, cam: str, img):
        h, w = self.shapes.setdefault(cam, (int(img.shape[0]), int(img.shape[1])))     # the first frame fixes an unknown size
        if img.shape[0] != h or img.shape[1] != w:
            import cv2
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
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
            frame[f"observation.images.{cam}"] = self._fit(cam, images[cam])
        if self.backend == "lerobot":
            if self.dataset is None:
                self._open_lerobot()
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
        meta = {"repo_id": self.repo_id, "fps": self.fps, "episodes": ep + 1, "backend": "npz", "features": self._features()}
        (self.path / "meta.json").write_text(json.dumps(meta, indent=1))
        self._buf.clear()
        self.episodes = ep + 1
        return {"episode": ep, "frames": self.frames, "backend": "npz", "path": str(self.path)}

    def close(self) -> None:
        """Finish writing (LeRobot's finalize closes the parquet writers; without it the dataset cannot be re-opened)."""
        if self.backend == "lerobot" and self.dataset is not None:
            try:
                self.dataset.finalize()
            except Exception as e:  # noqa: BLE001
                print(f"[recorder] finalize failed: {e}", flush=True)
            self.dataset = None

    def discard_episode(self) -> None:
        if self.backend == "lerobot":
            if self.dataset is not None:
                self.dataset.clear_episode_buffer()
        else:
            self._buf.clear()
        self.frames = 0


def load_episode_npz(path: Path | str) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
