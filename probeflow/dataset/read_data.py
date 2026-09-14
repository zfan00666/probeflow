import numpy as np
import cv2
from pathlib import Path

class EpisodeReader:
    def __init__(self, episode_dir: str | Path, cam_dir_name: str = "camera"):
        self.episode_dir = Path(episode_dir)
        self.cam_dir = self.episode_dir / cam_dir_name

        self.pose_data = np.loadtxt(self.episode_dir / "pose.txt")
        self.joint_data = np.loadtxt(self.episode_dir / "joint.txt")
        self.force_data = np.loadtxt(self.episode_dir / "force.txt")

        self.timestamps = self.pose_data[:, 0].astype(np.int64)

    def __len__(self):
        return len(self.timestamps)

    def get_step_data(self, idx: int, rgb_suffix: str = "jpg", depth_suffix: str = "npy"):
        ts = self.timestamps[idx]

        pose = self.pose_data[idx, 1:]
        joint = self.joint_data[idx, 1:]
        force = self.force_data[idx, 1:]

        rgb_path = self.cam_dir / f"rgb_{ts}.{rgb_suffix}"
        if not rgb_path.exists():
            print(f"DEBUG Error: missing image at {rgb_path}")
        depth_path = self.cam_dir / f"d_{ts}.{depth_suffix}"

        rgb = cv2.imread(str(rgb_path)) if rgb_path.exists() else None
        if rgb is not None:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = np.load(str(depth_path)) if depth_path.exists() else None

        return {
            "timestamp": ts,
            "pose": pose,
            "joint": joint,
            "force": force,
            "rgb": rgb,
            "depth": depth
        }

class TaskReader:
    def __init__(self, task_dir: str | Path, cam_dir_name: str = "camera"):
        self.task_dir = Path(task_dir)
        self.episodes = []

        ep_dirs = sorted([d for d in self.task_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])

        for ep_dir in ep_dirs:
            self.episodes.append(EpisodeReader(ep_dir, cam_dir_name))

    def __len__(self):
        return len(self.episodes)

    def get_episode(self, idx: int) -> EpisodeReader:
        return self.episodes[idx]

    def get_all_steps(self):
        for ep_idx, ep in enumerate(self.episodes):
            for step_idx in range(len(ep)):
                yield ep_idx, step_idx, ep.get_step_data(step_idx)

def compute_task_stats(task_reader: TaskReader):
    all_poses = []
    for ep_idx in range(len(task_reader)):
        ep = task_reader.get_episode(ep_idx)
        all_poses.append(ep.pose_data[:, 1:]) 

    all_poses = np.concatenate(all_poses, axis=0)
    return {
        "pose_min": all_poses.min(axis=0),
        "pose_max": all_poses.max(axis=0)
    }

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

class EvoRealDataset(Dataset):
    def __init__(self, task_reader, norm_stats, prompt, horizon=16, max_state_dim=24, max_action_dim=24, max_views=3, embodiment_id=0):
        self.reader = task_reader
        self.prompt = prompt
        self.horizon = horizon
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.max_views = max_views
        self.embodiment_id = embodiment_id

        self.state_min = norm_stats["pose_min"]
        self.state_max = norm_stats["pose_max"]
        self.action_min = norm_stats["pose_min"]
        self.action_max = norm_stats["pose_max"]

        self.arm2stats_dict = {
            "real_robot": {
                "observation.state": {"min": self.state_min.tolist(), "max": self.state_max.tolist()},
                "action": {"min": self.action_min.tolist(), "max": self.action_max.tolist()}
            }
        }

        self.index_map = []
        for ep_idx in range(len(self.reader)):
            ep = self.reader.get_episode(ep_idx)
            for step_idx in range(len(ep) - self.horizon + 1):
                self.index_map.append((ep_idx, step_idx))

        self.transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.index_map)

    def _pad_tensor(self, source_tensor: torch.Tensor, max_dim: int):
        source_dim = source_tensor.shape[-1]
        padded_shape = (*source_tensor.shape[:-1], max_dim)
        padded_tensor = torch.zeros(padded_shape, dtype=source_tensor.dtype)
        mask = torch.zeros(padded_shape, dtype=torch.bool)
        data_slice = (..., slice(0, source_dim))
        padded_tensor[data_slice] = source_tensor
        mask[data_slice] = True
        return padded_tensor, mask

    def __getitem__(self, idx):
        ep_idx, step_idx = self.index_map[idx]
        episode = self.reader.get_episode(ep_idx)
        step_data = episode.get_step_data(step_idx)

        images = []
        if step_data["rgb"] is not None:
            img_pil = Image.fromarray(step_data["rgb"])
            images.append(self.transform(img_pil))

        image_mask = torch.zeros(self.max_views, dtype=torch.bool)
        image_mask[:len(images)] = True
        while len(images) < self.max_views:
            images.append(torch.zeros(3, 448, 448))
        images = torch.stack(images)

        raw_state = step_data["pose"] 
        state = torch.tensor(raw_state, dtype=torch.float32)
        s_min = torch.tensor(self.state_min, dtype=torch.float32)
        s_max = torch.tensor(self.state_max, dtype=torch.float32)
        state = 2 * (state - s_min) / (s_max - s_min + 1e-8) - 1
        state = torch.clamp(state, -1.0, 1.0)
        state_padded, state_mask = self._pad_tensor(state, self.max_state_dim)

        action_seq = []
        for t in range(step_idx, step_idx + self.horizon):
            future_step = episode.get_step_data(t)
            action_seq.append(future_step["pose"])

        action = torch.tensor(np.stack(action_seq), dtype=torch.float32)
        action = 2 * (action - s_min.unsqueeze(0)) / (s_max.unsqueeze(0) - s_min.unsqueeze(0) + 1e-8) - 1
        action = torch.clamp(action, -1.0, 1.0)
        action_padded, action_mask = self._pad_tensor(action, self.max_action_dim)

        return {
            "prompt": self.prompt,
            "images": images,
            "image_mask": image_mask,
            "state": state_padded.to(dtype=torch.bfloat16),
            "state_mask": state_mask,
            "action": action_padded.to(dtype=torch.bfloat16),
            "action_mask": action_mask,
            "embodiment_id": torch.tensor(self.embodiment_id, dtype=torch.long)
        }

if __name__ == "__main__":
    task_reader = TaskReader("/mnt/data_ssd/zhoufang/code/probeflow/Evo_1/dataset/real_data/task_1")
    print(f"Total episodes found: {len(task_reader)}")

    for ep_idx, step_idx, step_data in task_reader.get_all_steps():
        print(f"Ep: {ep_idx}, Step: {step_idx}, Pose shape: {step_data['pose'].shape}")
        break
