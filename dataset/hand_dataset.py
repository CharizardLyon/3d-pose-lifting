import os
import json
import numpy as np
from torch.utils.data import Dataset


class HandPoseDataset(Dataset):
    def __init__(self, json_file, use_images=False, transform=None):
        """
        Args:
            json_file (str): path to train.json or val.json
            use_images (bool): whether to also load raw frames (jpg)
            transform (callable): optional transform for images
        """
        with open(json_file, "r") as f:
            self.samples = json.load(f)

        self.use_images = use_images
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load 2D keypoints
        kp2d = np.load(sample["kp2d_path"])  # (21, 2)

        # Load 3D keypoints
        kp3d = np.load(sample["kp3d_path"])  # (21, 3)

        output = {
            "kp2d": kp2d.astype(np.float32),
            "kp3d": kp3d.astype(np.float32),
            "session": sample["session"],
            "camera": sample["camera"],
            "frame": sample["frame"],
        }

        # Optionally load image
        if self.use_images:
            import cv2

            img = cv2.imread(sample["img_path"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            output["image"] = img

        return output
