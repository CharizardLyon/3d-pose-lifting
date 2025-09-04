import os
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = output_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        frames.append(out_path)
        frame_idx += 1
    cap.release()
    return frames


def load_2d_from_csv(csv_path):
    """Load 2D keypoints (frames, 21, 2) from Mediapipe CSV"""
    df = pd.read_csv(csv_path, header=[0, 1, 2])

    joints = [
        "WRIST",
        "THUMB_CMC",
        "THUMB_MCP",
        "THUMB_IP",
        "THUMB_TIP",
        "INDEX_FINGER_MCP",
        "INDEX_FINGER_PIP",
        "INDEX_FINGER_DIP",
        "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP",
        "MIDDLE_FINGER_PIP",
        "MIDDLE_FINGER_DIP",
        "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP",
        "RING_FINGER_PIP",
        "RING_FINGER_DIP",
        "RING_FINGER_TIP",
        "PINKY_MCP",
        "PINKY_PIP",
        "PINKY_DIP",
        "PINKY_TIP",
    ]

    values = []
    for j in joints:
        x = df[("Mediapipe", j, "x")].to_numpy(dtype=np.float32)
        y = df[("Mediapipe", j, "y")].to_numpy(dtype=np.float32)
        values.append(np.stack([x, y], axis=-1))

    return np.stack(values, axis=1)


def load_3d_from_csv(csv_path):
    """Load 3D keypoints (frames, 21, 3) from DLC-style CSV"""
    df = pd.read_csv(csv_path)

    joints = [
        "WRIST",
        "THUMB_CMC",
        "THUMB_MCP",
        "THUMB_IP",
        "THUMB_TIP",
        "INDEX_FINGER_MCP",
        "INDEX_FINGER_PIP",
        "INDEX_FINGER_DIP",
        "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP",
        "MIDDLE_FINGER_PIP",
        "MIDDLE_FINGER_DIP",
        "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP",
        "RING_FINGER_PIP",
        "RING_FINGER_DIP",
        "RING_FINGER_TIP",
        "PINKY_MCP",
        "PINKY_PIP",
        "PINKY_DIP",
        "PINKY_TIP",
    ]

    values = []
    for j in joints:
        x = df[f"{j}_x"].to_numpy(dtype=np.float32)
        y = df[f"{j}_y"].to_numpy(dtype=np.float32)
        z = df[f"{j}_z"].to_numpy(dtype=np.float32)
        values.append(np.stack([x, y, z], axis=-1))

    return np.stack(values, axis=1)  # (frames, joints, 3)


def build_session(session_dir, out_dir, index_train, index_val):
    session_dir = Path(session_dir)
    session_name = session_dir.name
    out_dir = Path(out_dir)

    # Load 3D ground truth
    kp3d_all = load_3d_from_csv(session_dir / "pose-3d" / "pose-3d.csv")

    # Process each camera
    for cam_file in sorted((session_dir / "videos-raw").glob("*.mp4")):
        cam_name = cam_file.stem  # camA, camB...
        print(f"Processing {session_name} - {cam_name}")

        # Load 2D keypoints
        csv_path = session_dir / "pose-2d" / f"{cam_name}.csv"
        kp2d_all = load_2d_from_csv(csv_path)

        # Output dirs
        img_dir = out_dir / "images" / session_name / cam_name
        kp2d_dir = out_dir / "keypoints_2d" / session_name / cam_name
        kp3d_dir = out_dir / "keypoints_3d" / session_name
        img_dir.mkdir(parents=True, exist_ok=True)
        kp2d_dir.mkdir(parents=True, exist_ok=True)
        kp3d_dir.mkdir(parents=True, exist_ok=True)

        # Extract frames
        frames = extract_frames(cam_file, img_dir)

        assert (
            len(frames) == len(kp2d_all) == len(kp3d_all)
        ), f"Mismatch in {session_name}-{cam_name}"

        # Split indices (train/val)
        indices = np.arange(len(frames))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

        # Save annotations
        for i, frame_path in enumerate(frames):
            # Save 2D
            np.save(kp2d_dir / f"{frame_path.stem}.npy", kp2d_all[i])

            # Save 3D once (only from camA)
            if cam_name == "camA":
                np.save(kp3d_dir / f"{frame_path.stem}.npy", kp3d_all[i])

            # Add to split index
            entry = {
                "session": session_name,
                "camera": cam_name,
                "frame": frame_path.stem,
                "img_path": str(frame_path),
                "kp2d_path": str(kp2d_dir / f"{frame_path.stem}.npy"),
                "kp3d_path": str(kp3d_dir / f"{frame_path.stem}.npy"),
            }
            if i in train_idx:
                index_train.append(entry)
            else:
                index_val.append(entry)


def build_dataset(root_dir, out_dir):
    root_dir = Path(root_dir)
    index_train, index_val = [], []

    for session_dir in root_dir.iterdir():
        if session_dir.is_dir() and (session_dir / "pose-2d").exists():
            build_session(session_dir, out_dir, index_train, index_val)

    # Save JSON indices
    out_dir = Path(out_dir)
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "splits" / "train.json", "w") as f:
        json.dump(index_train, f, indent=2)
    with open(out_dir / "splits" / "val.json", "w") as f:
        json.dump(index_val, f, indent=2)

    print(f"Saved {len(index_train)} train samples, {len(index_val)} val samples")


if __name__ == "__main__":
    root_input = r"base_path"
    output = "data"
    build_dataset(root_input, output)
