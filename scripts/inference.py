import sys
import csv

sys.path.append("path/to/root")

import cv2
import mediapipe as mp
import numpy as np
import torch

from models.hybrid_resnet import HybridResnet
from torchvision import transforms

VIDEO_PATH = "path/to/video"
MODEL_PATH = "path/to/model"
OUTPUT_CSV = "predictions_model.csv"

JOINTS = [
    "WRIST",

    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",

    "INDEX_CMC",
    "INDEX_MCP",
    "INDEX_IP",
    "INDEX_TIP",

    "MIDDLE_CMC",
    "MIDDLE_MCP",
    "MIDDLE_IP",
    "MIDDLE_TIP", 

    "RING_CMC",
    "RING_MCP",
    "RING_IP",
    "RING_TIP",

    "PNIKY_CMC",
    "PINKY_MCP",
    "PINKY_IP",
    "PINKY_TIP",
]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Load PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridResnet(num_joints=21)
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
    )
)
model.to(device)
model.eval()

image_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def preprocess_keypoints_2d(kp2d):
    kp2d_tensor = torch.from_numpy(kp2d).float().unsqueeze(0).to(device)
    return kp2d_tensor


def infer_frame(image, kp2d):
    input_img = image_transform(image).unsqueeze(0).to(device)
    input_kp = preprocess_keypoints_2d(kp2d)
    with torch.no_grad():
        pred_3d = model(input_img, input_kp)
    return pred_3d.squeeze(0).cpu().numpy()


def get_hand_keypoints_2d(image):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        h, w, _ = image.shape
        landmarks = results.multi_hand_landmarks[0]
        keypoints = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            keypoints.append([x, y])
        return np.array(keypoints)
    else:
        return None


cap = cv2.VideoCapture(VIDEO_PATH)

frame_index = 0
fps = cap.get(cv2.CAP_PROP_FPS) or 30

with open(OUTPUT_CSV, mode="w", newline="") as csv_file:

    writer = csv.writer(csv_file)

    #Header del CSV
    writer.writerow([
        "frame",
        "timestamp",
        "joint_id",
        "joint_name",
        "x",
        "y",
        "z"
    ])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_index / fps

        kp2d = get_hand_keypoints_2d(frame)
        if kp2d is not None:
            pred_3d = infer_frame(frame, kp2d)

            for joint_id, joint in enumerate(pred_3d):
                x = float(joint[0])
                y = float(joint[1])
                z = float(joint[2])

                writer.writerow([
                    frame_index,
                    timestamp,
                    joint_id,
                    JOINTS[joint_id],
                    x,
                    y,
                    z
                ])

            print(
                f"Frame {frame_index} processed"
            )

        else:
            print(f"Frame {frame_index} - No hand detected")

        frame_index += 1

cap.release()
hands.close()

print(f"Video completado y csv guardado {OUTPUT_CSV}")
