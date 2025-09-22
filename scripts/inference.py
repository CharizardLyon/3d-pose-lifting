import sys

sys.path.append("D:/3d-pose-lifting")

import cv2
import mediapipe as mp
import numpy as np
import torch
from models.hybrid_resnet import HybridResnet
from torchvision import transforms

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
        "../checkpoints-resnet18/best_model.pth",
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


video_path = "camA.mp4"
cap = cv2.VideoCapture(video_path)

frame_index = 0
fps = cap.get(cv2.CAP_PROP_FPS) or 30

while True:
    ret, frame = cap.read()
    if not ret:
        break

    kp2d = get_hand_keypoints_2d(frame)
    if kp2d is not None:
        pred_3d = infer_frame(frame, kp2d)
        print(f"Frame {frame_index} Prediction:\n", pred_3d)
    else:
        print(f"Frame {frame_index} - No hand detected")

    frame_index += 1

cap.release()
hands.close()
