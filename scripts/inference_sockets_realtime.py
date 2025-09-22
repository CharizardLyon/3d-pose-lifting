import sys
sys.path.append("D:/3d-pose-lifting")

import cv2
import mediapipe as mp
import numpy as np
import torch
import json
import socket
import time
from models.hybrid_resnet import HybridResnet
from torchvision import transforms

# =========================
# Config de cámara y ventana
# =========================
CAM_INDEX = 0            # 0, 1, 2...
TARGET_W, TARGET_H = 1080, 720  # resolución deseada (no siempre se respeta)
DESIRED_FPS = 60           # no todos los drivers lo respetan
SHOW_WINDOW = True

# =========================
# MediaPipe Hands
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# =========================
# Socket: Python CLIENTE (Unity es servidor)
# =========================
HOST = "127.0.0.1"
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print(f"Conectando a Unity en {HOST}:{PORT} ...")
for i in range(20):
    try:
        sock.connect((HOST, PORT))
        print("Conectado a Unity ✅")
        break
    except OSError as e:
        print(f"Reintento {i+1}/20: {e}")
        time.sleep(0.5)
else:
    raise RuntimeError("No se pudo conectar a Unity en 127.0.0.1:5005")

# =========================
# Modelo PyTorch
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridResnet(num_joints=21)
model.load_state_dict(
    torch.load(
        "../checkpoints/resnet/best_model.pth",
        map_location=device,
    )
)
model.to(device)
model.eval()

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def preprocess_keypoints_2d(kp2d):
    # kp2d: (21,2) en pixeles
    kp2d_tensor = torch.from_numpy(kp2d).float().unsqueeze(0).to(device)
    return kp2d_tensor

def infer_frame(image_bgr, kp2d):
    # Evita copias innecesarias; convertimos solo para el modelo
    input_img = image_transform(image_bgr).unsqueeze(0).to(device)
    input_kp = preprocess_keypoints_2d(kp2d)
    with torch.no_grad():
        pred_3d = model(input_img, input_kp)
    return pred_3d.squeeze(0).cpu().numpy()

def get_hand_keypoints_2d(image_bgr):
    # MediaPipe usa RGB
    results = hands.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        h, w, _ = image_bgr.shape
        landmarks = results.multi_hand_landmarks[0]
        keypoints = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            keypoints.append([x, y])
        return np.array(keypoints)  # (21,2)
    else:
        return None

def send_prediction(pred_3d):
    pred_list = pred_3d.tolist()
    msg = json.dumps(pred_list).encode("utf-8")
    length = len(msg)
    sock.sendall(length.to_bytes(4, "big") + msg)

# =========================
# Captura en tiempo real
# =========================
# En Windows, CAP_DSHOW suele reducir latencia
backend = cv2.CAP_DSHOW
cap = cv2.VideoCapture(CAM_INDEX, backend)

if not cap.isOpened():
    # Fallback sin backend explícito
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {CAM_INDEX}")

# Intento de fijar resolución y FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
cap.set(cv2.CAP_PROP_FPS,          DESIRED_FPS)

src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
print(f"[INFO] Cámara abierta: {CAM_INDEX}  {src_w}x{src_h} @ {src_fps:.1f} FPS (reportado)")

frame_index = 0
t_prev = time.perf_counter()
ema_fps = None
send_enabled = True  # alterna con tecla 's'

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Frame no disponible. Saliendo.")
            break

        # === Detección 2D + inferencia 3D ===
        kp2d = get_hand_keypoints_2d(frame)
        if kp2d is not None:
            pred_3d = infer_frame(frame, kp2d)
            if send_enabled:
                send_prediction(pred_3d)
            status = f"Frame {frame_index} - Mano OK - SEND={'ON' if send_enabled else 'OFF'}"
        else:
            status = f"Frame {frame_index} - No hand detected"

        # === Overlay (opcional): dibujar puntos 2D ===
        if kp2d is not None:
            for (x, y) in kp2d.astype(int):
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        # === FPS (instantáneo + suavizado) ===
        t_now = time.perf_counter()
        dt = t_now - t_prev
        t_prev = t_now
        inst_fps = (1.0 / dt) if dt > 0 else 0.0
        ema_fps = inst_fps if ema_fps is None else (0.9 * ema_fps + 0.1 * inst_fps)

        # === Ventana ===
        if SHOW_WINDOW:
            cv2.putText(frame, f"FPS: {ema_fps:.1f} (inst {inst_fps:.1f})",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, status, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("Realtime Hand Tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:     # ESC
                break
            elif key == ord('s'):
                send_enabled = not send_enabled
            elif key == ord('p'):
                print("[PAUSA] Presiona cualquier tecla para continuar...")
                cv2.waitKey(0)

        frame_index += 1

finally:
    # Liberar recursos
    try:
        hands.close()
    except:
        pass
    try:
        cap.release()
    except:
        pass
    try:
        cv2.destroyAllWindows()
    except:
        pass
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except:
        pass
    try:
        sock.close()
    except:
        pass
