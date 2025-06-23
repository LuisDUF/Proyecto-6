import cv2
import numpy as np
import os
from ultralytics import YOLO

# === CONFIGURACIÓN ===
POSES_DIR = "Proyect - 06\Skeleton v2 - YOLO\poses_guardadas"
os.makedirs(POSES_DIR, exist_ok=True)

model = YOLO("yolov8s-pose.pt")  # Cambia por yolov8n-pose.pt si deseas

# === CONTROL DE VISUALIZACIÓN ===
mostrar_puntos = False  # 🔄 CAMBIA A False PARA OCULTAR PUNTOS

# === FUNCIONES AUXILIARES ===

def normalize_keypoints(keypoints, frame_shape):
    h, w = frame_shape[:2]
    return keypoints / np.array([[w, h]])

def compare_pose(pose1, pose2):
    if pose1 is None or pose2 is None or pose1.shape != pose2.shape:
        return float('inf')
    return np.mean(np.linalg.norm(pose1 - pose2, axis=1))

def save_pose(keypoints, name):
    np.save(os.path.join(POSES_DIR, f"{name}.npy"), keypoints)

def load_pose(name):
    path = os.path.join(POSES_DIR, f"{name}.npy")
    return np.load(path) if os.path.exists(path) else None

def draw_keypoints(frame, keypoints, color=(0, 255, 0)):
    if not mostrar_puntos:
        return
    for x, y in keypoints:
        cv2.circle(frame, (int(x), int(y)), 5, color, -1)

def extract_pose_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    all_keypoints = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        if results.keypoints and len(results.keypoints.xy) > 0:
            keypoints = results.keypoints.xy[0].cpu().numpy()  # Primera persona
            norm_kps = normalize_keypoints(keypoints, frame.shape)
            all_keypoints.append(norm_kps)

    cap.release()
    return np.mean(all_keypoints, axis=0) if all_keypoints else None

def get_or_extract_pose(name, video_path):
    pose = load_pose(name)
    if pose is None:
        print(f"[⏳] Extrayendo pose de {video_path}...")
        pose = extract_pose_from_video(video_path)
        if pose is not None:
            save_pose(pose, name)
            print(f"[✔] Pose '{name}' guardada.")
        else:
            print(f"[✘] No se pudo extraer '{name}'.")
    else:
        print(f"[✔] Pose '{name}' cargada desde archivo.")
    return pose

# === CARGA DE POSES DE REFERENCIA ===
print("Cargando poses de referencia...")
walking_pose = get_or_extract_pose("caminando", "videos/walking.mp4")
sitting_pose = get_or_extract_pose("sentado", "videos/sit.mp4")
aim_pose = get_or_extract_pose("sentado", "videos/aim.mp4")

reference_poses = {
    "Caminando": walking_pose,
    "Sentado": sitting_pose,
    "Apuntando": aim_pose
}

# === DETECCIÓN EN TIEMPO REAL ===
cap = cv2.VideoCapture("video_02.mp4")  # Cambia a archivo si deseas usar video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    output = frame.copy()

    for i, kp in enumerate(results.keypoints.xy):
        keypoints = kp.cpu().numpy()
        keypoints_norm = normalize_keypoints(keypoints, frame.shape)

        distances = {label: compare_pose(keypoints_norm, ref_pose)
                     for label, ref_pose in reference_poses.items()}
        action = min(distances, key=distances.get)

        draw_keypoints(output, keypoints)

        if results.boxes and len(results.boxes.xyxy) > i:
            x1, y1, x2, y2 = map(int, results.boxes.xyxy[i])
            cv2.putText(output, f'{action}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("YOLOv8 Pose - Tiempo Real", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
