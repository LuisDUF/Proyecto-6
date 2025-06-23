import cv2
import numpy as np
import mediapipe as mp
import os
from functools import lru_cache

# === CONFIGURACIÓN ===
POSES_DIR = "Proyect - 06/Skeleton v2/poses_guardadas"  # Carpeta donde guardar los archivos .npy
os.makedirs(POSES_DIR, exist_ok=True)

# === INICIALIZACIÓN DE MEDIAPIPE ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# === FUNCIONES AUXILIARES ===

# Obtener keypoints normalizados
def get_keypoints(frame, pose_model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(frame_rgb)
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark])
    return None

# Comparar dos poses (promedio de distancias punto a punto)
def compare_pose(pose1, pose2):
    if pose1 is None or pose2 is None or pose1.shape != pose2.shape:
        return float('inf')
    return np.mean(np.linalg.norm(pose1 - pose2, axis=1))

# Extraer keypoints promedio de un video
@lru_cache(maxsize=3)
def extract_pose_from_video(path):
    cap = cv2.VideoCapture(path)
    all_keypoints = []

    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      enable_segmentation=False, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose_model:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))  # Reducción opcional
            keypoints = get_keypoints(frame, pose_model)
            if keypoints is not None:
                all_keypoints.append(keypoints)

    cap.release()
    return np.mean(all_keypoints, axis=0) if all_keypoints else None

# Guardar los keypoints promedio en archivo .npy
def save_pose(keypoints, name):
    path = os.path.join(POSES_DIR, f"{name}.npy")
    np.save(path, keypoints)

# Cargar los keypoints promedio desde archivo .npy
def load_pose(name):
    path = os.path.join(POSES_DIR, f"{name}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None

# Obtener la pose desde archivo o procesarla si no existe
def get_or_extract_pose(name, video_path):
    pose = load_pose(name)
    if pose is None:
        print(f"Procesando {name} desde {video_path}...")
        pose = extract_pose_from_video(video_path)
        if pose is not None:
            save_pose(pose, name)
    else:
        print(f"{name} cargado desde archivo.")
    return pose

# Dibujar puntos clave en el frame
def draw_keypoints(frame, keypoints):
    h, w, _ = frame.shape
    for x, y in keypoints:
        cv2.circle(frame, (int(x * w), int(y * h)), 5, (0, 255, 0), -1)

# === CARGA DE POSES DE REFERENCIA ===
print("Cargando poses de referencia...")
walking_pose = get_or_extract_pose("caminando", "walking.mp4")
sitting_pose = get_or_extract_pose("sentado", "sit.mp4")
suffering_pose = get_or_extract_pose("sufriendo", "suffering.mp4")

reference_poses = {
    "Caminando": walking_pose,
    "Sentado": sitting_pose,
    "Sufriendo": suffering_pose
}

# === DETECCIÓN EN TIEMPO REAL DESDE VIDEO O CÁMARA ===
cap = cv2.VideoCapture(1)  # Cambia a 0 para usar la cámara

with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                  enable_segmentation=False, min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose_model:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))  # Reducción opcional
        keypoints = get_keypoints(frame, pose_model)
        action = "No detectado"

        if keypoints is not None:
            # Comparar con poses de referencia
            distances = {label: compare_pose(keypoints, ref_pose)
                         for label, ref_pose in reference_poses.items()}
            action = min(distances, key=distances.get)

        # Mostrar resultados
        output_frame = frame.copy()
        if keypoints is not None:
            draw_keypoints(output_frame, keypoints)
        cv2.putText(output_frame, f'Acción: {action}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow('Detección de Acción', output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
