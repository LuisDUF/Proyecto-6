import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
from sklearn.neighbors import KNeighborsClassifier
import joblib
from ultralytics.utils.plotting import colors, Annotator
from ultralytics.data.augment import LetterBox

# === CONFIGURACIÓN ===
POSES_DIR = "Proyect - 06/Skeleton v2 - YOLO/poses_guardadas"
os.makedirs(POSES_DIR, exist_ok=True)
s_mode = 0  # 0: semántico, 1: instancia

# === DISPOSITIVO (GPU si está disponible) ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Usando dispositivo: {device.upper()}")

# === CARGA DE MODELOS ===
pose_model = YOLO("yolov8s-pose.pt").to(device)
seg_model = YOLO("yolo11s-seg.pt").to(device)

mostrar_puntos = True
mostrar_segmentacion = True
usar_comparacion_npy = True  # False para usar clasificador automático

# === CLASIFICADOR (opcional) ===
if not usar_comparacion_npy:
    clf = joblib.load("clasificador_acciones.joblib")

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

def extract_pose_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    all_keypoints = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_model(frame, verbose=False, device=device)[0]

        if results.keypoints and len(results.keypoints.xy) > 0:
            keypoints = results.keypoints.xy[0].cpu().numpy()
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
reference_poses = {}
if usar_comparacion_npy:
    print("Cargando poses de referencia...")
    walking_pose = get_or_extract_pose("caminando", "videos/walking.mp4")
    walkingup_pose = get_or_extract_pose("caminando_arriba", "Proyect - 06/videos/walking_up.mp4")
    walking01_pose = get_or_extract_pose("caminando_01", "Proyect - 06/videos/walking_01.mp4")

    reference_poses = {
        "Caminando": walking_pose,
        "Caminando Arriba": walkingup_pose,
        "Caminando 01": walking01_pose        
    }

# === DETECCIÓN EN TIEMPO REAL ===
cap = cv2.VideoCapture("Proyect - 06/videos/students_01.mp4")  # Usa 0 para cámara

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    output = frame.copy()

    # === SEGMENTACIÓN ===
    seg_results = seg_model.track(frame, persist=True, device=device, verbose=False)[0]
    annotator = Annotator(output)

    if seg_results.boxes.id is not None:
        boxes = seg_results.boxes.xyxy.tolist()
        tids = seg_results.boxes.id.int().tolist()
        clss = seg_results.boxes.cls.cpu().tolist()
        masks = seg_results.masks

        # Mostrar máscaras
        if mostrar_segmentacion and masks is not None and masks.data is not None:
            img = LetterBox(masks.shape[1:])(image=annotator.result())
            im_gpu = (torch.as_tensor(img, dtype=torch.float16, device=masks.data.device)
                      .permute(2, 0, 1).flip(0).contiguous() / 255)

            ids_or_classes = tids if s_mode == 1 else clss
            annotator.masks(masks.data, colors=[colors(x, True) for x in ids_or_classes], im_gpu=im_gpu)

        # === POSE ===
        pose_results = pose_model(frame, verbose=False, device=device)[0]

        for i, (b, t, c) in enumerate(zip(boxes, tids, clss)):
            if int(c) != 0:  # solo para "person"
                continue

            label = seg_model.names[c]  # "person"

            # Obtener keypoints y acción
            if pose_results.keypoints is not None and i < len(pose_results.keypoints.xy):
                keypoints = pose_results.keypoints.xy[i].cpu().numpy()
                keypoints_norm = normalize_keypoints(keypoints, frame.shape)

                if usar_comparacion_npy:
                    distances = {
                        lbl: compare_pose(keypoints_norm, ref)
                        for lbl, ref in reference_poses.items()
                    }
                    action = min(distances, key=distances.get)
                else:
                    flat = keypoints_norm.flatten().reshape(1, -1)
                    action = clf.predict(flat)[0]

                label += f" - {action}"

                # Dibujar puntos
                if mostrar_puntos:
                    for x, y in keypoints:
                        cv2.circle(output, (int(x), int(y)), 4, (0, 255, 0), -1)

            # Dibujar caja y etiqueta
            annotator.box_label(
                b,
                color=colors(t if s_mode == 1 else c, True),
                label=label
            )

        # === Mostrar conteo en pantalla ===
        total_personas = sum(1 for cl in clss if int(cl) == 0)
        cv2.putText(output, f"Personas detectadas: {total_personas}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("YOLOv8 Pose + Segmentación", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()