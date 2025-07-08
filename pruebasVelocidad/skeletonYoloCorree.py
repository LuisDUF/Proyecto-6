import cv2
import numpy as np
import os
from ultralytics import YOLO
import lmstudio as lms
import torch #para ejecutar con gpu
from PIL import Image
contador = 0

# === CONFIGURACIÓN ===
POSES_DIR = "..\Proyect - 06\Skeleton v2 - YOLO\poses_guardadas"
os.makedirs(POSES_DIR, exist_ok=True)

device = torch.device("cuda")

model = YOLO("yolov8s-pose.pt").to(device)  # Cambia por yolov8n-pose.pt si deseas

# === CONTROL DE VISUALIZACIÓN ===
mostrar_puntos = True  # 🔄 CAMBIA A False PARA OCULTAR PUNTOS

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
walking_pose = get_or_extract_pose("caminando", "walking.mp4")
sitting_pose = get_or_extract_pose("sentado", "sit.mp4")

reference_poses = {
    "Caminando": walking_pose,
    "Sentado": sitting_pose,
}

# === DETECCIÓN EN TIEMPO REAL ===

#plaza mexico
#cap = cv2.VideoCapture("https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1751511017/ei/iZtlaMvZOoWOpfgPup7KyAo/ip/2806:266:487:b56:42b:e097:c410:5f98/id/e9T0L_POAOk.36/itag/96/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D137/rqh/1/hls_chunk_host/rr5---sn-9gv7zn7e.googlevideo.com/xpc/EgVo2aDSNQ%3D%3D/playlist_duration/30/manifest_duration/30/bui/AY1jyLM2DVJUQj0ttKhqIxfnV0QpwUMdr2SyYCMXvOtvkOxTj0Lw_5PbGa7PD4xh2CPa5AgtrGjf91Rr/spc/l3OVKQS-avdmZVwGqtYw-Wl6a5NQUiKsPEUlc1-7QpMZDZ4F-jziAjgS6LzZgABsYpywthn5cEM/vprv/1/playlist_type/DVR/initcwndbps/1992500/met/1751489419,/mh/k_/mm/44/mn/sn-9gv7zn7e/ms/lva/mv/m/mvi/5/pl/49/rms/lva,lva/dover/11/pacing/0/keepalive/yes/fexp/51355912/mt/1751489072/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,rqh,xpc,playlist_duration,manifest_duration,bui,spc,vprv,playlist_type/sig/AJfQdSswRAIgeZogZ3_Dt_8GkseMxP7HkKs8K7JCQJU4BqYTSPFhkMkCIGiCCZUmehLN_27l3_mZQWqa-UIXD48s2ERvX5gpP-n2/lsparams/hls_chunk_host,initcwndbps,met,mh,mm,mn,ms,mv,mvi,pl,rms/lsig/APaTxxMwRgIhAPuRvPQQjr19SNuhT0xMJzQH-cGBDlot3jNFltNUD7DWAiEA19lrfnUvmlARgk5A_cqJdmf6QWGkwtNdUH1PyFi7N5g%3D/playlist/index.m3u8")  # Cambia a archivo si deseas usar video
#viedo tienda https://www.youtube.com/watch?v=6MMXJrzT5c0
cap = cv2.VideoCapture("vel.mp4")
#partido
#cap = cv2.VideoCapture("partido.mp4")

"""
Control de velocidad 
<==>
"""
fps = cap.get(cv2.CAP_PROP_FPS)  #PARA QUE SE REPRODUSCA NORMAL 
wait_time = int(1000 / fps) if fps > 0 else 33
"""
<==>
"""


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
        

            
        # Ver si hay 2 o más personas juntas
        # Calcula la distancia entre el centro de cada persona detectada
        centros = []
        for kp_person in results.keypoints.xy:
            kp_np = kp_person.cpu().numpy()
            centro = np.mean(kp_np, axis=0)
            centros.append(centro)

        # Si hay más de una persona, verifica distancias
        if len(centros) > 1:
            for idx1, c1 in enumerate(centros):
                for idx2, c2 in enumerate(centros):
                    if idx1 < idx2:
                        dist = np.linalg.norm(c1 - c2)
                        if dist < 150:  # Puedes ajustar el umbral según el tamaño de la imagen
                            x, y = int((c1[0] + c2[0]) / 2), int((c1[1] + c2[1]) / 2)
                            cv2.putText(output, "Personas juntas", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
        # Índices de keypoints para piernas, pies y centro
        selected_indices = [15,16]  # Solo índices válidos para 17 keypoints
        centros = []
        for kp_person in results.keypoints.xy:
            kp_np = kp_person.cpu().numpy()
            # Selecciona solo los keypoints deseados
            selected_kps = kp_np[selected_indices]
            if selected_indices:
                avg_point = np.mean(selected_kps, axis=0)
                centros.append(avg_point)


        # Estimar velocidad de persona
        if not hasattr(compare_pose, "prev_centers"):
            compare_pose.prev_centers = [None] * len(centros)
            compare_pose.prev_times = [None] * len(centros)

        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        for idx, centro in enumerate(centros):
            prev_center = compare_pose.prev_centers[idx] if idx < len(compare_pose.prev_centers) else None
            prev_time = compare_pose.prev_times[idx] if idx < len(compare_pose.prev_times) else None

            speed = None
            if prev_center is not None and prev_time is not None:
                dist = np.linalg.norm(centro - prev_center)
                dt = current_time - prev_time
                if dt > 0:
                    speed = dist / dt  # píxeles por segundo

            x, y = int(centro[0]), int(centro[1])

            # Mostrar acción solo si la velocidad es mayor o igual a 5 px/s
            if speed is not None:
                
                if speed >= 10:
                    x1, y1, x2, y2 = map(int, results.boxes.xyxy[i])
                    cv2.putText(output, f'{action}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                else:
                    if results.boxes and len(results.boxes.xyxy) > i:
                        x1, y1, x2, y2 = map(int, results.boxes.xyxy[i])
                        cv2.putText(output, "Quieto", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)




            # Actualizar valores previos
            if idx < len(compare_pose.prev_centers):
                compare_pose.prev_centers[idx] = centro
                compare_pose.prev_times[idx] = current_time
            else:
                compare_pose.prev_centers.append(centro)
                compare_pose.prev_times.append(current_time)

    output_resized = cv2.resize(output, (800, 600))

    cv2.imshow("YOLOv8 Pose - Tiempo Real", output_resized)
    
    cv2.resizeWindow("YOLOv8 Pose - Tiempo Real",800,600)
    key = cv2.waitKey(wait_time) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        if wait_time == 0:
            wait_time = int(1000 / fps) if fps > 0 else 33
        else:
            wait_time = 0
    elif key == ord('w'):
        wait_time = 1  # Acelera el video
    else:
        wait_time = int(1000 / fps) if fps > 0 else 33

cap.release()
cv2.destroyAllWindows()
