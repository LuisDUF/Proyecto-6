import cv2
import numpy as np
import os
from ultralytics import YOLO
import tempfile
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
cap = cv2.VideoCapture("https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1751493994/ei/ClllaJzXIb36ruEPmpjciAE/ip/45.177.43.24/id/9FgfkpQGCdg.1/itag/94/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D135/rqh/1/hls_chunk_host/rr2---sn-v2uvxoa5jxnhm-hahe.googlevideo.com/xpc/EgVo2aDSNQ%3D%3D/playlist_duration/30/manifest_duration/30/bui/AY1jyLMB-zcc8SllMG9_B5NpHZZc9poQqU26nU3AZlCGi6hRxe5bGDTly0rMSnQqJuhYWgXOv-WcNH61/spc/l3OVKY1j20YUWsrxp9wEyrQpuYlfzRyBJ_EV8GSMxhe-RR_qooeYLNHu_Hlb7p1Qjiiknk0QtwM/vprv/1/playlist_type/DVR/initcwndbps/1233750/met/1751472396,/mh/pg/mm/44/mn/sn-v2uvxoa5jxnhm-hahe/ms/lva/mv/m/mvi/2/pl/24/rms/lva,lva/dover/11/pacing/0/keepalive/yes/fexp/51355912/mt/1751472038/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,rqh,xpc,playlist_duration,manifest_duration,bui,spc,vprv,playlist_type/sig/AJfQdSswRgIhAPxYIJx76PAULUu6GZlqZl0kuSRNTLTPd5ghMBDQx4tiAiEAuv8JkN7RNOEvxDU4ApgVOQIQR0GJj2MHshOXZGY9nlc%3D/lsparams/hls_chunk_host,initcwndbps,met,mh,mm,mn,ms,mv,mvi,pl,rms/lsig/APaTxxMwRAIgWwnp7UM1JAXovtT1Nm4dYYRktdPIj7O877hsAtm3TpYCIEUVRBk6ODK0BhseVQf0qYOh4e_NkdAxBBp9cxGuWy5D/playlist/index.m3u8")  # Cambia a archivo si deseas usar video
#viedo tienda https://www.youtube.com/watch?v=6MMXJrzT5c0
cap = cv2.VideoCapture("https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1751497977/ei/mWhlaJH6Brm6-L4P7NCL6Q8/ip/45.177.43.24/id/6MMXJrzT5c0.2/itag/96/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D137/rqh/1/hls_chunk_host/rr1---sn-v2uvxoa5jxnhm-hahe.googlevideo.com/xpc/EgVo2aDSNQ%3D%3D/playlist_duration/30/manifest_duration/30/bui/AY1jyLPpiswAzUrOFGNUy1kjqTk_A8CrPQsQVR5N_xCAR2FImyJpIVx29UjDpooY_ZAR3LOvJePURkzB/spc/l3OVKeSqEdzYM_KzCvTaOoUr5PX2D0MJHwnKmXgdKNsiRheBtCACmfEmqjJjVuSddXL9yOzugII/vprv/1/playlist_type/DVR/initcwndbps/948750/met/1751476378,/mh/GE/mm/44/mn/sn-v2uvxoa5jxnhm-hahe/ms/lva/mv/m/mvi/1/pl/24/rms/lva,lva/dover/11/pacing/0/keepalive/yes/fexp/51355912/mt/1751475875/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,rqh,xpc,playlist_duration,manifest_duration,bui,spc,vprv,playlist_type/sig/AJfQdSswRgIhAPlu_PlEcQbF6t7ExcMVtpwBf5KtANmQ2mmobDDR0DYxAiEAup7jsMFx0u0okFfM73IGdHv_4OudRnM0m-rEh09-8no%3D/lsparams/hls_chunk_host,initcwndbps,met,mh,mm,mn,ms,mv,mvi,pl,rms/lsig/APaTxxMwRQIgR2RtBF919DhiREuhHUXu8Ock4I82LxrVPaC4GruyXB4CIQDImVbvSoFFPzAtBNmPl8gnF4qw-LQK8bLXvJEtoCSTjQ%3D%3D/playlist/index.m3u8")
#partido
#cap = cv2.VideoCapture("partido.mp4")

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
        centers = []
        for kp_person in results.keypoints.xy:
            kp_np = kp_person.cpu().numpy()
            center = np.mean(kp_np, axis=0)
            centers.append(center)

        # Si hay más de una persona, verifica distancias
        if len(centers) > 1:
            for idx1, c1 in enumerate(centers):
                for idx2, c2 in enumerate(centers):
                    if idx1 < idx2:
                        dist = np.linalg.norm(c1 - c2)
                        if dist < 150:  # Puedes ajustar el umbral según el tamaño de la imagen
                            x, y = int((c1[0] + c2[0]) / 2), int((c1[1] + c2[1]) / 2)
                            cv2.putText(output, "Personas juntas", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Estimar velocidad de persona
        if not hasattr(compare_pose, "prev_centers"):
            compare_pose.prev_centers = [None] * len(centers)
            compare_pose.prev_times = [None] * len(centers)

        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        for idx, center in enumerate(centers):
            prev_center = compare_pose.prev_centers[idx] if idx < len(compare_pose.prev_centers) else None
            prev_time = compare_pose.prev_times[idx] if idx < len(compare_pose.prev_times) else None

            speed = None
            if prev_center is not None and prev_time is not None:
                dist = np.linalg.norm(center - prev_center)
                dt = current_time - prev_time
                if dt > 0:
                    speed = dist / dt  # píxeles por segundo

            x, y = int(center[0]), int(center[1])

            # Mostrar acción solo si la velocidad es mayor o igual a 5 px/s
            if speed is not None:
                if speed >= 0.5:
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
                compare_pose.prev_centers[idx] = center
                compare_pose.prev_times[idx] = current_time
            else:
                compare_pose.prev_centers.append(center)
                compare_pose.prev_times.append(current_time)

    output_resized = cv2.resize(output, (800, 600))

    cv2.imshow("YOLOv8 Pose - Tiempo Real", output_resized)
    
    cv2.resizeWindow("YOLOv8 Pose - Tiempo Real",800,600)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
