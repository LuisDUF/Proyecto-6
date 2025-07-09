import cv2
import numpy as np
import os
import torchvision.transforms as T
from ultralytics import YOLO
import torch #para ejecutar con gpu

tipo_modelo = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", tipo_modelo)
midas.eval()

cap = cv2.VideoCapture("students_01.mp4")
device = torch.device("cuda")
midas.to(device)
model = YOLO("yolov8s-pose.pt").to(device)

# Preparador del modelo
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if tipo_modelo == "DPT_Large" or tipo_modelo == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# === CONFIGURACIÓN ===
POSES_DIR = ".\poses_guardadas"
os.makedirs(POSES_DIR, exist_ok=True)

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
walking_pose = get_or_extract_pose("caminando_hori", "camiar_hori.mp4")
walkin_pose_old = get_or_extract_pose("caminando","walking.mp4")
aim_pose_hori = get_or_extract_pose("apuntando_hori","apuntar.mp4")
sitting_pose = get_or_extract_pose("sentado", "sit.mp4")
crouching_hori = get_or_extract_pose("agachar_hori","agachar_hori.mp4")

reference_poses = {
    "Caminando_hori": walking_pose,
    "Caminando_old": walkin_pose_old,
    "Sentado": sitting_pose,
    "Apuntando hori": aim_pose_hori,
    "Agachado hori": crouching_hori
}


fps = cap.get(cv2.CAP_PROP_FPS)  #Obtener fps del video
tiempo_espera = int(1000 / fps) if fps > 0 else 33 #Convertir fps a cuadros por milisegundo
contador = 0
cercania = False
distancias = []
while cap.isOpened():
    contador += 1
    ret, frame = cap.read()
    if not ret:
        break
    
    personas = model(frame, verbose=False)[0]
    output = frame.copy()
    if not hasattr(compare_pose, "datos_anteriores"):
        compare_pose.datos_anteriores = []  # Lista de tuplas: (centro_anterior, tiempo)

    tiempo_actual = cv2.getTickCount() / cv2.getTickFrequency()
    nuevos_datos_anteriores = []

    ids_usados = set()

    for i in range(len(personas.boxes)):
        box = personas.boxes[i]
        keypoints = personas.keypoints.xy[i].cpu()  
        keypoints_norm = normalize_keypoints(keypoints, frame.shape)

        distances = {label: compare_pose(keypoints_norm, ref_pose)
                     for label, ref_pose in reference_poses.items()}
        action = min(distances, key=distances.get)

        #draw_keypoints(output, keypoints)

        if(contador%1==0 and cercania):
            # Asegúrate de que la imagen esté en RGB
            input_image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            input_image_small = cv2.resize(input_image,(384,256))
            # Aplica transformaciones y agrega batch dimension
            input_tensor = transform(input_image_small).to(device)
            
            # Pasar por el modelo (sin usar gradientes)
            with torch.no_grad():
                prediction = midas(input_tensor)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=input_image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            # Convertir predicción a numpy y normalizar
            depth_map = prediction.cpu().numpy()
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_MAGMA)

            box = personas.boxes[i]
            #keypoints = personas.keypoints.xy[i]  # (17, 2) array de [x, y]
            #draw_keypoints(output,keypoints)
            cls = int(box.cls[0])

            # Coordenadas de la caja (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            h, w = depth_map.shape
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                # Recorta esa región en el depth map
                person_depth = depth_map[y1:y2, x1:x2]
                avg_depth = np.mean(person_depth)
                distancias.append(avg_depth)
                # Clasificación de distancia
                if avg_depth > 10:
                    label = "Cerca"
                    color = (0, 0, 255)
                else:
                    label = "Lejos"
                    color = (0, 255, 0)

                # Dibuja el resultado
                #cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                cv2.putText(output, label, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            

            
            arregloNP = keypoints.numpy()  # Convierte los keypoints a un arreglo de NumPy
            centro = np.mean(arregloNP, axis=0)  # Calcula el promedio de los keypoints
            centros.append(centro)
            if len(centros) > 1:
                for i1, c1 in enumerate(centros):  # Recorre cada centro
                    for j, c2 in enumerate(centros):  # Compara con todos los demás centros
                        if i1 < j:  # Evita comparar consigo mismo
                            dist = np.linalg.norm(c1 - c2)  # Calcula la distancia
                            #diferencia = np.linalg.norm(distancias[i]-distancias[j])
                            diferencia = abs(distancias[i1] - distancias[j])
                            if dist < 90 and float(diferencia)>1.14: #Modificable
                                # Calcula un punto intermedio entre las dos personas para mostrar el mensaje
                                x, y = int((c1[0] + c2[0]) / 2), int((c1[1] + c2[1]) / 2)
                                cv2.putText(output, "Personas juntas", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            #CALCULO DE VELOCIDAD
            keypoint_promedio = np.mean(arregloNP[[15,16]], axis=0) #Promedio de keypoints
            #convertir keypoint promedio a formato de torch (tensor necesario para vectorizar)
            centro_t = torch.tensor(keypoint_promedio, device='cuda' if torch.cuda.is_available() else 'cpu')
            
            if compare_pose.datos_anteriores:  
                #Obtener el centro anterior y convertir a tensor 
                centros_anteriores = torch.tensor([c for c, _ in compare_pose.datos_anteriores], device=centro_t.device)
                #Hacer resta para obtener distancia entre el centro actual y todos los anteriores
                dists = torch.norm(centros_anteriores - centro_t, dim=1)
                # Buscar el índice del centro anterior más cercano
                arreglo_ordenado = torch.argsort(dists)
                idCoincidente = None
                for id in arreglo_ordenado:
                    id = id.item()
                    if id not in ids_usados:
                        idCoincidente = id
                        ids_usados.add(id)
                        break
            else:
                idCoincidente = None

            velocidad = None
            
            if idCoincidente is not None:
                #obtener el centro y el tiempo del id coincidente
                centro_anterior, tiempo_anterior = compare_pose.datos_anteriores[idCoincidente]
                #resta del centro actual con el anterior y tiempo actual con tiempo anterior
                dist = torch.norm(centro_t - torch.tensor(centro_anterior, device=centro_t.device)).item()
                dt = tiempo_actual - tiempo_anterior
                if dt > 0: #evita division por 0
                    velocidad = dist / dt  # píxeles por segundo

            if velocidad is not None:
                if velocidad >= 12: #Umbral de velocidad
                    x1, y1, x2, y2 = map(int, personas.boxes.xyxy[i])
                    cv2.putText(output, f'{action}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                else:
                    if personas.boxes and len(personas.boxes.xyxy) > i:
                        x1, y1, x2, y2 = map(int, personas.boxes.xyxy[i])
                        cv2.putText(output, "Quieto", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Guardar para el siguiente output
            nuevos_datos_anteriores.append((keypoint_promedio, tiempo_actual))
    compare_pose.datos_anteriores = nuevos_datos_anteriores
                        
    output = cv2.resize(output, (800, 600))
    cv2.putText(output, "-C: Cercania",(580,510), cv2.FONT_HERSHEY_SIMPLEX, 0.7,  (255, 0, 0), 2)
    cv2.putText(output, "-W: Camara rapida",(580,530), cv2.FONT_HERSHEY_SIMPLEX, 0.7,  (255, 0, 0), 2)
    cv2.putText(output, "-Esp: Pausa",(580,550), cv2.FONT_HERSHEY_SIMPLEX, 0.7,  (255, 0, 0), 2)
    cv2.putText(output, "-Q: Salir",(580,570), cv2.FONT_HERSHEY_SIMPLEX, 0.7,  (0, 0, 255), 2)

    cv2.imshow("YOLOv8 Pose - Tiempo Real", output)
    
    cv2.resizeWindow("YOLOv8 Pose - Tiempo Real",800,600)
    centros = []
    distancias = []

    tecla = cv2.waitKey(1) & 0xFF #Esperar TIEMPO_ESPERA milisegundos una tecla
    if tecla == ord('q'):
        break
    elif tecla == ord(' '):
        if tiempo_espera == 0:
            tiempo_espera = int(1000 / fps) if fps > 0 else 33 #Fps del video original en milisegundos
        else:
            tiempo_espera = 0 #Pausar el video
    elif tecla == ord('w'):
        tiempo_espera = 1  # "Acelera" el video
    elif tecla == ord('c'):
        if(cercania):
            cercania = False
        else:
            cercania = True  # Activa cercanía
    else:
        tiempo_espera = int(1000 / fps) if fps > 0 else 33


cap.release()
cv2.destroyAllWindows()