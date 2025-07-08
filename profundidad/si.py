import cv2
import numpy as np
import torchvision.transforms as T
from ultralytics import YOLO
import torch #para ejecutar con gpu

tipo_modelo = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", tipo_modelo)
midas.eval()

# Preparador del modelo
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if tipo_modelo == "DPT_Large" or tipo_modelo == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

cap = cv2.VideoCapture("students_01.mp4")
device = torch.device("cuda")
midas.to(device)
model = YOLO("yolov8s-pose.pt").to(device)

def normalize_keypoints(keypoints, frame_shape):
    h, w = frame_shape[:2]
    return keypoints / np.array([[w, h]])

def draw_keypoints(frame, keypoints, color=(0, 255, 0)):
    for x, y in keypoints:
        cv2.circle(frame, (int(x), int(y)), 5, color, -1)

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

    if(contador%1==0 and cercania):
        # Asegúrate de que la imagen esté en RGB
        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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


    personas = model(frame, verbose=False)[0]

    for i in range(len(personas.boxes)):
        box = personas.boxes[i]
        keypoints = personas.keypoints.xy[i]  # (17, 2) array de [x, y]
        #draw_keypoints(frame,keypoints)
        cls = int(box.cls[0])
        if(contador%1==0 and cercania):
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
                #cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if(contador%1==0 and cercania):
        centros = []

        for persona in personas.keypoints.xy:
            arregloNP = persona.cpu().numpy()  # Convierte los keypoints a un arreglo de NumPy
            centro = np.mean(arregloNP, axis=0)  # Calcula el promedio de los keypoints
            centros.append(centro)
        if len(centros) > 1:
            for i, c1 in enumerate(centros):  # Recorre cada centro
                for j, c2 in enumerate(centros):  # Compara con todos los demás centros
                    if i < j:  # Evita comparar consigo mismo
                        dist = np.linalg.norm(c1 - c2)  # Calcula la distancia
                        #diferencia = np.linalg.norm(distancias[i]-distancias[j])
                        diferencia = abs(distancias[i] - distancias[j])
                        if dist < 90: #Modificable
                            print(diferencia)
                            # Calcula un punto intermedio entre las dos personas para mostrar el mensaje
                            x, y = int((c1[0] + c2[0]) / 2), int((c1[1] + c2[1]) / 2)
                            cv2.putText(frame, "Personas juntas", (x, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
    frame = cv2.resize(frame, (800, 600))
    cv2.putText(frame, "-C: Cercania",(580,510), cv2.FONT_HERSHEY_SIMPLEX, 0.7,  (255, 0, 0), 2)
    cv2.putText(frame, "-W: Camara rapida",(580,530), cv2.FONT_HERSHEY_SIMPLEX, 0.7,  (255, 0, 0), 2)
    cv2.putText(frame, "-Esp: Pausa",(580,550), cv2.FONT_HERSHEY_SIMPLEX, 0.7,  (255, 0, 0), 2)
    cv2.putText(frame, "-Q: Salir",(580,570), cv2.FONT_HERSHEY_SIMPLEX, 0.7,  (0, 0, 255), 2)

    cv2.imshow("YOLOv8 Pose - Tiempo Real", frame)
    
    cv2.resizeWindow("YOLOv8 Pose - Tiempo Real",800,600)

    tecla = cv2.waitKey(tiempo_espera) & 0xFF #Esperar TIEMPO_ESPERA milisegundos una tecla
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