import torch
import numpy as np
import cv2
from qai_hub_models.models.mediapipe_pose import Model

# Cargar modelo
print("Cargando modelo...")
model = Model.from_pretrained()
print("Modelo cargado.")

# Video local
video_path = "partido.mp4"  # Cambia esto por la ruta a tu video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fin del video o error.")
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img_np = img.astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_tensor = torch.tensor(img_np).unsqueeze(0)

    with torch.no_grad():
        detection = model.pose_detector(img_tensor)
    print("Detection:", type(detection))
    # Suponiendo que detection[0] son las cajas y detection[1] los keypoints
    boxes = detection[0]
    keypoints = detection[1]

    # Si hay detecciones
    if boxes is not None and len(boxes) > 0:
        print(f"Detecciones: {len(boxes)}")
        # Dibuja keypoints para la primera persona detectada
        for i in range(min(len(keypoints), len(boxes))):
            kps = keypoints[i].cpu().numpy().flatten()
            # Prueba primero con pares (x, y)
            num_points = len(kps) // 2
            for j in range(num_points):
                x = int(kps[2*j] / 256 * frame.shape[1])
                y = int(kps[2*j+1] / 256 * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    #else:
        #print("No se detectaron personas en este frame.")

    cv2.imshow("Video con keypoints", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
